import math
import os

# Render / CPU 環境穩定設定：避免 TensorFlow 佔用過多執行緒
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import json
import numpy as np
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

# 保留 LSTM：有安裝 tensorflow-cpu 時會啟用；若環境還沒裝好，不會讓整個服務直接掛掉
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
    TF_IMPORT_ERROR = ""
except Exception as e:
    tf = None
    Sequential = None
    LSTM = Dense = Dropout = Input = None
    Adam = None
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(e)

from deepseek_client import DeepSeekClient

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not TF_AVAILABLE:
    logger.warning(f"TensorFlow 未啟用，LSTM 會暫時回傳 0.5。原因：{TF_IMPORT_ERROR}")
else:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

# ============ 環境變數 ============
B_PRIOR = float(os.getenv("B_PRIOR", "0.4586"))
P_PRIOR = float(os.getenv("P_PRIOR", "0.4462"))
T_PRIOR = float(os.getenv("T_PRIOR", "0.0952"))

# 模型權重（最終融合）
MARKOV_WEIGHT = float(os.getenv("MARKOV_WEIGHT", "0.26"))
ROAD_WEIGHT = float(os.getenv("ROAD_WEIGHT", "0.24"))
STREAK_WEIGHT = float(os.getenv("STREAK_WEIGHT", "0.18"))
BALANCE_WEIGHT = float(os.getenv("BALANCE_WEIGHT", "0.12"))
RECENT_WEIGHT = float(os.getenv("RECENT_WEIGHT", "0.10"))
TIE_WEIGHT = float(os.getenv("TIE_WEIGHT", "0.04"))
AI_BLEND = float(os.getenv("AI_BLEND", "0.16"))

# ML模型權重（在規律模型之後進行二次校準）
ML_WEIGHT = float(os.getenv("ML_WEIGHT", "0.25"))  # ML模型在最終決策中的權重

TIE_SHRINK = float(os.getenv("TIE_SHRINK", "0.35"))
TIE_MAX_PROB = float(os.getenv("TIE_MAX_PROB", "0.18"))
ALLOW_TIE_RECOMMEND = os.getenv("ALLOW_TIE_RECOMMEND", "0") == "1"
TIE_RECOMMEND_MIN = float(os.getenv("TIE_RECOMMEND_MIN", "0.165"))
MIN_HISTORY_FOR_AI = int(os.getenv("MIN_HISTORY_FOR_AI", "6"))
MIN_HISTORY_FOR_SIGNAL = int(os.getenv("MIN_HISTORY_FOR_SIGNAL", "4"))

# LSTM參數
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", "12"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "8"))
LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "8"))
ML_RETRAIN_INTERVAL = int(os.getenv("ML_RETRAIN_INTERVAL", "10"))

# ============ 全局模型實例（單例模式） ============
class MLModels:
    """機器學習模型容器（單例）"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # 初始化模型
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1
        )
        self.lr = LogisticRegression(
            max_iter=300,
            random_state=42,
            C=1.0
        )
        self.lstm = None
        self.scaler = StandardScaler()
        
        # 訓練狀態
        self.is_trained = False
        self.training_samples = 0
        self.last_training_history = []
        self.last_training_key = ""
        
        # Render 啟動穩定版：
        # 不在服務啟動/import predictor.py 時建立 LSTM，避免 uvicorn 卡住導致 Render 偵測不到 port。
        # LSTM 會在資料足夠並進入 train() 時才建立與訓練。
        # self._build_lstm()
    
    def _build_lstm(self):
        """建立 LSTM 模型架構（權重需訓練）"""
        if not TF_AVAILABLE:
            self.lstm = None
            return None

        self.lstm = Sequential([
            Input(shape=(LSTM_SEQUENCE_LENGTH, 1)),
            LSTM(48, return_sequences=True),
            Dropout(0.20),
            LSTM(24, return_sequences=False),
            Dropout(0.20),
            Dense(12, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.lstm.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.lstm

    def _encode_sequence(self, non_tie: List[str]) -> np.ndarray:
        """編碼牌路序列為數值"""
        mapping = {'B': 1, 'P': 0}
        return np.array([mapping.get(x, 0) for x in non_tie]).reshape(-1, 1)
    
    def _extract_features(self, non_tie: List[str]) -> np.ndarray:
        """提取ML特徵（修正版：無資料洩漏）"""
        if len(non_tie) < 6:
            return np.zeros((1, 12))
        
        n = len(non_tie)
        
        # 基本統計
        b_count = non_tie.count('B')
        p_count = n - b_count
        b_rate = b_count / n if n > 0 else 0.5
        
        # 近期趨勢
        recent = non_tie[-10:] if n >= 10 else non_tie
        recent_b_rate = recent.count('B') / len(recent) if len(recent) > 0 else 0.5
        
        # 轉換率
        if n >= 2:
            switches = sum(1 for i in range(1, n) if non_tie[i] != non_tie[i-1])
            switch_rate = switches / (n - 1)
        else:
            switch_rate = 0.5
        
        # 當前連莊
        current_streak = 1
        if n >= 2:
            for i in range(n-2, -1, -1):
                if non_tie[i] == non_tie[-1]:
                    current_streak += 1
                else:
                    break
        
        # 最大連莊
        max_streak = 1
        current = 1
        for i in range(1, n):
            if non_tie[i] == non_tie[i-1]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1
        
        # 最近5局
        last_5 = non_tie[-5:] if n >= 5 else non_tie
        last_5_b = last_5.count('B') / len(last_5) if len(last_5) > 0 else 0.5
        
        # 最近3局
        last_3 = non_tie[-3:] if n >= 3 else non_tie
        last_3_b = last_3.count('B') / len(last_3) if len(last_3) > 0 else 0.5
        
        # 特徵向量（12個特徵）
        features = np.array([[
            b_rate,                          # 1. 整體莊家率
            recent_b_rate,                   # 2. 近期莊家率
            switch_rate,                     # 3. 轉換率
            current_streak / max(10, n),     # 4. 當前連莊（正規化）
            max_streak / max(10, n),         # 5. 最大連莊（正規化）
            last_5_b,                        # 6. 最近5局莊家率
            last_3_b,                        # 7. 最近3局莊家率
            b_count / max(10, n),            # 8. 莊家總數（正規化）
            p_count / max(10, n),            # 9. 玩家總數（正規化）
            1 if non_tie[-1] == 'B' else 0,  # 10. 上一局結果
            (b_count - p_count) / max(10, n), # 11. 莊家優勢
            n / 100                          # 12. 樣本數（正規化）
        ]])
        
        return features
    
    def _prepare_lstm_data(self, non_tie: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """準備LSTM序列資料"""
        if len(non_tie) < LSTM_SEQUENCE_LENGTH + 1:
            return np.array([]), np.array([])
        
        encoded = self._encode_sequence(non_tie)
        X, y = [], []
        
        for i in range(LSTM_SEQUENCE_LENGTH, len(encoded)):
            X.append(encoded[i-LSTM_SEQUENCE_LENGTH:i, 0])
            y.append(encoded[i, 0])
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        return np.array(X).reshape(-1, LSTM_SEQUENCE_LENGTH, 1), np.array(y)
    
    def train(self, non_tie: List[str], training_key: str = "") -> Dict[str, Any]:
        """訓練所有 ML 模型：LR + RF + LSTM（有 TensorFlow 才啟用）"""
        if len(non_tie) < 30:
            return {
                "status": "error",
                "message": f"需要至少30局歷史資料，目前{len(non_tie)}局"
            }

        try:
            # 1. 提取特徵（滾動窗口，避免拿未來資料訓練）
            X_features = []
            y_labels = []

            for i in range(12, len(non_tie)):
                features = self._extract_features(non_tie[:i])
                X_features.append(features[0])
                y_labels.append(1 if non_tie[i] == 'B' else 0)

            X_features = np.array(X_features)
            y_labels = np.array(y_labels)

            if len(X_features) < 10:
                return {"status": "error", "message": "有效訓練樣本不足"}

            if len(set(y_labels.tolist())) < 2:
                return {"status": "error", "message": "訓練資料只有單一類別，暫不訓練 ML"}

            # 2. 正規化 + 訓練 Logistic Regression / Random Forest
            X_scaled = self.scaler.fit_transform(X_features)
            self.lr.fit(X_scaled, y_labels)
            self.rf.fit(X_scaled, y_labels)

            # 3. 訓練 LSTM（保留，但避免 TensorFlow 未安裝時讓服務掛掉）
            lstm_status = "disabled"
            if TF_AVAILABLE:
                X_lstm, y_lstm = self._prepare_lstm_data(non_tie)
                if len(X_lstm) > 10 and len(set(y_lstm.tolist())) >= 2:
                    self._build_lstm()  # 每次重訓時重置 LSTM
                    callbacks = [
                        tf.keras.callbacks.EarlyStopping(
                            patience=3,
                            restore_best_weights=True
                        )
                    ]
                    self.lstm.fit(
                        X_lstm,
                        y_lstm,
                        epochs=LSTM_EPOCHS,
                        batch_size=LSTM_BATCH_SIZE,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=callbacks
                    )
                    lstm_status = "trained"
                else:
                    lstm_status = "not_enough_sequence"
            else:
                self.lstm = None
                lstm_status = f"tensorflow_unavailable: {TF_IMPORT_ERROR}"

            self.is_trained = True
            self.training_samples = len(X_features)
            self.last_training_history = list(non_tie)
            self.last_training_key = training_key

            return {
                "status": "success",
                "samples": self.training_samples,
                "lstm_status": lstm_status,
                "message": "ML模型訓練完成"
            }

        except Exception as e:
            logger.error(f"ML訓練錯誤: {e}")
            return {"status": "error", "message": str(e)}

    def predict(self, non_tie: List[str]) -> Dict[str, float]:
        """使用ML模型預測"""
        default_result = {
            'lr': 0.5,
            'rf': 0.5,
            'lstm': 0.5,
            'ensemble': 0.5
        }
        
        if len(non_tie) < 12 or not self.is_trained:
            return default_result
        
        try:
            # 提取特徵
            features = self._extract_features(non_tie)
            features_scaled = self.scaler.transform(features)
            
            predictions = {}
            
            # Logistic Regression
            try:
                lr_prob = self.lr.predict_proba(features_scaled)[0][1]
                predictions['lr'] = float(lr_prob)
            except:
                predictions['lr'] = 0.5
            
            # Random Forest
            try:
                rf_prob = self.rf.predict_proba(features_scaled)[0][1]
                predictions['rf'] = float(rf_prob)
            except:
                predictions['rf'] = 0.5
            
            # LSTM
            try:
                if self.lstm is not None and len(non_tie) >= LSTM_SEQUENCE_LENGTH:
                    encoded = self._encode_sequence(non_tie[-LSTM_SEQUENCE_LENGTH:])
                    X_lstm = np.array(encoded).reshape(1, LSTM_SEQUENCE_LENGTH, 1)
                    lstm_prob = float(self.lstm.predict(X_lstm, verbose=0)[0][0])
                    predictions['lstm'] = lstm_prob
                else:
                    predictions['lstm'] = 0.5
            except:
                predictions['lstm'] = 0.5
            
            # 集成預測（加權平均）
            weights = {'lr': 0.25, 'rf': 0.35, 'lstm': 0.40}
            ensemble = sum(predictions[k] * weights[k] for k in weights)
            predictions['ensemble'] = float(ensemble)
            
            return predictions
            
        except Exception as e:
            logger.error(f"ML預測錯誤: {e}")
            return default_result

# 全局實例
ml_models = MLModels()

# ============ 輔助函數 ============
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _normalize_three(b: float, p: float, t: float) -> Tuple[float, float, float]:
    b = max(0.001, b)
    p = max(0.001, p)
    t = max(0.001, min(TIE_MAX_PROB, t))
    s = b + p + t
    return b / s, p / s, t / s

def _last_non_tie(history: List[str]) -> List[str]:
    return [x for x in history if x in {"B", "P"}]

def _streak(non_tie: List[str]) -> Tuple[str, int]:
    if not non_tie:
        return "", 0
    last = non_tie[-1]
    n = 1
    for x in reversed(non_tie[:-1]):
        if x == last:
            n += 1
        else:
            break
    return last, n

def _transition_prob(non_tie: List[str]) -> Dict[str, float]:
    counts = defaultdict(lambda: Counter())
    for a, b in zip(non_tie, non_tie[1:]):
        counts[a][b] += 1
    if not non_tie:
        return {"B": 0.5, "P": 0.5, "sample": 0}
    last = non_tie[-1]
    c = counts[last]
    sample = c["B"] + c["P"]
    alpha = float(os.getenv("MARKOV_ALPHA", "2.4"))
    b = (c["B"] + alpha) / (sample + 2 * alpha)
    p = (c["P"] + alpha) / (sample + 2 * alpha)
    shrink = min(1.0, sample / float(os.getenv("MARKOV_FULL_SAMPLE", "14")))
    b = 0.5 * (1 - shrink) + b * shrink
    p = 0.5 * (1 - shrink) + p * shrink
    return {"B": b, "P": p, "sample": sample}

def _road_pattern_score(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 3:
        return {"B": 0.5, "P": 0.5, "label": "資料不足", "strength": 0.0}

    last, streak_n = _streak(non_tie)
    opp = "P" if last == "B" else "B"
    recent = non_tie[-12:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)

    b = p = 0.5
    label = "混合盤"
    strength = 0.08

    if streak_n >= 5:
        cont = 0.53 + min(0.05, (streak_n - 5) * 0.008)
        b = cont if last == "B" else 1 - cont
        p = cont if last == "P" else 1 - cont
        label = f"長龍{last}{streak_n}"
        strength = 0.18
    elif switch_rate >= 0.72 and len(recent) >= 6:
        b = 0.57 if opp == "B" else 0.43
        p = 0.57 if opp == "P" else 0.43
        label = "跳路偏強"
        strength = 0.16
    elif len(non_tie) >= 6 and non_tie[-6:] in [list("BBPPBB"), list("PPBBPP")]:
        next_side = non_tie[-2]
        b = 0.56 if next_side == "B" else 0.44
        p = 0.56 if next_side == "P" else 0.44
        label = "雙跳/兩房型"
        strength = 0.15
    elif len(non_tie) >= 8:
        key = "".join(non_tie[-4:])
        follows = []
        seq = "".join(non_tie)
        for i in range(0, len(seq) - 4):
            if seq[i:i + 4] == key and i + 4 < len(seq):
                follows.append(seq[i + 4])
        if follows:
            c = Counter(follows)
            total = c["B"] + c["P"]
            b_raw = c["B"] / total
            shrink = min(0.75, total / 10)
            b = 0.5 * (1 - shrink) + b_raw * shrink
            p = 1 - b
            label = f"四碼回測{key}"
            strength = min(0.20, 0.08 + total * 0.015)
        else:
            b_count = recent.count("B")
            p_count = recent.count("P")
            if abs(b_count - p_count) >= 4:
                scarce = "B" if b_count < p_count else "P"
                b = 0.54 if scarce == "B" else 0.46
                p = 0.54 if scarce == "P" else 0.46
                label = "短窗均衡修正"
                strength = 0.10

    return {"B": b, "P": p, "label": label, "strength": strength, "switch_rate": switch_rate, "streak": streak_n}

def _recent_score(non_tie: List[str]) -> Dict[str, float]:
    if not non_tie:
        return {"B": 0.5, "P": 0.5}
    recent = non_tie[-10:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)
    last, n = _streak(non_tie)
    opp = "P" if last == "B" else "B"
    if switch_rate > 0.66:
        side = opp
        edge = 0.055
    elif n >= 3:
        side = last
        edge = 0.045 + min(0.025, (n - 3) * 0.008)
    else:
        b_count = recent.count("B")
        p_count = recent.count("P")
        side = "B" if b_count < p_count else "P"
        edge = min(0.035, abs(b_count - p_count) * 0.006)
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}

def _balance_score(non_tie: List[str]) -> Dict[str, float]:
    if len(non_tie) < 8:
        return {"B": 0.5, "P": 0.5}
    b = non_tie.count("B")
    p = non_tie.count("P")
    diff = b - p
    edge = min(0.055, abs(diff) / max(1, len(non_tie)) * 0.16)
    side = "B" if diff < 0 else "P"
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}

def _streak_score(non_tie: List[str]) -> Dict[str, float]:
    last, n = _streak(non_tie)
    if not last:
        return {"B": 0.5, "P": 0.5}
    opp = "P" if last == "B" else "B"
    if n == 1:
        side, edge = opp, 0.025
    elif n == 2:
        side, edge = last, 0.030
    elif n == 3:
        side, edge = last, 0.045
    elif n >= 4:
        side, edge = last, min(0.075, 0.050 + (n - 4) * 0.008)
    else:
        side, edge = last, 0.0
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}

def _tie_score(history: List[str]) -> float:
    if not history:
        return T_PRIOR
    recent = history[-18:]
    t_rate = recent.count("T") / len(recent)
    gap_since_tie = 0
    for x in reversed(history):
        if x == "T":
            break
        gap_since_tie += 1
    pressure = T_PRIOR * (1 - TIE_SHRINK) + t_rate * TIE_SHRINK
    if gap_since_tie >= 18:
        pressure += 0.012
    if recent[-4:].count("T") >= 2:
        pressure += 0.018
    return _clamp(pressure, 0.055, TIE_MAX_PROB)

def _confidence(b: float, p: float, t: float, history_len: int, agreement: float, ml_agreement: float = 0.0) -> Tuple[float, str]:
    gap = abs(b - p)
    base = gap * 3.6 + agreement * 0.22 + ml_agreement * 0.10 + min(0.16, history_len / 80)
    conf = _clamp(base, 0.08, 0.94)
    if history_len < MIN_HISTORY_FOR_SIGNAL:
        return min(conf, 0.35), "冷啟動"
    if conf >= 0.68:
        return conf, "強訊號"
    if conf >= 0.48:
        return conf, "中訊號"
    return conf, "弱訊號"

# ============ 主要預測函數 ============
def predict(history: List[str], venue: str = "", room: str = "", shoe_id: str = "") -> Dict[str, Any]:
    """
    整合預測函數：規律模型 + ML模型 + DeepSeek校準
    """
    history = [x.upper() for x in history if x.upper() in {"B", "P", "T"}]
    non_tie = _last_non_tie(history)
    
    # ============ 1. 規律模型（原有邏輯） ============
    markov = _transition_prob(non_tie)
    road = _road_pattern_score(non_tie)
    recent = _recent_score(non_tie)
    balance = _balance_score(non_tie)
    streak = _streak_score(non_tie)
    
    total_w = MARKOV_WEIGHT + ROAD_WEIGHT + STREAK_WEIGHT + BALANCE_WEIGHT + RECENT_WEIGHT
    b_side = (
        markov["B"] * MARKOV_WEIGHT
        + road["B"] * ROAD_WEIGHT
        + streak["B"] * STREAK_WEIGHT
        + balance["B"] * BALANCE_WEIGHT
        + recent["B"] * RECENT_WEIGHT
    ) / total_w
    p_side = 1 - b_side
    
    tie_prob = _tie_score(history)
    b_prob = b_side * (1 - tie_prob)
    p_prob = p_side * (1 - tie_prob)
    
    # ============ 2. ML模型預測 ============
    # 每個場館 / 房間 / 靴號建立獨立 key，避免不同桌互相污染
    training_key = f"{venue}|{room}|{shoe_id}" if (venue or room or shoe_id) else "global"

    # 資料足夠才訓練；換桌/換靴或新增資料達門檻才重訓，避免每局都重訓拖慢 Render
    should_train = (
        len(non_tie) >= 30
        and (
            not ml_models.is_trained
            or getattr(ml_models, "last_training_key", "") != training_key
            or len(non_tie) - len(getattr(ml_models, "last_training_history", [])) >= ML_RETRAIN_INTERVAL
        )
    )

    if should_train:
        train_result = ml_models.train(non_tie, training_key=training_key)
        logger.info(f"ML訓練結果: {train_result}")

    # 訓練後再預測，避免剛訓練完還拿到舊的 0.5
    ml_pred = ml_models.predict(non_tie)
    ml_b_prob = ml_pred.get('ensemble', 0.5)

    # 如果 ML 模型已訓練，使用 ML 預測做小幅修正；避免 ML 過度壓過牌路模型
    if ml_models.is_trained:
        ml_weight = ML_WEIGHT * (0.5 + 0.5 * min(1.0, ml_models.training_samples / 50))
        b_prob = b_prob * (1 - ml_weight) + ml_b_prob * ml_weight
        p_prob = p_prob * (1 - ml_weight) + (1 - ml_b_prob) * ml_weight

    # ============ 3. DeepSeek校準 ============
    feature_payload = {
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "history_len": len(history),
        "history_tail": "".join(history[-36:]),
        "non_tie_tail": "".join(non_tie[-36:]),
        "markov": markov,
        "road": road,
        "recent": recent,
        "balance": balance,
        "streak": streak,
        "ml_predictions": ml_pred,
        "tf_available": TF_AVAILABLE,
        "training_key": training_key,
        "local_probs": {"B": round(b_prob, 5), "P": round(p_prob, 5), "T": round(tie_prob, 5)},
    }
    
    ai_result = None
    if len(history) >= MIN_HISTORY_FOR_AI and AI_BLEND > 0:
        ai_result = DeepSeekClient().calibrate(feature_payload)
        if ai_result and not ai_result.get("error"):
            try:
                ba = _clamp(float(ai_result.get("banker_adjust", 0)), -0.035, 0.035)
                pa = _clamp(float(ai_result.get("player_adjust", 0)), -0.035, 0.035)
                ta = _clamp(float(ai_result.get("tie_adjust", 0)), -0.020, 0.020)
                ai_conf = _clamp(float(ai_result.get("confidence", 0.4)), 0, 1)
                blend = AI_BLEND * (0.45 + ai_conf * 0.55)
                b_prob += ba * blend
                p_prob += pa * blend
                tie_prob += ta * blend
            except Exception:
                pass
    
    # ============ 4. 正規化 ============
    b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)
    
    # ============ 5. 投票一致性 ============
    votes = [
        "B" if markov["B"] >= markov["P"] else "P",
        "B" if road["B"] >= road["P"] else "P",
        "B" if streak["B"] >= streak["P"] else "P",
        "B" if balance["B"] >= balance["P"] else "P",
        "B" if recent["B"] >= recent["P"] else "P",
    ]
    if ml_models.is_trained:
        votes.append("B" if ml_b_prob >= 0.5 else "P")
    
    main_pick = "B" if b_prob >= p_prob else "P"
    agreement = votes.count(main_pick) / len(votes)
    
    # ML一致性
    ml_agreement = 1.0 - abs(b_prob - 0.5) * 2 if ml_models.is_trained else 0.0
    
    # ============ 6. 推薦與信心 ============
    if ALLOW_TIE_RECOMMEND and tie_prob >= TIE_RECOMMEND_MIN and tie_prob > max(b_prob, p_prob) * 0.55:
        recommend = "T"
    else:
        recommend = main_pick
    
    conf, level = _confidence(b_prob, p_prob, tie_prob, len(history), agreement, ml_agreement)
    
    # ============ 7. 原因說明 ============
    reason_parts = [road.get("label", "牌路"), f"一致{int(agreement * 100)}%"]
    if ml_models.is_trained:
        reason_parts.append(f"ML集體{int(ml_b_prob*100)}%")
    if ai_result and ai_result.get("pattern_label"):
        reason_parts.append(f"AI:{ai_result.get('pattern_label')}")
    elif ai_result and ai_result.get("error"):
        reason_parts.append("AI離線改本地判斷")
    
    # ============ 8. 返回結果 ============
    return {
        "ok": True,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "round_no": len(history) + 1,
        "history_len": len(history),
        "banker_rate": round(b_prob * 100, 1),
        "player_rate": round(p_prob * 100, 1),
        "tie_rate": round(tie_prob * 100, 1),
        "recommend": recommend,
        "recommend_text": {"B": "莊", "P": "閒", "T": "和"}[recommend],
        "confidence": round(conf, 3),
        "signal_level": level,
        "pattern_label": road.get("label", ""),
        "reason": " / ".join(reason_parts),
        "ai_used": bool(ai_result and not ai_result.get("error")),
        "ml_trained": ml_models.is_trained,
        "ml_samples": ml_models.training_samples,
        "tf_available": TF_AVAILABLE,
        "ml_predictions": {
            "lr": round(ml_pred.get('lr', 0.5), 4),
            "rf": round(ml_pred.get('rf', 0.5), 4),
            "lstm": round(ml_pred.get('lstm', 0.5), 4),
            "ensemble": round(ml_pred.get('ensemble', 0.5), 4)
        } if ml_models.is_trained else None,
        "ai_result": ai_result if os.getenv("DEBUG_AI_RESULT", "0") == "1" else None,
        "debug": feature_payload if os.getenv("DEBUG_PREDICTOR", "0") == "1" else None,
    }
