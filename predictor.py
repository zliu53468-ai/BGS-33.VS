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

# 模型權重（固定權重模式會使用；動態模式會依牌路型態自動調整）
MARKOV_WEIGHT = float(os.getenv("MARKOV_WEIGHT", "0.24"))
ROAD_WEIGHT = float(os.getenv("ROAD_WEIGHT", "0.21"))
STREAK_WEIGHT = float(os.getenv("STREAK_WEIGHT", "0.16"))
BALANCE_WEIGHT = float(os.getenv("BALANCE_WEIGHT", "0.08"))
RECENT_WEIGHT = float(os.getenv("RECENT_WEIGHT", "0.15"))
NGRAM_WEIGHT = float(os.getenv("NGRAM_WEIGHT", "0.11"))
ROAD_ENGINE_WEIGHT = float(os.getenv("ROAD_ENGINE_WEIGHT", "0.10"))
TIE_WEIGHT = float(os.getenv("TIE_WEIGHT", "0.04"))
AI_BLEND = float(os.getenv("AI_BLEND", "0.12"))

# 動態權重開關：只調整融合比例，不加入觀望/下注決策
USE_DYNAMIC_REGIME_WEIGHTS = os.getenv("USE_DYNAMIC_REGIME_WEIGHTS", "1") == "1"
USE_ONLINE_WEIGHTING = os.getenv("USE_ONLINE_WEIGHTING", "1") == "1"
USE_ROAD_ENGINE = os.getenv("USE_ROAD_ENGINE", "1") == "1"
ONLINE_WEIGHT_WINDOW = int(os.getenv("ONLINE_WEIGHT_WINDOW", "18"))
ONLINE_WEIGHT_MIN_COUNT = int(os.getenv("ONLINE_WEIGHT_MIN_COUNT", "5"))
ONLINE_WEIGHT_ALPHA = float(os.getenv("ONLINE_WEIGHT_ALPHA", "0.45"))
ONLINE_DISABLE_BELOW = float(os.getenv("ONLINE_DISABLE_BELOW", "0.48"))
ONLINE_BOOST_ABOVE = float(os.getenv("ONLINE_BOOST_ABOVE", "0.55"))

# RoadEngine 路紙引擎參數
ROAD_ENGINE_ROWS = int(os.getenv("ROAD_ENGINE_ROWS", "6"))
ROAD_ENGINE_MIN_HISTORY = int(os.getenv("ROAD_ENGINE_MIN_HISTORY", "10"))
ROAD_ENGINE_BREAK_STREAK = int(os.getenv("ROAD_ENGINE_BREAK_STREAK", "5"))
ROAD_ENGINE_DERIVED_LOOKBACK = int(os.getenv("ROAD_ENGINE_DERIVED_LOOKBACK", "10"))
ROAD_ENGINE_BLUE_BREAK_BIAS = float(os.getenv("ROAD_ENGINE_BLUE_BREAK_BIAS", "0.018"))
ROAD_ENGINE_RED_CONT_BIAS = float(os.getenv("ROAD_ENGINE_RED_CONT_BIAS", "0.014"))

# ML模型權重（在規律模型之後進行二次校準）
ML_WEIGHT = float(os.getenv("ML_WEIGHT", "0.14"))
ML_LR_WEIGHT = float(os.getenv("ML_LR_WEIGHT", "0.35"))
ML_RF_WEIGHT = float(os.getenv("ML_RF_WEIGHT", "0.45"))
ML_LSTM_WEIGHT = float(os.getenv("ML_LSTM_WEIGHT", "0.20"))

TIE_SHRINK = float(os.getenv("TIE_SHRINK", "0.30"))
TIE_MAX_PROB = float(os.getenv("TIE_MAX_PROB", "0.16"))
ALLOW_TIE_RECOMMEND = os.getenv("ALLOW_TIE_RECOMMEND", "0") == "1"
TIE_RECOMMEND_MIN = float(os.getenv("TIE_RECOMMEND_MIN", "0.165"))
MIN_HISTORY_FOR_AI = int(os.getenv("MIN_HISTORY_FOR_AI", "6"))
MIN_HISTORY_FOR_SIGNAL = int(os.getenv("MIN_HISTORY_FOR_SIGNAL", "4"))

# LSTM參數：預設改保守，避免單靴資料少時過擬合
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", "10"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "5"))
LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "8"))
ML_RETRAIN_INTERVAL = int(os.getenv("ML_RETRAIN_INTERVAL", "8"))

# ============ 全局模型實例（單例模式） ============
class MLModels:
    """機器學習模型容器：每個 user_id / 場館 / 房間 / 靴號 可建立獨立實例"""

    def __init__(self):
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

        self.is_trained = False
        self.training_samples = 0
        self.last_training_history = []
        self.last_training_key = ""

        # Render 啟動穩定版：不在 import 時建立 LSTM，避免服務啟動卡住。
        # LSTM 會在資料足夠並進入 train() 時才建立與訓練。

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
        """提取ML特徵（無資料洩漏版本）。維持原本 12 維，避免舊模型流程被大改。"""
        if len(non_tie) < 6:
            return np.zeros((1, 12))

        n = len(non_tie)
        b_count = non_tie.count('B')
        p_count = n - b_count
        b_rate = b_count / n if n > 0 else 0.5

        recent = non_tie[-10:] if n >= 10 else non_tie
        recent_b_rate = recent.count('B') / len(recent) if len(recent) > 0 else 0.5

        if n >= 2:
            switches = sum(1 for i in range(1, n) if non_tie[i] != non_tie[i - 1])
            switch_rate = switches / (n - 1)
        else:
            switch_rate = 0.5

        current_streak = 1
        if n >= 2:
            for i in range(n - 2, -1, -1):
                if non_tie[i] == non_tie[-1]:
                    current_streak += 1
                else:
                    break

        max_streak = 1
        current = 1
        for i in range(1, n):
            if non_tie[i] == non_tie[i - 1]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1

        last_5 = non_tie[-5:] if n >= 5 else non_tie
        last_5_b = last_5.count('B') / len(last_5) if len(last_5) > 0 else 0.5

        last_3 = non_tie[-3:] if n >= 3 else non_tie
        last_3_b = last_3.count('B') / len(last_3) if len(last_3) > 0 else 0.5

        features = np.array([[
            b_rate,
            recent_b_rate,
            switch_rate,
            current_streak / max(10, n),
            max_streak / max(10, n),
            last_5_b,
            last_3_b,
            b_count / max(10, n),
            p_count / max(10, n),
            1 if non_tie[-1] == 'B' else 0,
            (b_count - p_count) / max(10, n),
            n / 100
        ]])

        return features

    def _prepare_lstm_data(self, non_tie: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """準備LSTM序列資料"""
        if len(non_tie) < LSTM_SEQUENCE_LENGTH + 1:
            return np.array([]), np.array([])

        encoded = self._encode_sequence(non_tie)
        X, y = [], []

        for i in range(LSTM_SEQUENCE_LENGTH, len(encoded)):
            X.append(encoded[i - LSTM_SEQUENCE_LENGTH:i, 0])
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

            X_scaled = self.scaler.fit_transform(X_features)
            self.lr.fit(X_scaled, y_labels)
            self.rf.fit(X_scaled, y_labels)

            lstm_status = "disabled"
            if TF_AVAILABLE:
                X_lstm, y_lstm = self._prepare_lstm_data(non_tie)
                if len(X_lstm) > 10 and len(set(y_lstm.tolist())) >= 2:
                    self._build_lstm()
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
            features = self._extract_features(non_tie)
            features_scaled = self.scaler.transform(features)

            predictions = {}

            try:
                lr_prob = self.lr.predict_proba(features_scaled)[0][1]
                predictions['lr'] = float(lr_prob)
            except Exception:
                predictions['lr'] = 0.5

            try:
                rf_prob = self.rf.predict_proba(features_scaled)[0][1]
                predictions['rf'] = float(rf_prob)
            except Exception:
                predictions['rf'] = 0.5

            try:
                if self.lstm is not None and len(non_tie) >= LSTM_SEQUENCE_LENGTH:
                    encoded = self._encode_sequence(non_tie[-LSTM_SEQUENCE_LENGTH:])
                    X_lstm = np.array(encoded).reshape(1, LSTM_SEQUENCE_LENGTH, 1)
                    lstm_prob = float(self.lstm.predict(X_lstm, verbose=0)[0][0])
                    predictions['lstm'] = lstm_prob
                else:
                    predictions['lstm'] = 0.5
            except Exception:
                predictions['lstm'] = 0.5

            total_model_w = max(0.0001, ML_LR_WEIGHT + ML_RF_WEIGHT + ML_LSTM_WEIGHT)
            weights = {
                'lr': ML_LR_WEIGHT / total_model_w,
                'rf': ML_RF_WEIGHT / total_model_w,
                'lstm': ML_LSTM_WEIGHT / total_model_w,
            }
            ensemble = sum(predictions[k] * weights[k] for k in weights)
            predictions['ensemble'] = float(ensemble)

            return predictions

        except Exception as e:
            logger.error(f"ML預測錯誤: {e}")
            return default_result

# ============ 模型快取池 ============
MAX_MODEL_CACHE = int(os.getenv("MAX_MODEL_CACHE", "30"))
_MODEL_CACHE: Dict[str, MLModels] = {}
_MODEL_CACHE_ORDER: List[str] = []


def _get_ml_models(training_key: str) -> MLModels:
    key = training_key or "global"

    if key in _MODEL_CACHE:
        try:
            _MODEL_CACHE_ORDER.remove(key)
        except ValueError:
            pass
        _MODEL_CACHE_ORDER.append(key)
        return _MODEL_CACHE[key]

    while len(_MODEL_CACHE) >= MAX_MODEL_CACHE and _MODEL_CACHE_ORDER:
        old_key = _MODEL_CACHE_ORDER.pop(0)
        _MODEL_CACHE.pop(old_key, None)

    model = MLModels()
    _MODEL_CACHE[key] = model
    _MODEL_CACHE_ORDER.append(key)
    return model

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


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        n = max(1, len(clean))
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


def _pick_from_score(score: Dict[str, Any], min_edge: float = 0.001) -> str:
    b = float(score.get("B", 0.5))
    p = float(score.get("P", 0.5))
    if abs(b - p) < min_edge:
        return ""
    return "B" if b > p else "P"

# ============ 規律 / 牌路模型 ============
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


def _ngram_score(non_tie: List[str], max_k: int = 6) -> Dict[str, Any]:
    """通用 N-Gram 回測：尋找最近 k 碼在本靴過去出現後，下一手較常接 B 或 P。"""
    if len(non_tie) < 10:
        return {"B": 0.5, "P": 0.5, "label": "NGram資料不足", "sample": 0, "strength": 0.0, "key": ""}

    seq = "".join(non_tie)
    upper_k = min(max_k, len(non_tie) - 1)

    for k in range(upper_k, 1, -1):
        key = seq[-k:]
        follows = []

        for i in range(0, len(seq) - k):
            if seq[i:i + k] == key and i + k < len(seq):
                follows.append(seq[i + k])

        if len(follows) >= 2:
            c = Counter(follows)
            total = c["B"] + c["P"]
            alpha = float(os.getenv("NGRAM_ALPHA", "1.6"))
            b_raw = (c["B"] + alpha) / (total + 2 * alpha)
            shrink = min(0.80, total / float(os.getenv("NGRAM_FULL_SAMPLE", "8")))
            b = 0.5 * (1 - shrink) + b_raw * shrink
            p = 1 - b
            return {
                "B": b,
                "P": p,
                "label": f"NGram{k}碼:{key}",
                "sample": total,
                "strength": min(0.22, 0.08 + total * 0.02),
                "key": key,
            }

    return {"B": 0.5, "P": 0.5, "label": "NGram無重複", "sample": 0, "strength": 0.0, "key": ""}


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

# ============ RoadEngine：大路 / 衍生路紙特徵 ============
def _build_big_road(non_tie: List[str], rows: int = ROAD_ENGINE_ROWS) -> Dict[str, Any]:
    """
    建立簡化且穩定的大路矩陣。
    - 同邊：往下排；到底或被占用則往右延伸。
    - 換邊：新欄第一列。
    回傳位置、欄高、最後位置等，供 RoadEngine 特徵使用。
    """
    rows = max(3, int(rows or 6))
    grid: Dict[Tuple[int, int], str] = {}
    positions: List[Dict[str, Any]] = []

    last_side = ""
    row = 0
    col = 0

    for idx, side in enumerate(non_tie):
        if side not in {"B", "P"}:
            continue

        if idx == 0:
            row, col = 0, 0
        elif side != last_side:
            col = col + 1
            row = 0
            while (row, col) in grid:
                col += 1
        else:
            target_row = row + 1
            target_col = col
            if target_row < rows and (target_row, target_col) not in grid:
                row = target_row
            else:
                # 到底或下方被占用，往右黏邊延伸
                target_row = row
                target_col = col + 1
                while (target_row, target_col) in grid:
                    target_col += 1
                col = target_col
                row = target_row

        grid[(row, col)] = side
        positions.append({"i": idx, "side": side, "row": row, "col": col})
        last_side = side

    col_heights = Counter()
    col_sides: Dict[int, str] = {}
    for (r, c), side in grid.items():
        col_heights[c] += 1
        if r == 0:
            col_sides[c] = side

    max_col = max([p["col"] for p in positions], default=0)
    last_pos = positions[-1] if positions else {"i": -1, "side": "", "row": 0, "col": 0}

    return {
        "rows": rows,
        "grid": grid,
        "positions": positions,
        "col_heights": dict(col_heights),
        "col_sides": col_sides,
        "max_col": max_col,
        "last": last_pos,
    }


def _derived_color_at(layout: Dict[str, Any], pos: Dict[str, Any], offset: int) -> int:
    """
    衍生路紙紅藍簡化規則。
    回傳：1=紅，-1=藍，0=資料不足。
    目的不是下注決策，而是把路紙整齊/斷點特徵量化。
    """
    col = int(pos.get("col", 0))
    row = int(pos.get("row", 0))
    heights = layout.get("col_heights", {})

    if col <= offset:
        return 0

    if row == 0:
        left_h = int(heights.get(col - 1, 0))
        compare_h = int(heights.get(col - 1 - offset, 0))
        if left_h == 0 or compare_h == 0:
            return 0
        return 1 if left_h == compare_h else -1

    # 同一欄向下時，看左側相對欄位是否同樣有該列；越整齊越偏紅
    has_left_same_row = ((row, col - offset) in layout.get("grid", {}))
    has_left_prev_row = ((row - 1, col - offset) in layout.get("grid", {}))
    if has_left_same_row == has_left_prev_row:
        return 1
    return -1


def _derived_series(layout: Dict[str, Any], offset: int) -> List[int]:
    series = []
    for pos in layout.get("positions", []):
        color = _derived_color_at(layout, pos, offset)
        if color != 0:
            series.append(color)
    return series


def _color_stats(series: List[int], lookback: int = ROAD_ENGINE_DERIVED_LOOKBACK) -> Dict[str, Any]:
    tail = series[-lookback:] if series else []
    if not tail:
        return {"last": 0, "red_rate": 0.5, "blue_rate": 0.5, "count": 0, "tail": ""}
    red = tail.count(1)
    blue = tail.count(-1)
    total = red + blue
    return {
        "last": tail[-1],
        "red_rate": round(red / total, 4) if total else 0.5,
        "blue_rate": round(blue / total, 4) if total else 0.5,
        "count": total,
        "tail": "".join("R" if x == 1 else "B" for x in tail),
    }


def _road_engine_score(non_tie: List[str]) -> Dict[str, Any]:
    """
    RoadEngine：把大路、大眼仔、小路、蟑螂路轉成數值特徵與一個輕量方向分數。
    注意：這不是必勝路紙，只是讓模型多一層「傳統路紙特徵工程」。
    """
    default = {
        "B": 0.5,
        "P": 0.5,
        "label": "RoadEngine資料不足",
        "strength": 0.0,
        "big_road": {},
        "derived": {},
        "break_risk": 0.0,
        "consistency": 0.5,
    }

    if not USE_ROAD_ENGINE or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return default

    layout = _build_big_road(non_tie)
    last_side, streak_n = _streak(non_tie)
    opp = "P" if last_side == "B" else "B"

    recent = non_tie[-16:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)

    last = layout.get("last", {})
    last_col = int(last.get("col", 0))
    last_row = int(last.get("row", 0))
    col_heights = layout.get("col_heights", {})
    current_col_height = int(col_heights.get(last_col, 0))

    big_eye = _derived_series(layout, offset=1)
    small_road = _derived_series(layout, offset=2)
    cockroach = _derived_series(layout, offset=3)

    big_eye_stats = _color_stats(big_eye)
    small_road_stats = _color_stats(small_road)
    cockroach_stats = _color_stats(cockroach)

    derived_stats = {
        "big_eye": big_eye_stats,
        "small_road": small_road_stats,
        "cockroach": cockroach_stats,
    }

    red_rates = []
    blue_rates = []
    counts = []
    for stats in derived_stats.values():
        if stats.get("count", 0) > 0:
            red_rates.append(float(stats.get("red_rate", 0.5)))
            blue_rates.append(float(stats.get("blue_rate", 0.5)))
            counts.append(int(stats.get("count", 0)))

    red_pressure = sum(red_rates) / len(red_rates) if red_rates else 0.5
    blue_pressure = sum(blue_rates) / len(blue_rates) if blue_rates else 0.5
    derived_count = sum(counts)

    # 大路斷點風險：長龍、到底/黏邊、衍生路偏藍時提高。
    break_risk = 0.0
    if streak_n >= ROAD_ENGINE_BREAK_STREAK:
        break_risk += 0.24
    if last_row >= ROAD_ENGINE_ROWS - 1:
        break_risk += 0.16
    if blue_pressure >= 0.58:
        break_risk += min(0.22, (blue_pressure - 0.5) * 0.60)
    if switch_rate >= 0.70:
        break_risk += 0.10
    break_risk = _clamp(break_risk, 0.0, 0.80)

    # 方向分數：保持輕量，不讓 RoadEngine 壓過原始模型。
    b = p = 0.5
    label = "RoadEngine混合"
    strength = 0.08

    if switch_rate >= 0.72:
        side = opp
        edge = 0.045 + min(0.015, (switch_rate - 0.72) * 0.10)
        label = "RoadEngine跳路"
        strength = 0.13
    elif streak_n >= 4:
        cont_edge = 0.042 + min(0.025, (streak_n - 4) * 0.006)
        if red_pressure >= 0.58:
            cont_edge += ROAD_ENGINE_RED_CONT_BIAS
            label = "RoadEngine紅路續龍"
        elif blue_pressure >= 0.58:
            cont_edge -= ROAD_ENGINE_BLUE_BREAK_BIAS
            label = "RoadEngine藍路斷龍壓力"
        else:
            label = "RoadEngine大路長龍"

        cont_edge = _clamp(cont_edge, 0.018, 0.075)
        side = last_side if break_risk < 0.58 else opp
        edge = cont_edge if side == last_side else min(0.045, cont_edge * 0.70)
        strength = 0.15 + min(0.05, streak_n * 0.006)
    elif derived_count >= 8 and red_pressure >= 0.64:
        side = last_side
        edge = 0.035
        label = "RoadEngine衍生紅路整齊"
        strength = 0.12
    elif derived_count >= 8 and blue_pressure >= 0.64:
        side = opp
        edge = 0.035
        label = "RoadEngine衍生藍路變化"
        strength = 0.12
    elif current_col_height >= 3:
        side = last_side
        edge = 0.032
        label = "RoadEngine欄高延續"
        strength = 0.10
    else:
        side = last_side if red_pressure >= blue_pressure else opp
        edge = min(0.025, abs(red_pressure - blue_pressure) * 0.06)

    b = 0.5 + edge if side == "B" else 0.5 - edge
    p = 1 - b

    consistency = _clamp(max(red_pressure, blue_pressure), 0.5, 1.0)

    return {
        "B": b,
        "P": p,
        "label": label,
        "strength": round(strength, 4),
        "big_road": {
            "last_side": last_side,
            "last_col": last_col,
            "last_row": last_row,
            "current_col_height": current_col_height,
            "max_col": layout.get("max_col", 0),
            "is_dragon": streak_n >= 4,
            "streak": streak_n,
            "switch_rate_16": round(switch_rate, 4),
        },
        "derived": derived_stats,
        "break_risk": round(break_risk, 4),
        "red_pressure": round(red_pressure, 4),
        "blue_pressure": round(blue_pressure, 4),
        "consistency": round(consistency, 4),
    }


def _periodicity_score(non_tie: List[str], window: int = 16) -> Dict[str, Any]:
    recent = non_tie[-window:]
    best_period_score = 0.0
    best_period = 0

    for k in range(2, 6):
        if len(recent) > k:
            score = sum(
                1 for i in range(k, len(recent))
                if recent[i] == recent[i - k]
            ) / max(1, len(recent) - k)
            if score > best_period_score:
                best_period_score = score
                best_period = k

    return {"period": best_period, "score": best_period_score}


def _detect_regime(non_tie: List[str]) -> Dict[str, Any]:
    """偵測目前牌路型態，只用於調整權重，不做觀望/下注決策。"""
    fixed_weights = _normalize_weights({
        "markov": MARKOV_WEIGHT,
        "road": ROAD_WEIGHT,
        "streak": STREAK_WEIGHT,
        "balance": BALANCE_WEIGHT,
        "recent": RECENT_WEIGHT,
        "ngram": NGRAM_WEIGHT,
        "road_engine": ROAD_ENGINE_WEIGHT if USE_ROAD_ENGINE else 0.0,
    })

    if not USE_DYNAMIC_REGIME_WEIGHTS:
        return {
            "regime": "fixed",
            "weights": fixed_weights,
            "switch_rate": 0.0,
            "period_score": 0.0,
            "period": 0,
            "streak": 0,
        }

    if len(non_tie) < 8:
        weights = {
            "markov": 0.24,
            "road": 0.20,
            "streak": 0.17,
            "balance": 0.09,
            "recent": 0.16,
            "ngram": 0.07,
            "road_engine": 0.07,
        }
        if NGRAM_WEIGHT <= 0:
            weights["ngram"] = 0.0
        if ROAD_ENGINE_WEIGHT <= 0 or not USE_ROAD_ENGINE:
            weights["road_engine"] = 0.0
        return {
            "regime": "cold",
            "weights": _normalize_weights(weights),
            "switch_rate": 0.0,
            "period_score": 0.0,
            "period": 0,
            "streak": _streak(non_tie)[1],
        }

    recent = non_tie[-16:]
    last, streak_n = _streak(non_tie)
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)
    b_rate = recent.count("B") / len(recent)
    period_info = _periodicity_score(non_tie, window=16)
    best_period_score = period_info["score"]
    best_period = period_info["period"]

    if streak_n >= 4:
        regime = "trend_dragon"
        weights = {
            "markov": 0.23,
            "road": 0.20,
            "streak": 0.27,
            "balance": 0.05,
            "recent": 0.09,
            "ngram": 0.06,
            "road_engine": 0.10,
        }
    elif switch_rate >= 0.72:
        regime = "single_jump"
        weights = {
            "markov": 0.25,
            "road": 0.17,
            "streak": 0.07,
            "balance": 0.05,
            "recent": 0.26,
            "ngram": 0.09,
            "road_engine": 0.11,
        }
    elif best_period_score >= 0.70:
        regime = f"periodic_{best_period}"
        weights = {
            "markov": 0.18,
            "road": 0.23,
            "streak": 0.09,
            "balance": 0.05,
            "recent": 0.16,
            "ngram": 0.19,
            "road_engine": 0.10,
        }
    elif abs(b_rate - 0.5) >= 0.22:
        regime = "biased_side"
        weights = {
            "markov": 0.27,
            "road": 0.20,
            "streak": 0.16,
            "balance": 0.07,
            "recent": 0.14,
            "ngram": 0.06,
            "road_engine": 0.10,
        }
    elif 0.42 <= switch_rate <= 0.62 and streak_n <= 2 and best_period_score < 0.62:
        regime = "chaos_mixed"
        weights = {
            "markov": 0.23,
            "road": 0.16,
            "streak": 0.10,
            "balance": 0.07,
            "recent": 0.16,
            "ngram": 0.15,
            "road_engine": 0.13,
        }
    else:
        regime = "mixed"
        weights = {
            "markov": 0.23,
            "road": 0.20,
            "streak": 0.14,
            "balance": 0.07,
            "recent": 0.16,
            "ngram": 0.10,
            "road_engine": 0.10,
        }

    # 讓環境變數 NGRAM_WEIGHT / ROAD_ENGINE_WEIGHT 仍可控制影響力。
    if NGRAM_WEIGHT <= 0:
        weights["ngram"] = 0.0
    else:
        weights["ngram"] *= _clamp(NGRAM_WEIGHT / 0.11, 0.25, 2.00)

    if ROAD_ENGINE_WEIGHT <= 0 or not USE_ROAD_ENGINE:
        weights["road_engine"] = 0.0
    else:
        weights["road_engine"] *= _clamp(ROAD_ENGINE_WEIGHT / 0.10, 0.25, 2.00)

    return {
        "regime": regime,
        "weights": _normalize_weights(weights),
        "switch_rate": round(switch_rate, 4),
        "period_score": round(best_period_score, 4),
        "period": best_period,
        "streak": streak_n,
        "recent_b_rate": round(b_rate, 4),
    }


def _rolling_model_performance(non_tie: List[str]) -> Dict[str, Any]:
    """
    用最近 N 局做本靴內部回測，估計各子模型近期準度。
    不需要額外儲存狀態，也不改變 predict 的輸入介面。
    """
    model_names = ["markov", "road", "streak", "balance", "recent", "ngram", "road_engine"]
    result = {
        name: {"acc": 0.5, "count": 0, "correct": 0, "factor": 1.0}
        for name in model_names
    }

    if not USE_ONLINE_WEIGHTING or len(non_tie) < 12:
        return result

    start = max(6, len(non_tie) - ONLINE_WEIGHT_WINDOW)

    for i in range(start, len(non_tie)):
        prefix = non_tie[:i]
        truth = non_tie[i]
        if truth not in {"B", "P"}:
            continue

        scores = {
            "markov": _transition_prob(prefix),
            "road": _road_pattern_score(prefix),
            "streak": _streak_score(prefix),
            "balance": _balance_score(prefix),
            "recent": _recent_score(prefix),
            "ngram": _ngram_score(prefix),
            "road_engine": _road_engine_score(prefix),
        }

        for name, score in scores.items():
            pick = _pick_from_score(score, min_edge=0.002)
            if not pick:
                continue
            result[name]["count"] += 1
            if pick == truth:
                result[name]["correct"] += 1

    for name in model_names:
        cnt = result[name]["count"]
        cor = result[name]["correct"]
        if cnt > 0:
            acc = cor / cnt
            result[name]["acc"] = round(acc, 4)
        else:
            acc = 0.5
            result[name]["acc"] = 0.5

        factor = 1.0
        if cnt >= ONLINE_WEIGHT_MIN_COUNT:
            factor = 1.0 + (acc - 0.5) * 2 * ONLINE_WEIGHT_ALPHA
            if acc <= ONLINE_DISABLE_BELOW:
                factor = min(factor, 0.70)
            elif acc >= ONLINE_BOOST_ABOVE:
                factor = max(factor, 1.08)
            factor = _clamp(factor, 0.55, 1.35)

        result[name]["factor"] = round(factor, 4)

    return result


def _apply_online_weighting(base_weights: Dict[str, float], performance: Dict[str, Any]) -> Dict[str, float]:
    if not USE_ONLINE_WEIGHTING:
        return _normalize_weights(base_weights)

    adjusted = {}
    for name, weight in base_weights.items():
        factor = float(performance.get(name, {}).get("factor", 1.0))
        adjusted[name] = weight * factor
    return _normalize_weights(adjusted)


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
def predict(history: List[str], venue: str = "", room: str = "", shoe_id: str = "", user_id: str = "") -> Dict[str, Any]:
    """
    整合預測函數：規律模型 + NGram + RoadEngine + 牌路型態動態權重 + ML模型 + DeepSeek校準
    注意：本版不加入觀望/EV/下注決策，仍固定輸出 B/P/T 推薦。
    """
    history = [str(x).upper() for x in history if str(x).upper() in {"B", "P", "T"}]
    non_tie = _last_non_tie(history)

    # ============ 1. 規律模型 + 新增 NGram / RoadEngine / Regime ============
    markov = _transition_prob(non_tie)
    road = _road_pattern_score(non_tie)
    recent = _recent_score(non_tie)
    balance = _balance_score(non_tie)
    streak = _streak_score(non_tie)
    ngram = _ngram_score(non_tie)
    road_engine = _road_engine_score(non_tie)

    regime_info = _detect_regime(non_tie)
    online_performance = _rolling_model_performance(non_tie)
    dynamic_weights = _apply_online_weighting(regime_info.get("weights", {}), online_performance)

    total_w = sum(dynamic_weights.values()) or 1.0
    b_side = (
        markov["B"] * dynamic_weights.get("markov", 0.0)
        + road["B"] * dynamic_weights.get("road", 0.0)
        + streak["B"] * dynamic_weights.get("streak", 0.0)
        + balance["B"] * dynamic_weights.get("balance", 0.0)
        + recent["B"] * dynamic_weights.get("recent", 0.0)
        + ngram["B"] * dynamic_weights.get("ngram", 0.0)
        + road_engine["B"] * dynamic_weights.get("road_engine", 0.0)
    ) / total_w
    p_side = 1 - b_side

    tie_prob = _tie_score(history)
    b_prob = b_side * (1 - tie_prob)
    p_prob = p_side * (1 - tie_prob)

    # ============ 2. ML模型預測 ============
    identity = str(user_id or "anonymous")
    training_key = f"{identity}|{venue}|{room}|{shoe_id}" if (venue or room or shoe_id) else f"{identity}|global"
    ml_models = _get_ml_models(training_key)

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

    ml_pred = ml_models.predict(non_tie)
    ml_b_prob = ml_pred.get('ensemble', 0.5)

    if ml_models.is_trained:
        ml_weight = ML_WEIGHT * (0.5 + 0.5 * min(1.0, ml_models.training_samples / 50))
        b_prob = b_prob * (1 - ml_weight) + ml_b_prob * ml_weight
        p_prob = p_prob * (1 - ml_weight) + (1 - ml_b_prob) * ml_weight

    # ============ 3. DeepSeek校準 ============
    feature_payload = {
        "user_id": user_id,
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
        "ngram": ngram,
        "road_engine": road_engine,
        "regime": regime_info,
        "dynamic_weights": {k: round(v, 4) for k, v in dynamic_weights.items()},
        "online_performance": online_performance,
        "ml_predictions": ml_pred,
        "tf_available": TF_AVAILABLE,
        "training_key": training_key,
        "local_probs": {"B": round(b_prob, 5), "P": round(p_prob, 5), "T": round(tie_prob, 5)},
    }

    ai_result = None
    if len(history) >= MIN_HISTORY_FOR_AI and AI_BLEND > 0:
        try:
            ai_result = DeepSeekClient().calibrate(feature_payload)
        except Exception as e:
            ai_result = {"error": True, "message": str(e)}

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
    votes = []
    for score in [markov, road, streak, balance, recent, ngram, road_engine]:
        pick = _pick_from_score(score)
        if pick:
            votes.append(pick)

    if ml_models.is_trained:
        votes.append("B" if ml_b_prob >= 0.5 else "P")

    if not votes:
        votes = ["B" if b_prob >= p_prob else "P"]

    main_pick = "B" if b_prob >= p_prob else "P"
    agreement = votes.count(main_pick) / len(votes)

    # 修正版 ML 一致性：ML 方向與主模型一致，且 ML 自己有偏離 0.5，才提高信心。
    if ml_models.is_trained:
        ml_pick = "B" if ml_b_prob >= 0.5 else "P"
        ml_strength = abs(ml_b_prob - 0.5) * 2
        ml_agreement = ml_strength if ml_pick == main_pick else 0.0
    else:
        ml_agreement = 0.0

    # ============ 6. 推薦與信心（本版不加入觀望決策） ============
    if ALLOW_TIE_RECOMMEND and tie_prob >= TIE_RECOMMEND_MIN and tie_prob > max(b_prob, p_prob) * 0.55:
        recommend = "T"
    else:
        recommend = main_pick

    conf, level = _confidence(b_prob, p_prob, tie_prob, len(history), agreement, ml_agreement)

    # ============ 7. 原因說明 ============
    reason_parts = [
        road.get("label", "牌路"),
        f"型態:{regime_info.get('regime', '')}",
        f"{ngram.get('label', '')}",
        f"{road_engine.get('label', '')}",
        f"一致{int(agreement * 100)}%",
    ]
    if ml_models.is_trained:
        reason_parts.append(f"ML集體{int(ml_b_prob * 100)}%")
    if ai_result and ai_result.get("pattern_label"):
        reason_parts.append(f"AI:{ai_result.get('pattern_label')}")
    elif ai_result and ai_result.get("error"):
        reason_parts.append("AI離線改本地判斷")

    # ============ 8. 返回結果 ============
    return {
        "ok": True,
        "user_id": user_id,
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
        "regime": regime_info.get("regime", ""),
        "ngram_label": ngram.get("label", ""),
        "ngram_sample": ngram.get("sample", 0),
        "road_engine_label": road_engine.get("label", ""),
        "road_engine_break_risk": road_engine.get("break_risk", 0.0),
        "road_engine_consistency": road_engine.get("consistency", 0.5),
        "road_engine_big_road": road_engine.get("big_road", {}),
        "road_engine_derived": road_engine.get("derived", {}),
        "dynamic_weights": {k: round(v, 4) for k, v in dynamic_weights.items()},
        "online_model_performance": online_performance,
        "reason": " / ".join([x for x in reason_parts if x]),
        "ai_used": bool(ai_result and not ai_result.get("error")),
        "ml_trained": ml_models.is_trained,
        "ml_samples": ml_models.training_samples,
        "tf_available": TF_AVAILABLE,
        "training_key": training_key,
        "model_cache_size": len(_MODEL_CACHE),
        "ml_predictions": {
            "lr": round(ml_pred.get('lr', 0.5), 4),
            "rf": round(ml_pred.get('rf', 0.5), 4),
            "lstm": round(ml_pred.get('lstm', 0.5), 4),
            "ensemble": round(ml_pred.get('ensemble', 0.5), 4)
        } if ml_models.is_trained else None,
        "ai_result": ai_result if os.getenv("DEBUG_AI_RESULT", "0") == "1" else None,
        "debug": feature_payload if os.getenv("DEBUG_PREDICTOR", "0") == "1" else None,
    }
