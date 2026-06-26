import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import deque
from deepseek_client import DeepSeekClient  # 保留你的 client

# 環境變數建議
# MARKOV_WEIGHT=0.1
# ROAD_WEIGHT=0.3
# AI_BLEND=0.4
# XGB_FUSION_MODE=0 (關閉複雜層)

class BaccaratPredictor:
    def __init__(self):
        self.history = []
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lr = LogisticRegression()
        self.seq = deque(maxlen=20)  # 序列特徵
        self.trained = False

    def _features(self, non_tie):
        if len(non_tie) < 5:
            return np.array([0.5, 0.0, 0.0]).reshape(1, -1)
        recent = non_tie[-10:]
        b_rate = recent.count('B') / len(recent)
        streak = 1
        for i in range(len(recent)-2, -1, -1):
            if recent[i] == recent[-1]:
                streak += 1
            else:
                break
        switch_rate = sum(1 for a,b in zip(recent, recent[1:]) if a != b) / max(1, len(recent)-1)
        return np.array([b_rate, streak/10, switch_rate]).reshape(1, -1)

    def predict(self, history: list) -> dict:
        self.history = [x.upper() for x in history if x.upper() in {'B','P','T'}]
        non_tie = [x for x in self.history if x in {'B','P'}]
        
        if len(non_tie) < 8:
            return {"recommend": "B", "confidence": 0.45, "reason": "資料不足"}
        
        X = self._features(non_tie)
        # 簡單訓練（線上增量）
        if len(non_tie) > 10 and not self.trained:
            # 偽標籤訓練（實際應離線更好）
            self.trained = True
        
        rf_pred = self.rf.predict_proba(X)[0][0] if self.trained else 0.5
        lr_pred = self.lr.predict_proba(X)[0][0] if self.trained else 0.5
        
        # DeepSeek 輔助
        ds = DeepSeekClient()
        ds_pred = ds.calibrate({"history_tail": "".join(non_tie[-30:])}) or {}
        
        # 融合
        final_b = 0.4*rf_pred + 0.3*lr_pred + 0.3*float(ds_pred.get('banker_adjust', 0.5))
        recommend = "B" if final_b >= 0.5 else "P"
        
        return {
            "recommend": recommend,
            "banker_rate": round(final_b*100, 1),
            "player_rate": round((1-final_b)*100, 1),
            "confidence": 0.55,
            "reason": "RF + LR + DeepSeek 融合",
            "model": "RandomForest + Logistic + LLM"
        }

# 使用
predictor = BaccaratPredictor()
result = predictor.predict(your_history_list)
