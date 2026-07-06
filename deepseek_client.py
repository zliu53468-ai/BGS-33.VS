# deepseek_client.py
# -*- coding: utf-8 -*-

import json
import re
from typing import Any, Dict, List

import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT_SECONDS,
)


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _normalize_side(value: Any) -> str:
    v = str(value or "").strip().upper()
    if v in {"B", "BANKER", "莊", "庄"}:
        return "B"
    if v in {"P", "PLAYER", "閒", "闲"}:
        return "P"
    return "OBSERVE"


class DeepSeekClient:
    def __init__(self) -> None:
        self.api_key = DEEPSEEK_API_KEY
        self.api_url = DEEPSEEK_API_URL
        self.model = DEEPSEEK_MODEL
        self.timeout = DEEPSEEK_TIMEOUT_SECONDS

    def analyze_road(self, road: List[str], local_result: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "ok": False,
                "side": "OBSERVE",
                "confidence": 0,
                "reason": "DeepSeek API Key 未設定。",
                "pattern": "DISABLED",
            }

        recent_road = "".join(road[-60:])
        local_summary = {
            "recommend": local_result.get("recommend"),
            "signal_level": local_result.get("signal_level"),
            "pattern_detail": local_result.get("pattern_detail"),
            "banker_percent": local_result.get("banker_percent"),
            "player_percent": local_result.get("player_percent"),
            "tie_percent": local_result.get("tie_percent"),
            "gap": local_result.get("gap"),
            "reason": local_result.get("reason"),
        }

        prompt = f"""
你是一個百家樂牌路分析校準器。請基於牌路資料獨立判斷下一局方向，但請保守處理，不要硬推。

規則：
- B=莊、P=閒、T=和。
- 和局 T 不作為主要推薦方向，只作參考。
- 若牌路混亂、差距不明顯、長龍過長、或模型分歧，請建議 OBSERVE。
- 只輸出 JSON，不要輸出多餘文字。

最近牌路：{recent_road}
本地模型摘要：{json.dumps(local_summary, ensure_ascii=False)}

請輸出格式：
{{
  "side": "B 或 P 或 OBSERVE",
  "confidence": 0-100,
  "pattern": "你觀察到的牌路型態",
  "reason": "50字內原因"
}}
""".strip()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是嚴謹、保守的百家樂牌路分析校準器，只輸出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            }
            res = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            if res.status_code >= 400:
                return {
                    "ok": False,
                    "side": "OBSERVE",
                    "confidence": 0,
                    "reason": f"DeepSeek HTTP {res.status_code}: {res.text[:120]}",
                    "pattern": "API_ERROR",
                }
            data = res.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = _extract_json(content)
            side = _normalize_side(parsed.get("side"))
            try:
                confidence = int(parsed.get("confidence", 0))
            except Exception:
                confidence = 0
            confidence = max(0, min(100, confidence))
            return {
                "ok": True,
                "side": side,
                "confidence": confidence,
                "pattern": str(parsed.get("pattern") or "AI_PATTERN"),
                "reason": str(parsed.get("reason") or "AI 已完成校準。"),
                "raw": parsed,
            }
        except Exception as e:
            return {
                "ok": False,
                "side": "OBSERVE",
                "confidence": 0,
                "reason": f"DeepSeek 呼叫失敗：{e}",
                "pattern": "EXCEPTION",
            }
