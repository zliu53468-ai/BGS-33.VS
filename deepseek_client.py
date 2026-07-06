# deepseek_client.py
# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, List, Optional
import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT_SECONDS,
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # 兼容模型用 ```json 包住
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


class DeepSeekClient:
    def __init__(self, api_key: str = ""):
        self.api_key = (api_key or DEEPSEEK_API_KEY).strip()
        self.api_url = DEEPSEEK_API_URL
        self.model = DEEPSEEK_MODEL
        self.timeout = DEEPSEEK_TIMEOUT_SECONDS

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def analyze_road(self, road: List[str], local_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        回傳格式固定：
        {
          side: "B" | "P" | "OBSERVE",
          confidence: 0-100,
          reason: "...",
          pattern: "..."
        }
        """
        if not self.enabled:
            return {
                "ok": False,
                "side": "OBSERVE",
                "confidence": 0,
                "reason": "DeepSeek API Key 未設定。",
                "pattern": "DISABLED",
            }

        road_text = "".join(road[-80:])
        local_summary = {
            "recommend": local_result.get("recommend"),
            "signal_level": local_result.get("signal_level"),
            "pattern_detail": local_result.get("pattern_detail"),
            "banker_percent": local_result.get("banker_percent"),
            "player_percent": local_result.get("player_percent"),
            "tie_percent": local_result.get("tie_percent"),
            "reason": local_result.get("reason"),
        }

        system_prompt = (
            "你是百家樂牌路分析校準器，只能基於使用者給的 B/P/T 牌路做統計與型態分析。"
            "B=莊，P=閒，T=和。請不要保證命中，不要誇大勝率。"
            "你的任務是獨立判斷下一手偏向 B、P 或 OBSERVE。"
            "請只輸出 JSON，不要加多餘文字。"
        )
        user_prompt = {
            "road": road_text,
            "local_model_result": local_summary,
            "output_schema": {
                "side": "B 或 P 或 OBSERVE",
                "confidence": "0到100的整數",
                "pattern": "看到的主要牌路型態",
                "reason": "50字內中文原因",
            },
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
            "temperature": 0.15,
            "max_tokens": 300,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            res = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            if res.status_code >= 400:
                return {
                    "ok": False,
                    "side": "OBSERVE",
                    "confidence": 0,
                    "reason": f"DeepSeek API 錯誤：HTTP {res.status_code}",
                    "pattern": "API_ERROR",
                    "raw": res.text[:300],
                }

            data = res.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = _extract_json(content)
            if not parsed:
                return {
                    "ok": False,
                    "side": "OBSERVE",
                    "confidence": 0,
                    "reason": "DeepSeek 回傳格式無法解析。",
                    "pattern": "PARSE_ERROR",
                    "raw": content[:300],
                }

            side = str(parsed.get("side", "OBSERVE")).upper().strip()
            if side in ("莊", "BANKER"):
                side = "B"
            elif side in ("閒", "闲", "PLAYER"):
                side = "P"
            elif side not in ("B", "P"):
                side = "OBSERVE"

            try:
                confidence = int(float(parsed.get("confidence", 0)))
            except Exception:
                confidence = 0
            confidence = max(0, min(100, confidence))

            return {
                "ok": True,
                "side": side,
                "confidence": confidence,
                "pattern": str(parsed.get("pattern", "AI_PATTERN"))[:80],
                "reason": str(parsed.get("reason", "AI 獨立校準完成。"))[:120],
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
