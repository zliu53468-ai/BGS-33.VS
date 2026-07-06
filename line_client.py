# line_client.py
# -*- coding: utf-8 -*-

import base64
import hashlib
import hmac
import json
from typing import Any, Dict, List

import requests

from config import LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET


def verify_signature(body: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        return True
    digest = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")


def _post_line(endpoint: str, payload: Dict[str, Any]) -> None:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE token 未設定，訊息內容：")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=15)
        if res.status_code >= 400:
            print("LINE API error:", res.status_code, res.text[:500])
    except Exception as e:
        print("LINE API exception:", e)


def reply_line(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    _post_line("https://api.line.me/v2/bot/message/reply", {
        "replyToken": reply_token,
        "messages": messages[:5],
    })


def push_line(to: str, messages: List[Dict[str, Any]]) -> None:
    _post_line("https://api.line.me/v2/bot/message/push", {
        "to": to,
        "messages": messages[:5],
    })
