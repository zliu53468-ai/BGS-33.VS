# line_client.py
# -*- coding: utf-8 -*-

import base64
import hashlib
import hmac
import json
from typing import Any, Dict, List

import requests

from config import (
    LINE_CHANNEL_ACCESS_TOKEN,
    LINE_CHANNEL_SECRET,
    LINE_LOADING_ENABLED,
    LINE_LOADING_SECONDS,
)


def verify_signature(body: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        return True
    digest = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


def _post_line(endpoint: str, payload: Dict[str, Any]) -> None:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE token 未設定，訊息內容：")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    try:
        res = requests.post(endpoint, headers=_headers(), json=payload, timeout=15)
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


def show_loading(chat_id: str, seconds: int | None = None) -> None:
    """
    LINE 官方 Loading Animation。
    只支援一對一 userId；groupId / roomId 會被 LINE 拒絕，所以這裡靜默略過。
    """
    if not LINE_LOADING_ENABLED or not LINE_CHANNEL_ACCESS_TOKEN:
        return
    if not chat_id or not str(chat_id).startswith("U"):
        return

    allowed = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60}
    loading_seconds = int(seconds or LINE_LOADING_SECONDS or 20)
    if loading_seconds not in allowed:
        loading_seconds = 20

    payload = {
        "chatId": chat_id,
        "loadingSeconds": loading_seconds,
    }
    try:
        res = requests.post(
            "https://api.line.me/v2/bot/chat/loading/start",
            headers=_headers(),
            json=payload,
            timeout=5,
        )
        if res.status_code >= 400:
            print("LINE loading API error:", res.status_code, res.text[:300])
    except Exception as e:
        print("LINE loading API exception:", e)
