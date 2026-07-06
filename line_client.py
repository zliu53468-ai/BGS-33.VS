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

    digest = hmac.new(
        LINE_CHANNEL_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).digest()

    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")


def _line_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


def reply_message(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN 未設定，以下是原本要 reply 的內容：")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        return

    payload = {
        "replyToken": reply_token,
        "messages": messages[:5],
    }

    try:
        response = requests.post(
            "https://api.line.me/v2/bot/message/reply",
            headers=_line_headers(),
            json=payload,
            timeout=15,
        )
        if response.status_code >= 400:
            print("LINE reply error:", response.status_code, response.text)
    except Exception as exc:
        print("LINE reply exception:", exc)


def push_message(to: str, messages: List[Dict[str, Any]]) -> None:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN 未設定，以下是原本要 push 的內容：")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        return

    payload = {
        "to": to,
        "messages": messages[:5],
    }

    try:
        response = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers=_line_headers(),
            json=payload,
            timeout=15,
        )
        if response.status_code >= 400:
            print("LINE push error:", response.status_code, response.text)
    except Exception as exc:
        print("LINE push exception:", exc)
