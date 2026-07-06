# line_client.py
# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, List
import requests

from config import LINE_CHANNEL_ACCESS_TOKEN


def reply_line(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN 未設定，reply messages:")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"replyToken": reply_token, "messages": messages[:5]}
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code >= 400:
            print("LINE Reply Error:", res.status_code, res.text)
    except Exception as e:
        print("LINE Reply Exception:", e)


def push_line(user_id: str, messages: List[Dict[str, Any]]) -> None:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN 未設定，push messages:")
        print(user_id, json.dumps(messages, ensure_ascii=False, indent=2))
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"to": user_id, "messages": messages[:5]}
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code >= 400:
            print("LINE Push Error:", res.status_code, res.text)
    except Exception as e:
        print("LINE Push Exception:", e)
