# session_store.py
# -*- coding: utf-8 -*-

import json
import os
import time
from typing import Any, Dict, List

SESSIONS_FILE = os.getenv("SESSIONS_FILE", "sessions.json")
USER_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _default_session() -> Dict[str, Any]:
    return {
        "step": "IDLE",
        "platform": None,
        "hall": None,
        "table_id": None,
        "game_no": None,
        "dealer": None,
        "online_count": 0,
        "countdown": 0,
        "status": "未開始",
        "road": [],
        "last_round_key": None,
        "last_prediction": None,
        "running": False,
        "real_data": False,
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def load_sessions() -> None:
    global USER_SESSIONS
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    USER_SESSIONS = data
    except Exception:
        USER_SESSIONS = {}


def save_sessions() -> None:
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(USER_SESSIONS, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_session(user_id: str) -> Dict[str, Any]:
    if not USER_SESSIONS:
        load_sessions()
    if user_id not in USER_SESSIONS:
        USER_SESSIONS[user_id] = _default_session()
        save_sessions()
    USER_SESSIONS[user_id]["updated_at"] = time.time()
    return USER_SESSIONS[user_id]


def reset_session(user_id: str) -> Dict[str, Any]:
    USER_SESSIONS[user_id] = _default_session()
    save_sessions()
    return USER_SESSIONS[user_id]


def update_session(user_id: str, **kwargs) -> Dict[str, Any]:
    session = get_session(user_id)
    session.update(kwargs)
    session["updated_at"] = time.time()
    save_sessions()
    return session


def session_to_public(session: Dict[str, Any]) -> Dict[str, Any]:
    return dict(session)
