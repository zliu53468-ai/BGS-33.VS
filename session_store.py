# session_store.py
# -*- coding: utf-8 -*-

import json
import os
import threading
import time
from typing import Any, Dict

SESSIONS_FILE = os.getenv("SESSIONS_FILE", "sessions.json")
_LOCK = threading.RLock()
_USER_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _default_session() -> Dict[str, Any]:
    return {
        "step": "IDLE",
        "platform": None,
        "hall": None,
        "table_id": None,
        "game_no": None,
        "dealer": None,
        "online_count": None,
        "countdown": 0,
        "status": "待機",
        "road": [],
        "last_round_key": None,
        "last_prediction": None,
        "last_data": None,
        "running": False,
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def _load() -> None:
    global _USER_SESSIONS
    if not os.path.exists(SESSIONS_FILE):
        return
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _USER_SESSIONS = data
    except Exception:
        _USER_SESSIONS = {}


def _save() -> None:
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(_USER_SESSIONS, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_load()


def get_session(user_id: str) -> Dict[str, Any]:
    with _LOCK:
        if user_id not in _USER_SESSIONS:
            _USER_SESSIONS[user_id] = _default_session()
            _save()
        _USER_SESSIONS[user_id]["updated_at"] = time.time()
        return dict(_USER_SESSIONS[user_id])


def update_session(user_id: str, **kwargs) -> Dict[str, Any]:
    with _LOCK:
        if user_id not in _USER_SESSIONS:
            _USER_SESSIONS[user_id] = _default_session()
        _USER_SESSIONS[user_id].update(kwargs)
        _USER_SESSIONS[user_id]["updated_at"] = time.time()
        _save()
        return dict(_USER_SESSIONS[user_id])


def reset_session(user_id: str) -> Dict[str, Any]:
    with _LOCK:
        _USER_SESSIONS[user_id] = _default_session()
        _save()
        return dict(_USER_SESSIONS[user_id])


def all_sessions() -> Dict[str, Dict[str, Any]]:
    with _LOCK:
        return dict(_USER_SESSIONS)
