import time
from typing import Any, Dict

USER_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _empty_session() -> Dict[str, Any]:
    now = time.time()
    return {
        "step": "IDLE",
        "platform": None,
        "hall": None,
        "table_id": None,
        "game_no": None,
        "dealer": None,
        "online_count": None,
        "road": [],
        "last_round_key": None,
        "running": False,
        "created_at": now,
        "updated_at": now,
    }


def get_session(user_id: str) -> Dict[str, Any]:
    if user_id not in USER_SESSIONS:
        USER_SESSIONS[user_id] = _empty_session()

    USER_SESSIONS[user_id]["updated_at"] = time.time()
    return USER_SESSIONS[user_id]


def update_session(user_id: str, **kwargs) -> Dict[str, Any]:
    session = get_session(user_id)
    session.update(kwargs)
    session["updated_at"] = time.time()
    return session


def reset_session(user_id: str) -> Dict[str, Any]:
    USER_SESSIONS[user_id] = _empty_session()
    return USER_SESSIONS[user_id]
