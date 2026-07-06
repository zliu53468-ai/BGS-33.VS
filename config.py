# config.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in ("1", "true", "yes", "y", "on")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


@dataclass(frozen=True)
class PlatformConfig:
    key: str
    name: str
    url: str
    hall_labels: Dict[str, List[str]]


HALL_LABELS = {
    "BACCARAT": [
        "經典百家樂", "经典百家乐", "百家樂", "百家乐", "Baccarat", "BACCARAT",
    ],
    "DRAGON_TIGER": [
        "龍虎門", "龙虎门", "龍虎", "龙虎", "Dragon Tiger", "DRAGON TIGER",
    ],
}

PLATFORMS = {
    "GSA": PlatformConfig(
        key="GSA",
        name=os.getenv("BACCARAT_NAME_GSA", "歐博真人").strip(),
        url=os.getenv("BACCARAT_URL_GSA", "").strip(),
        hall_labels=HALL_LABELS,
    ),
    "DG": PlatformConfig(
        key="DG",
        name=os.getenv("BACCARAT_NAME_DG", "DG真人").strip(),
        url=os.getenv("BACCARAT_URL_DG", "").strip(),
        hall_labels=HALL_LABELS,
    ),
    "REBIRTH": PlatformConfig(
        key="REBIRTH",
        name=os.getenv("BACCARAT_NAME_REBIRTH", "REBIRTH真人").strip(),
        url=os.getenv("BACCARAT_URL_REBIRTH", "").strip(),
        hall_labels=HALL_LABELS,
    ),
}

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
LIFF_ID = os.getenv("LIFF_ID", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

HEADLESS = env_bool("HEADLESS", True)
POLL_INTERVAL_SECONDS = env_int("POLL_INTERVAL_SECONDS", 5)
FRONTEND_AUTO_POLL_MS = env_int("FRONTEND_AUTO_POLL_MS", 6000)
AUTO_PUSH_NEW_ROUND = env_bool("AUTO_PUSH_NEW_ROUND", True)

USE_DOM_READER = env_bool("USE_DOM_READER", True)
USE_NETWORK_READER = env_bool("USE_NETWORK_READER", True)
USE_COLOR_READER = env_bool("USE_COLOR_READER", False)

# 正式版預設關閉假桌號，避免 RB01 被誤認為真資料。
ALLOW_DEFAULT_TABLE_IDS = env_bool("ALLOW_DEFAULT_TABLE_IDS", False)
DEFAULT_TABLE_IDS = [
    x.strip().upper()
    for x in os.getenv("DEFAULT_TABLE_IDS", "").split(",")
    if x.strip()
]

READER_WAIT_MS = env_int("READER_WAIT_MS", 8000)
READER_TIMEOUT_MS = env_int("READER_TIMEOUT_MS", 60000)
BROWSER_WIDTH = env_int("BROWSER_WIDTH", 1280)
BROWSER_HEIGHT = env_int("BROWSER_HEIGHT", 900)

ROAD_X = env_int("ROAD_X", 0)
ROAD_Y = env_int("ROAD_Y", 0)
CELL_W = env_int("CELL_W", 28)
CELL_H = env_int("CELL_H", 28)
ROAD_COLS = env_int("ROAD_COLS", 30)
ROAD_ROWS = env_int("ROAD_ROWS", 6)

DEEPSEEK_ENABLED = env_bool("DEEPSEEK_ENABLED", False)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_TIMEOUT_SECONDS = env_int("DEEPSEEK_TIMEOUT_SECONDS", 8)
DEEPSEEK_WEIGHT = env_float("DEEPSEEK_WEIGHT", 0.30)
LOCAL_MODEL_WEIGHT = env_float("LOCAL_MODEL_WEIGHT", 0.70)


def enabled_platforms() -> List[PlatformConfig]:
    return [p for p in PLATFORMS.values() if p.url]


def get_platform(key: str) -> PlatformConfig:
    normalized = (key or "").upper().strip()
    platform = PLATFORMS.get(normalized)
    if not platform:
        raise ValueError(f"找不到平台：{key}")
    if not platform.url:
        raise ValueError(f"{platform.name} 尚未設定登入網址，請檢查 Render 環境變數。")
    return platform


def public_config() -> dict:
    return {
        "liffId": LIFF_ID,
        "frontendAutoPollMs": FRONTEND_AUTO_POLL_MS,
        "platforms": [
            {"key": p.key, "name": p.name}
            for p in enabled_platforms()
        ],
        "halls": [
            {"key": "BACCARAT", "name": "經典百家樂"},
            {"key": "DRAGON_TIGER", "name": "龍虎門"},
        ],
        "reader": {
            "useDomReader": USE_DOM_READER,
            "useNetworkReader": USE_NETWORK_READER,
            "useColorReader": USE_COLOR_READER,
            "allowDefaultTableIds": ALLOW_DEFAULT_TABLE_IDS,
        },
        "deepseek": {
            "enabled": DEEPSEEK_ENABLED and bool(DEEPSEEK_API_KEY),
            "model": DEEPSEEK_MODEL,
            "weight": DEEPSEEK_WEIGHT,
        },
    }
