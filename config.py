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
    return value in {"1", "true", "yes", "y", "on"}


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
USE_COLOR_READER = env_bool("USE_COLOR_READER", True)
ALLOW_DEFAULT_TABLE_IDS = env_bool("ALLOW_DEFAULT_TABLE_IDS", False)
READER_WAIT_MS = env_int("READER_WAIT_MS", 3500)

# Login/session health detection. If the token URL expires or the game account is kicked out,
# the reader will return a clear login_expired state instead of pretending table data was read.
LOGIN_CHECK_ENABLED = env_bool("LOGIN_CHECK_ENABLED", True)
LOGIN_EXPIRED_KEYWORDS = [
    x.strip().lower()
    for x in os.getenv(
        "LOGIN_EXPIRED_KEYWORDS",
        "token expired,session expired,expired token,invalid token,token invalid,login expired,session timeout,please login,please sign in,重新登入,請重新登入,请重新登录,請登入,请登录,登入失效,登录失效,登入逾時,登录超时,連線逾時,连接超时,無效連結,无效链接,試玩已結束,试玩已结束,帳號已登出,账号已登出"
    ).split(",")
    if x.strip()
]
LOGIN_MIN_TEXT_LENGTH = env_int("LOGIN_MIN_TEXT_LENGTH", 80)

# LINE chat loading animation. Works in one-on-one chats only.
LINE_LOADING_ENABLED = env_bool("LINE_LOADING_ENABLED", True)
LINE_LOADING_SECONDS = env_int("LINE_LOADING_SECONDS", 20)

# Reader strictness: avoid showing fake text-only table ids like DG66/E9 unless they contain real metadata.
STRICT_TABLE_METADATA = env_bool("STRICT_TABLE_METADATA", False)

# v5: manual-first table selection. Recommended for live casino lobby UIs where table text/card data is delayed or canvas-rendered.
MANUAL_FIRST_TABLE_MODE = env_bool("MANUAL_FIRST_TABLE_MODE", False)
AUTO_SCAN_TABLES_ON_HALL = env_bool("AUTO_SCAN_TABLES_ON_HALL", True)
TABLE_CLICK_TIMEOUT_MS = env_int("TABLE_CLICK_TIMEOUT_MS", 2500)
TARGET_TABLE_ONLY = env_bool("TARGET_TABLE_ONLY", True)


# Color road recognition. If ROAD_X/ROAD_Y are 0, fixed-grid mode is disabled; use ROI auto-detect instead.
ROAD_AUTO_DETECT = env_bool("ROAD_AUTO_DETECT", True)
AUTO_COLOR_FULL_SCAN = env_bool("AUTO_COLOR_FULL_SCAN", True)
ROAD_ROI_X = env_int("ROAD_ROI_X", 0)
ROAD_ROI_Y = env_int("ROAD_ROI_Y", 0)
ROAD_ROI_W = env_int("ROAD_ROI_W", 0)
ROAD_ROI_H = env_int("ROAD_ROI_H", 0)
COLOR_MIN_AREA = env_int("COLOR_MIN_AREA", 30)
COLOR_MAX_AREA = env_int("COLOR_MAX_AREA", 2500)

# Optional per-selected-table ROI. If you manually input a table/room and the page is opened there,
# set these to the bead-road/roadmap area only, not the full game lobby.
TARGET_ROAD_ROI_X = env_int("TARGET_ROAD_ROI_X", 0)
TARGET_ROAD_ROI_Y = env_int("TARGET_ROAD_ROI_Y", 0)
TARGET_ROAD_ROI_W = env_int("TARGET_ROAD_ROI_W", 0)
TARGET_ROAD_ROI_H = env_int("TARGET_ROAD_ROI_H", 0)



PLAYWRIGHT_BROWSERS_PATH = os.getenv("PLAYWRIGHT_BROWSERS_PATH", "").strip()

ROAD_X = env_int("ROAD_X", 0)
ROAD_Y = env_int("ROAD_Y", 0)
CELL_W = env_int("CELL_W", 28)
CELL_H = env_int("CELL_H", 28)
ROAD_COLS = env_int("ROAD_COLS", 30)
ROAD_ROWS = env_int("ROAD_ROWS", 6)

DEFAULT_TABLE_IDS = [
    x.strip().upper()
    for x in os.getenv("DEFAULT_TABLE_IDS", "").split(",")
    if x.strip()
]

DEEPSEEK_ENABLED = env_bool("DEEPSEEK_ENABLED", False)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_TIMEOUT_SECONDS = env_int("DEEPSEEK_TIMEOUT_SECONDS", 8)
DEEPSEEK_WEIGHT = env_float("DEEPSEEK_WEIGHT", 0.30)
LOCAL_MODEL_WEIGHT = env_float("LOCAL_MODEL_WEIGHT", 0.70)


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

PLATFORMS: Dict[str, PlatformConfig] = {
    "GSA": PlatformConfig(
        key="GSA",
        name="歐博真人",
        url=os.getenv("BACCARAT_URL_GSA", "").strip(),
        hall_labels=HALL_LABELS,
    ),
    "DG": PlatformConfig(
        key="DG",
        name="DG 真人",
        url=os.getenv("BACCARAT_URL_DG", "").strip(),
        hall_labels=HALL_LABELS,
    ),
    "REBIRTH": PlatformConfig(
        key="REBIRTH",
        name="REBIRTH 真人",
        url=os.getenv("BACCARAT_URL_REBIRTH", "").strip(),
        hall_labels=HALL_LABELS,
    ),
    "MT": PlatformConfig(
        key="MT",
        name="MT 真人",
        url=os.getenv("BACCARAT_URL_MT", "").strip(),
        hall_labels=HALL_LABELS,
    ),
    "T9": PlatformConfig(
        key="T9",
        name="T9 真人",
        url=os.getenv("BACCARAT_URL_T9", "").strip(),
        hall_labels=HALL_LABELS,
    ),
    "SA": PlatformConfig(
        key="SA",
        name="SA 真人",
        url=os.getenv("BACCARAT_URL_SA", "").strip(),
        hall_labels=HALL_LABELS,
    ),
}


def enabled_platforms() -> List[PlatformConfig]:
    return [p for p in PLATFORMS.values() if p.url]


def get_platform(key: str) -> PlatformConfig:
    key = (key or "").upper().strip()
    platform = PLATFORMS.get(key)
    if not platform:
        raise ValueError(f"找不到平台：{key}")
    if not platform.url:
        raise ValueError(f"{platform.name} 尚未設定網址，請到 Render 環境變數新增 BACCARAT_URL_{platform.key}")
    return platform


def public_url(path: str) -> str:
    path = path if path.startswith("/") else "/" + path
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL + path
    return path
