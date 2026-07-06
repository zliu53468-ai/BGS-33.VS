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


def env_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class PlatformConfig:
    key: str
    name: str
    url: str


LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()

HEADLESS = env_bool("HEADLESS", True)
POLL_INTERVAL_SECONDS = env_int("POLL_INTERVAL_SECONDS", 5)
AUTO_PUSH_NEW_ROUND = env_bool("AUTO_PUSH_NEW_ROUND", True)

USE_DOM_READER = env_bool("USE_DOM_READER", True)
USE_COLOR_READER = env_bool("USE_COLOR_READER", False)

ROAD_X = env_int("ROAD_X", 0)
ROAD_Y = env_int("ROAD_Y", 0)
CELL_W = env_int("CELL_W", 28)
CELL_H = env_int("CELL_H", 28)
ROAD_COLS = env_int("ROAD_COLS", 30)
ROAD_ROWS = env_int("ROAD_ROWS", 6)

DEFAULT_TABLE_IDS = env_list(
    "DEFAULT_TABLE_IDS",
    "RB01,RB02,RB03,RB04,RB05,RB06,RB07,RB08,RB09,RB10",
)

HALL_LABELS: Dict[str, List[str]] = {
    "BACCARAT": env_list(
        "BACCARAT_HALL_LABELS",
        "經典百家樂,经典百家乐,百家樂,百家乐,Baccarat",
    ),
    "DRAGON_TIGER": env_list(
        "DRAGON_TIGER_HALL_LABELS",
        "龍虎門,龙虎门,龍虎,龙虎,Dragon Tiger",
    ),
}

PLATFORMS: Dict[str, PlatformConfig] = {
    "GSA": PlatformConfig(
        key="GSA",
        name="歐博真人",
        url=os.getenv("BACCARAT_URL_GSA", "").strip(),
    ),
    "DG": PlatformConfig(
        key="DG",
        name="DG真人",
        url=os.getenv("BACCARAT_URL_DG", "").strip(),
    ),
    "REBIRTH": PlatformConfig(
        key="REBIRTH",
        name="Rebirth真人",
        url=os.getenv("BACCARAT_URL_REBIRTH", "").strip(),
    ),
}


def enabled_platforms() -> List[PlatformConfig]:
    return [p for p in PLATFORMS.values() if p.url]


def get_platform(platform_key: str) -> PlatformConfig:
    key = (platform_key or "").strip().upper()
    platform = PLATFORMS.get(key)

    if not platform:
        raise ValueError(f"找不到平台：{platform_key}")

    if not platform.url:
        raise ValueError(f"{platform.name} 尚未設定網址，請到 Render Environment Variables 設定。")

    return platform
