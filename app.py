import base64
import hashlib
import hmac
import json
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import store
from predictor import predict

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
LIFF_ID = os.getenv("LIFF_ID", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

DEFAULT_VENUES = "OB:歐博真人,DG:DG真人,MT:MT真人,T9:T9真人,SA:SA真人"
VENUES_RAW = os.getenv("VENUES", DEFAULT_VENUES)
DEFAULT_ROOMS = os.getenv(
    "DEFAULT_ROOMS",
    "百家樂-中文廳,百家樂-亞洲廳,百家樂-極速廳,百家樂-保險廳,百家樂-VIP廳",
)

app = FastAPI(title="Baccarat LINE LIFF AI Bot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class StartSessionIn(BaseModel):
    user_id: str
    venue: str = ""
    room: str = ""
    shoe_id: str = ""


class UserIn(BaseModel):
    user_id: str


class AddRoundIn(BaseModel):
    user_id: str
    result: str


class PredictIn(BaseModel):
    user_id: str


def parse_venues() -> List[Dict[str, str]]:
    venues = []
    for item in VENUES_RAW.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            code, name = item.split(":", 1)
        else:
            code, name = item, item
        venues.append({"code": code.strip(), "name": name.strip()})
    return venues


def build_liff_url(venue_code: str = "") -> str:
    query = urllib.parse.urlencode({"venue": venue_code}) if venue_code else ""
    if LIFF_ID:
        url = f"https://liff.line.me/{LIFF_ID}"
        return f"{url}?{query}" if query else url
    if PUBLIC_BASE_URL:
        url = f"{PUBLIC_BASE_URL}/liff"
        return f"{url}?{query}" if query else url
    return f"/liff?{query}" if query else "/liff"


def verify_line_signature(body: bytes, signature: Optional[str]) -> bool:
    if not CHANNEL_SECRET:
        return True
    if not signature:
        return False
    digest = hmac.new(CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    if not CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN is empty; reply skipped.")
        return
    r = requests.post(
        "https://api.line.me/v2/bot/message/reply",
        headers={
            "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"replyToken": reply_token, "messages": messages[:5]},
        timeout=8,
    )
    if r.status_code >= 300:
        print("LINE reply failed", r.status_code, r.text)


def text_msg(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def venue_flex() -> Dict[str, Any]:
    buttons = []
    for v in parse_venues():
        buttons.append(
            {
                "type": "button",
                "style": "primary",
                "color": "#FFD000",
                "height": "sm",
                "action": {
                    "type": "uri",
                    "label": v["name"],
                    "uri": build_liff_url(v["code"]),
                },
            }
        )
    return {
        "type": "flex",
        "altText": "請選擇遊戲館",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {"type": "text", "text": "AI 規律模型", "weight": "bold", "size": "xl", "color": "#FFD000"},
                    {"type": "text", "text": "請選擇遊戲館，接著會開啟輸入面板。", "size": "sm", "color": "#FFFFFF", "margin": "md", "wrap": True},
                    {"type": "separator", "margin": "lg", "color": "#FFD000"},
                    {"type": "box", "layout": "vertical", "spacing": "md", "margin": "lg", "contents": buttons},
                    {"type": "text", "text": "點擊後進入 LIFF 面板，莊/閒/和輸入不會洗版。", "size": "xs", "color": "#AAAAAA", "margin": "lg", "wrap": True},
                ],
            },
        },
    }


def result_flex(session: Dict[str, Any]) -> Dict[str, Any]:
    pred = session.get("last_prediction") or {}
    recommend = pred.get("recommend_text", "-")
    return {
        "type": "flex",
        "altText": f"分析結果：推薦 {recommend}",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#FFFFFF",
                "contents": [
                    {"type": "text", "text": "分析數據", "weight": "bold", "size": "lg", "align": "center", "color": "#111111"},
                    {"type": "separator", "margin": "md", "color": "#FFD000"},
                    kv("遊戲館", session.get("venue", "-")),
                    kv("遊戲廳", session.get("room", "-")),
                    kv("靴號", session.get("shoe_id", "-")),
                    kv("倒數計時", "依現場桌台"),
                    kv("遊戲狀態", session.get("status", "可押注")),
                    rate_line("莊", f"{pred.get('banker_rate', 0)}%", "#E60012"),
                    rate_line("閒", f"{pred.get('player_rate', 0)}%", "#0000CC"),
                    rate_line("和", f"{pred.get('tie_rate', 0)}%", "#00A000"),
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "margin": "lg",
                        "cornerRadius": "md",
                        "backgroundColor": "#FFD000",
                        "paddingAll": "10px",
                        "contents": [
                            {"type": "text", "text": "推薦", "weight": "bold", "color": "#111111", "flex": 1},
                            {"type": "text", "text": recommend, "weight": "bold", "align": "end", "color": "#111111", "flex": 2},
                        ],
                    },
                    {"type": "text", "text": f"{pred.get('signal_level', '')}｜{pred.get('reason', '')}", "size": "xs", "color": "#777777", "margin": "md", "wrap": True},
                ],
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {
                        "type": "button",
                        "style": "primary",
                        "color": "#FFD000",
                        "action": {"type": "uri", "label": "開啟輸入面板", "uri": build_liff_url(session.get("venue", ""))},
                    }
                ],
            },
        },
    }


def kv(label: str, value: str) -> Dict[str, Any]:
    return {
        "type": "box",
        "layout": "horizontal",
        "margin": "md",
        "contents": [
            {"type": "text", "text": label, "size": "sm", "color": "#333333", "flex": 2},
            {"type": "text", "text": str(value), "size": "sm", "color": "#333333", "align": "end", "flex": 4},
        ],
    }


def rate_line(label: str, value: str, color: str) -> Dict[str, Any]:
    return {
        "type": "box",
        "layout": "horizontal",
        "margin": "md",
        "contents": [
            {"type": "text", "text": label, "size": "md", "weight": "bold", "color": color, "flex": 1},
            {"type": "text", "text": value, "size": "sm", "color": "#333333", "align": "end", "flex": 3},
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "baccarat-line-liff-ai-bot"}


@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("pong")


@app.get("/liff")
def liff_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "liff.html")


@app.get("/api/config")
def api_config() -> Dict[str, Any]:
    return {
        "liffId": LIFF_ID,
        "venues": parse_venues(),
        "rooms": [x.strip() for x in DEFAULT_ROOMS.split(",") if x.strip()],
        "publicBaseUrl": PUBLIC_BASE_URL,
    }


@app.get("/api/session/current")
def api_current(user_id: str) -> Dict[str, Any]:
    session = store.get_session(user_id)
    if not session:
        session = store.upsert_session(user_id, {})
    return {"ok": True, "session": session}


@app.post("/api/session/start")
def api_start(body: StartSessionIn) -> Dict[str, Any]:
    session = store.new_session(body.user_id, body.venue, body.room, body.shoe_id)
    return {"ok": True, "session": session}


@app.post("/api/round/add")
def api_add_round(body: AddRoundIn) -> Dict[str, Any]:
    try:
        session = store.add_round(body.user_id, body.result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "session": session}


@app.post("/api/round/undo")
def api_undo(body: UserIn) -> Dict[str, Any]:
    session = store.undo_round(body.user_id)
    return {"ok": True, "session": session}


@app.post("/api/session/reset")
def api_reset(body: UserIn) -> Dict[str, Any]:
    session = store.clear_history(body.user_id)
    return {"ok": True, "session": session}


@app.post("/api/session/end")
def api_end(body: UserIn) -> Dict[str, Any]:
    session = store.end_session(body.user_id)
    return {"ok": True, "session": session}


@app.post("/api/predict")
def api_predict(body: PredictIn) -> Dict[str, Any]:
    session = store.get_session(body.user_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    pred = predict(
        history=session.get("history", []),
        venue=session.get("venue", ""),
        room=session.get("room", ""),
        shoe_id=session.get("shoe_id", ""),
    )
    session = store.upsert_session(body.user_id, {**session, "last_prediction": pred, "status": "可押注"})
    return {"ok": True, "session": session, "prediction": pred}


@app.post("/callback")
async def callback(request: Request) -> JSONResponse:
    body = await request.body()
    signature = request.headers.get("x-line-signature")
    if not verify_line_signature(body, signature):
        raise HTTPException(status_code=403, detail="invalid signature")

    payload = json.loads(body.decode("utf-8") or "{}")
    for event in payload.get("events", []):
        reply_token = event.get("replyToken")
        if not reply_token:
            continue
        event_type = event.get("type")

        if event_type == "message" and event.get("message", {}).get("type") == "text":
            text = event.get("message", {}).get("text", "").strip()
            if any(k in text for k in ["開始", "分析", "選館", "選擇", "遊戲館"]):
                line_reply(reply_token, [venue_flex()])
            else:
                line_reply(reply_token, [text_msg("請輸入「開始分析」開啟遊戲館選擇。")])

        elif event_type == "postback":
            line_reply(reply_token, [venue_flex()])

    return JSONResponse({"ok": True})
