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

# postback 輸入模式：
# silent = 莊/閒/和只記錄，不回聊天室訊息，最省訊息量
# panel  = 每次輸入都回覆新版面板，測試時可用，但會洗版
ROUND_INPUT_REPLY_MODE = os.getenv("ROUND_INPUT_REPLY_MODE", "silent").strip().lower()
ACK_EVERY_N_ROUNDS = int(os.getenv("ACK_EVERY_N_ROUNDS", "0") or "0")

app = FastAPI(title="Baccarat LINE Postback AI Bot", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if STATIC_DIR.exists():
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
    venues: List[Dict[str, str]] = []
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


def parse_rooms() -> List[str]:
    return [x.strip() for x in DEFAULT_ROOMS.split(",") if x.strip()]


def venue_name(venue_code: str) -> str:
    for v in parse_venues():
        if v["code"] == venue_code:
            return v["name"]
    return venue_code or "-"


def build_liff_url(venue_code: str = "") -> str:
    """保留 LIFF API 相容用。主流程已改成 postback，不會再主動跳網頁。"""
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


def postback_action(label: str, data: Dict[str, str]) -> Dict[str, Any]:
    # 不放 displayText，避免使用者每點一次按鈕，聊天室就多一則文字。
    return {
        "type": "postback",
        "label": label[:20],
        "data": urllib.parse.urlencode(data),
    }


def button(label: str, data: Dict[str, str], color: str = "#FFD000", style: str = "primary") -> Dict[str, Any]:
    return {
        "type": "button",
        "style": style,
        "color": color,
        "height": "sm",
        "action": postback_action(label, data),
    }


def get_source_user_id(event: Dict[str, Any]) -> str:
    source = event.get("source") or {}
    return source.get("userId") or source.get("groupId") or source.get("roomId") or "anonymous"


def get_session_or_create(user_id: str) -> Dict[str, Any]:
    session = store.get_session(user_id)
    if not session:
        session = store.upsert_session(user_id, {})
    return session


def history_text(history: List[str], limit: int = 28) -> str:
    if not history:
        return "尚未輸入"
    display = history[-limit:]
    text = " ".join(display)
    if len(history) > limit:
        text = "… " + text
    return text


def result_name(code: str) -> str:
    return {"B": "莊", "P": "閒", "T": "和"}.get(code, code)


def percent_text(value: Any) -> str:
    try:
        v = float(value)
        if v <= 1:
            v *= 100
        return f"{v:.0f}%"
    except Exception:
        return f"{value}%" if value not in [None, ""] else "--"


def kv(label: str, value: Any) -> Dict[str, Any]:
    return {
        "type": "box",
        "layout": "horizontal",
        "margin": "md",
        "contents": [
            {"type": "text", "text": label, "size": "sm", "color": "#333333", "flex": 2},
            {"type": "text", "text": str(value), "size": "sm", "color": "#333333", "align": "end", "flex": 4, "wrap": True},
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


def venue_flex() -> Dict[str, Any]:
    buttons = []
    for v in parse_venues():
        buttons.append(button(v["name"], {"action": "select_venue", "venue": v["code"]}))
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
                    {"type": "text", "text": "請選擇遊戲館，接著會在聊天室內操作，不會跳網頁。", "size": "sm", "color": "#FFFFFF", "margin": "md", "wrap": True},
                    {"type": "separator", "margin": "lg", "color": "#FFD000"},
                    {"type": "box", "layout": "vertical", "spacing": "md", "margin": "lg", "contents": buttons},
                    {"type": "text", "text": "按鈕採用 Postback，不會把點擊文字洗在聊天室。", "size": "xs", "color": "#AAAAAA", "margin": "lg", "wrap": True},
                ],
            },
        },
    }


def room_flex(venue_code: str) -> Dict[str, Any]:
    buttons = [
        button(room, {"action": "select_room", "venue": venue_code, "room": room})
        for room in parse_rooms()
    ]
    return {
        "type": "flex",
        "altText": "請選擇遊戲廳",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {"type": "text", "text": venue_name(venue_code), "weight": "bold", "size": "xl", "color": "#FFD000"},
                    {"type": "text", "text": "請選擇遊戲廳，選完後即可在聊天室內輸入莊 / 閒 / 和。", "size": "sm", "color": "#FFFFFF", "margin": "md", "wrap": True},
                    {"type": "separator", "margin": "lg", "color": "#FFD000"},
                    {"type": "box", "layout": "vertical", "spacing": "md", "margin": "lg", "contents": buttons},
                    {"type": "separator", "margin": "lg", "color": "#333333"},
                    {
                        "type": "button",
                        "style": "secondary",
                        "height": "sm",
                        "margin": "md",
                        "action": postback_action("重新選館", {"action": "open_venue"}),
                    },
                ],
            },
        },
    }


def input_panel_flex(session: Dict[str, Any]) -> Dict[str, Any]:
    history = session.get("history", []) or []
    venue = session.get("venue", "")
    room = session.get("room", "")
    shoe_id = session.get("shoe_id", "") or "可直接輸入靴號"
    round_no = len(history) + 1

    return {
        "type": "flex",
        "altText": "莊閒和輸入面板",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#FFFFFF",
                "paddingAll": "16px",
                "contents": [
                    {"type": "text", "text": "AI 規律分析", "weight": "bold", "size": "xl", "color": "#111111"},
                    {"type": "separator", "margin": "md", "color": "#FFD000"},
                    kv("遊戲館", venue_name(venue)),
                    kv("遊戲廳", room or "-"),
                    kv("靴號", shoe_id),
                    kv("目前局數", f"第 {round_no} 局"),
                    {"type": "text", "text": "目前紀錄", "size": "sm", "color": "#111111", "weight": "bold", "margin": "lg"},
                    {
                        "type": "box",
                        "layout": "vertical",
                        "backgroundColor": "#F7F7F7",
                        "cornerRadius": "md",
                        "paddingAll": "10px",
                        "margin": "sm",
                        "contents": [
                            {"type": "text", "text": history_text(history), "size": "sm", "wrap": True, "color": "#333333"}
                        ],
                    },
                    {"type": "text", "text": "輸入莊 / 閒 / 和", "size": "sm", "color": "#111111", "weight": "bold", "margin": "lg"},
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "spacing": "sm",
                        "margin": "sm",
                        "contents": [
                            button("莊 B", {"action": "add_round", "result": "B"}, "#FFE3E3"),
                            button("閒 P", {"action": "add_round", "result": "P"}, "#E3EAFF"),
                            button("和 T", {"action": "add_round", "result": "T"}, "#E3FFE7"),
                        ],
                    },
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "spacing": "sm",
                        "margin": "sm",
                        "contents": [
                            button("上一步", {"action": "undo_round"}, "#222222"),
                            button("查看紀錄", {"action": "view_panel"}, "#222222"),
                        ],
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "spacing": "sm",
                        "margin": "md",
                        "contents": [
                            button("開始AI判斷", {"action": "predict"}, "#FFD000"),
                            button("清除本靴", {"action": "reset_session"}, "#555555"),
                            button("結束分析", {"action": "end_session"}, "#111111"),
                        ],
                    },
                    {"type": "text", "text": "提示：點莊/閒/和只會背景記錄，不會每次跳訊息。按「查看紀錄」才更新面板。", "size": "xs", "color": "#888888", "margin": "lg", "wrap": True},
                ],
            },
        },
    }


def result_flex(session: Dict[str, Any]) -> Dict[str, Any]:
    pred = session.get("last_prediction") or {}
    recommend = pred.get("recommend_text") or pred.get("recommend") or "-"
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
                    kv("遊戲館", venue_name(session.get("venue", ""))),
                    kv("遊戲廳", session.get("room", "-")),
                    kv("靴號", session.get("shoe_id", "-")),
                    kv("局數", f"第 {len(session.get('history', []) or []) + 1} 局"),
                    kv("遊戲狀態", session.get("status", "可押注")),
                    rate_line("莊", percent_text(pred.get("banker_rate", 0)), "#E60012"),
                    rate_line("閒", percent_text(pred.get("player_rate", 0)), "#0000CC"),
                    rate_line("和", percent_text(pred.get("tie_rate", 0)), "#00A000"),
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "margin": "lg",
                        "cornerRadius": "md",
                        "backgroundColor": "#FFD000",
                        "paddingAll": "10px",
                        "contents": [
                            {"type": "text", "text": "推薦", "weight": "bold", "color": "#111111", "flex": 1},
                            {"type": "text", "text": str(recommend), "weight": "bold", "align": "end", "color": "#111111", "flex": 2},
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
                    button("繼續輸入", {"action": "view_panel"}, "#FFD000"),
                    button("重新選館", {"action": "open_venue"}, "#222222"),
                ],
            },
        },
    }


def predict_and_save(user_id: str) -> Dict[str, Any]:
    session = store.get_session(user_id)
    if not session:
        raise ValueError("請先輸入「開始分析」並選擇遊戲館。")
    pred = predict(
        history=session.get("history", []),
        venue=session.get("venue", ""),
        room=session.get("room", ""),
        shoe_id=session.get("shoe_id", ""),
    )
    return store.upsert_session(user_id, {**session, "last_prediction": pred, "status": "可押注"})


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "baccarat-line-postback-ai-bot", "version": "2.0.0"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "baccarat-line-postback-ai-bot"}


@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("pong")


@app.get("/liff")
def liff_page() -> Any:
    # 保留舊 LIFF 頁面相容，但主流程已改為聊天室 postback 操作。
    html_path = STATIC_DIR / "liff.html"
    if not html_path.exists():
        return JSONResponse({"ok": False, "detail": "static/liff.html not found. 主流程可直接用 LINE Postback，不需 LIFF。"}, status_code=404)
    return FileResponse(html_path)


@app.get("/api/config")
def api_config() -> Dict[str, Any]:
    return {
        "liffId": LIFF_ID,
        "venues": parse_venues(),
        "rooms": parse_rooms(),
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
    try:
        session = predict_and_save(body.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True, "session": session, "prediction": session.get("last_prediction")}


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
        user_id = get_source_user_id(event)

        if event_type == "follow":
            line_reply(reply_token, [venue_flex()])
            continue

        if event_type == "message" and event.get("message", {}).get("type") == "text":
            text = event.get("message", {}).get("text", "").strip()
            lower_text = text.lower()

            if any(k in text for k in ["開始", "選館", "遊戲館", "重新選館"]):
                line_reply(reply_token, [venue_flex()])
                continue

            if text in ["紀錄", "查看紀錄", "面板", "輸入面板"]:
                session = get_session_or_create(user_id)
                line_reply(reply_token, [input_panel_flex(session)])
                continue

            if text in ["AI", "開始AI判斷", "判斷", "預測"]:
                try:
                    session = predict_and_save(user_id)
                    line_reply(reply_token, [result_flex(session)])
                except Exception as exc:
                    line_reply(reply_token, [text_msg(str(exc))])
                continue

            if text in ["結束", "結束分析"]:
                session = store.end_session(user_id)
                line_reply(reply_token, [text_msg(f"已結束本靴分析。總局數：{len(session.get('history', []) or [])} 局")])
                continue

            mapping = {"莊": "B", "庄": "B", "b": "B", "B": "B", "閒": "P", "闲": "P", "p": "P", "P": "P", "和": "T", "t": "T", "T": "T"}
            if text in mapping or lower_text in mapping:
                result = mapping.get(text) or mapping.get(lower_text)
                try:
                    session = store.add_round(user_id, result)
                    line_reply(reply_token, [input_panel_flex(session)])
                except Exception as exc:
                    line_reply(reply_token, [text_msg(f"輸入失敗：{exc}")])
                continue

            # 已選廳後，使用者輸入一般文字時，當作靴號 / 桌號備註。
            session = get_session_or_create(user_id)
            if session.get("venue") or session.get("room"):
                session = store.upsert_session(user_id, {**session, "shoe_id": text})
                line_reply(reply_token, [input_panel_flex(session)])
            else:
                line_reply(reply_token, [text_msg("請輸入「開始分析」開啟遊戲館選擇。")])

        elif event_type == "postback":
            raw_data = event.get("postback", {}).get("data", "")
            data = {k: v[0] for k, v in urllib.parse.parse_qs(raw_data).items()}
            action = data.get("action", "")

            try:
                if action == "open_venue":
                    line_reply(reply_token, [venue_flex()])

                elif action == "select_venue":
                    venue = data.get("venue", "")
                    session = get_session_or_create(user_id)
                    store.upsert_session(user_id, {**session, "venue": venue, "status": "選擇遊戲廳"})
                    line_reply(reply_token, [room_flex(venue)])

                elif action == "select_room":
                    venue = data.get("venue", "")
                    room = data.get("room", "")
                    session = store.new_session(user_id, venue, room, "")
                    line_reply(reply_token, [input_panel_flex(session)])

                elif action == "view_panel":
                    session = get_session_or_create(user_id)
                    line_reply(reply_token, [input_panel_flex(session)])

                elif action == "add_round":
                    result = data.get("result", "")
                    session = store.add_round(user_id, result)
                    history_len = len(session.get("history", []) or [])
                    should_reply = ROUND_INPUT_REPLY_MODE == "panel" or (
                        ACK_EVERY_N_ROUNDS > 0 and history_len % ACK_EVERY_N_ROUNDS == 0
                    )
                    if should_reply:
                        line_reply(reply_token, [input_panel_flex(session)])
                    # silent 模式不回覆，避免每點一次莊/閒/和就洗版。

                elif action == "undo_round":
                    session = store.undo_round(user_id)
                    line_reply(reply_token, [input_panel_flex(session)])

                elif action == "reset_session":
                    session = store.clear_history(user_id)
                    line_reply(reply_token, [input_panel_flex(session)])

                elif action == "end_session":
                    session = store.end_session(user_id)
                    line_reply(reply_token, [text_msg(f"已結束本靴分析。總局數：{len(session.get('history', []) or [])} 局")])

                elif action == "predict":
                    session = predict_and_save(user_id)
                    line_reply(reply_token, [result_flex(session)])

                else:
                    line_reply(reply_token, [venue_flex()])

            except Exception as exc:
                line_reply(reply_token, [text_msg(f"操作失敗：{exc}")])

    return JSONResponse({"ok": True})
