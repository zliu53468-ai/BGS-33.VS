# app.py
# -*- coding: utf-8 -*-

import base64
import hashlib
import hmac
import json
import os
from typing import Any, Dict, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import LINE_CHANNEL_SECRET, public_config
from baccarat_reader import BaccaratReader
from predictor import predict
from session_store import get_session, reset_session, update_session, session_to_public
from monitor import MonitorManager
from line_client import reply_line, push_line
from line_messages import (
    text_message,
    build_guide_message,
    build_platform_message,
    build_hall_message,
    build_table_message,
    build_analysis_message,
)

app = FastAPI(title="富百家 AI Pro")
app.mount("/static", StaticFiles(directory="static"), name="static")

reader = BaccaratReader()
monitor = MonitorManager()


class UserOnly(BaseModel):
    user_id: str


class PlatformPayload(BaseModel):
    user_id: str
    platform: str


class HallPayload(BaseModel):
    user_id: str
    hall: str


class TablePayload(BaseModel):
    user_id: str
    table_id: str


@app.on_event("startup")
async def startup_event():
    async def _push(user_id: str, data: Dict[str, Any], prediction: Dict[str, Any]):
        push_line(user_id, [build_analysis_message(data, prediction)])
    monitor.set_push_callback(_push)


@app.get("/")
async def root():
    index = "static/index.html"
    if os.path.exists(index):
        return FileResponse(index)
    return PlainTextResponse("富百家 AI Pro Bot is running.")


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


@app.get("/api/config")
async def api_config():
    return JSONResponse(public_config())


def verify_line_signature(body: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        return True
    digest = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


async def analyze_current_session(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id)
    platform = session.get("platform")
    hall = session.get("hall")
    table_id = session.get("table_id")
    if not platform or not hall or not table_id:
        return {"error": "請先選擇平台、遊戲廳與桌號。"}
    result = await monitor.read_now(user_id)
    return result


async def handle_action(user_id: str, action: str) -> List[Dict[str, Any]]:
    action = (action or "").strip()

    if action in ("開始預測", "開始分析", "開始", "START"):
        reset_session(user_id)
        return [build_guide_message()]

    if action == "FLOW:START":
        update_session(user_id, step="PLATFORM_SELECT", running=False)
        return [build_platform_message()]

    if action.startswith("PLATFORM:"):
        platform = action.split(":", 1)[1].strip().upper()
        update_session(user_id, platform=platform, step="HALL_SELECT")
        return [build_hall_message(platform)]

    if action.startswith("HALL:"):
        hall = action.split(":", 1)[1].strip().upper()
        session = get_session(user_id)
        platform = session.get("platform")
        if not platform:
            return [text_message("請先選擇平台。")]
        update_session(user_id, hall=hall, step="TABLE_SELECT")
        try:
            tables = await reader.list_tables(platform_key=platform, hall_key=hall)
            return [build_table_message(tables)]
        except Exception as e:
            return [text_message(f"讀取桌號失敗：{e}\n目前不會再顯示假 RB01。請先測 /api/debug/baccarat 確認是否有接到真資料。")]

    if action.startswith("TABLE:"):
        table_id = action.split(":", 1)[1].strip().upper()
        session = update_session(user_id, table_id=table_id, step="ANALYZING", running=True)
        platform = session.get("platform")
        hall = session.get("hall")
        try:
            result = await monitor.start(user_id, platform, hall, table_id)
            return [build_analysis_message(result["data"], result["prediction"])]
        except Exception as e:
            return [text_message(f"系統讀取失敗：{e}")]

    if action in ("ANALYZE:CONTINUE", "繼續分析"):
        result = await analyze_current_session(user_id)
        if result.get("error"):
            return [text_message(result["error"])]
        return [build_analysis_message(result["data"], result["prediction"])]

    if action in ("ANALYZE:STOP", "結束分析"):
        await monitor.stop(user_id)
        return [text_message("已結束分析。需要重新開始時，請輸入：開始預測")]

    return [text_message("請輸入「開始預測」，或使用按鈕操作。")]


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    if not verify_line_signature(body, signature):
        raise HTTPException(status_code=400, detail="Invalid LINE signature")
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"})

    for event in payload.get("events", []):
        reply_token = event.get("replyToken")
        source = event.get("source", {})
        user_id = source.get("userId") or source.get("groupId") or source.get("roomId")
        if not reply_token or not user_id:
            continue
        action = ""
        if event.get("type") == "message":
            msg = event.get("message", {})
            if msg.get("type") == "text":
                action = msg.get("text", "").strip()
        elif event.get("type") == "postback":
            action = event.get("postback", {}).get("data", "").strip()
        if not action:
            reply_line(reply_token, [text_message("請輸入「開始預測」。")])
            continue
        try:
            messages = await handle_action(user_id, action)
            reply_line(reply_token, messages)
        except Exception as e:
            reply_line(reply_token, [text_message(f"系統讀取失敗：{e}")])
    return JSONResponse({"ok": True})


@app.get("/api/session/current")
async def api_session_current(user_id: str):
    return {"ok": True, "session": session_to_public(get_session(user_id))}


@app.post("/api/flow/start")
async def api_flow_start(payload: UserOnly):
    session = reset_session(payload.user_id)
    session["step"] = "PLATFORM_SELECT"
    return {"ok": True, "session": session_to_public(session), "config": public_config()}


@app.post("/api/flow/platform")
async def api_flow_platform(payload: PlatformPayload):
    session = update_session(payload.user_id, platform=payload.platform.upper(), step="HALL_SELECT")
    return {"ok": True, "session": session_to_public(session)}


@app.post("/api/flow/hall")
async def api_flow_hall(payload: HallPayload):
    session = get_session(payload.user_id)
    platform = session.get("platform")
    if not platform:
        raise HTTPException(status_code=400, detail="請先選擇平台")
    update_session(payload.user_id, hall=payload.hall.upper(), step="TABLE_SELECT")
    tables = await reader.list_tables(platform, payload.hall.upper())
    return {"ok": True, "session": session_to_public(get_session(payload.user_id)), "tables": tables, "real_table_count": len([t for t in tables if t.get('source') != 'fallback_default'])}


@app.post("/api/flow/table")
async def api_flow_table(payload: TablePayload):
    session = update_session(payload.user_id, table_id=payload.table_id.upper(), step="ANALYZING", running=True)
    platform = session.get("platform")
    hall = session.get("hall")
    if not platform or not hall:
        raise HTTPException(status_code=400, detail="請先選擇平台與遊戲廳")
    result = await monitor.start(payload.user_id, platform, hall, payload.table_id.upper())
    return {"ok": True, "session": session_to_public(get_session(payload.user_id)), **result}


@app.post("/api/analyze/continue")
async def api_analyze_continue(payload: UserOnly):
    result = await analyze_current_session(payload.user_id)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return {"ok": True, "session": session_to_public(get_session(payload.user_id)), **result}


@app.post("/api/analyze/stop")
async def api_analyze_stop(payload: UserOnly):
    await monitor.stop(payload.user_id)
    return {"ok": True, "session": session_to_public(get_session(payload.user_id))}


@app.get("/api/debug/playwright")
async def debug_playwright():
    from playwright.async_api import async_playwright
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = await browser.new_page()
            await page.goto("https://example.com", wait_until="domcontentloaded", timeout=30000)
            title = await page.title()
            current_url = page.url
            await browser.close()
            return {"ok": True, "stage": "playwright_browser_ok", "title": title, "current_url": current_url}
    except Exception as e:
        return {"ok": False, "stage": "playwright_browser_failed", "error": str(e)}


@app.get("/api/debug/baccarat")
async def debug_baccarat(platform: str = "DG", hall: str = "BACCARAT"):
    try:
        data = await reader.debug_page(platform.upper(), hall.upper())
        return data
    except Exception as e:
        return {"ok": False, "stage": "baccarat_page_failed", "platform": platform, "hall": hall, "error": str(e)}


@app.get("/api/debug/tables")
async def debug_tables(platform: str = "DG", hall: str = "BACCARAT"):
    try:
        tables = await reader.list_tables(platform.upper(), hall.upper())
        return {"ok": True, "platform": platform, "hall": hall, "table_count": len(tables), "tables": tables, "message": "若 table_count=0，代表目前沒有抓到真實桌台資料，且已關閉假 RB01 fallback。"}
    except Exception as e:
        return {"ok": False, "platform": platform, "hall": hall, "error": str(e)}


@app.get("/api/debug/table")
async def debug_table(platform: str = "DG", hall: str = "BACCARAT", table_id: str = "RB05"):
    try:
        data = await reader.read_table_data(platform.upper(), hall.upper(), table_id.upper())
        prediction = predict(data.get("road", []))
        return {"ok": True, "platform": platform, "hall": hall, "table_id": table_id, "data": data, "road_length": len(data.get("road", [])), "road_text": "".join(data.get("road", [])), "prediction": prediction}
    except Exception as e:
        return {"ok": False, "platform": platform, "hall": hall, "table_id": table_id, "error": str(e)}
