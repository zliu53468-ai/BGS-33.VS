# app.py
# -*- coding: utf-8 -*-

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from baccarat_reader import BaccaratReader
from config import FRONTEND_AUTO_POLL_MS, LIFF_ID, enabled_platforms
from line_client import push_line, reply_line, verify_signature
from line_messages import (
    build_analysis_message,
    build_guide_message,
    build_hall_message,
    build_loading_message,
    build_platform_message,
    build_table_message,
    text_message,
)
from monitor import MonitorManager
from predictor import predict
from session_store import get_session, reset_session, update_session

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

reader = BaccaratReader()
monitor_manager = MonitorManager()


@app.get("/")
async def root():
    return PlainTextResponse("Baccarat AI LINE Bot is running.")


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


async def push_analysis(user_id: str, data: Dict[str, Any], prediction: Dict[str, Any]) -> None:
    if data.get("error"):
        push_line(user_id, [text_message(f"系統讀取失敗：{data.get('error')}\n可稍後再試，或重新選擇平台/桌號。")])
        return
    push_line(user_id, [build_analysis_message(data, prediction)])


async def push_tables_after_loading(user_id: str, platform: str, hall: str) -> None:
    try:
        tables = await reader.list_tables(platform, hall)
        update_session(user_id, step="TABLE_SELECT", tables=tables)
        push_line(user_id, [build_table_message(tables)])
    except Exception as e:
        push_line(user_id, [text_message(f"讀取桌號失敗：{e}\n目前沒有讀取到真實桌號，請確認 token / 平台大廳是否可用。")])


async def handle_action(user_id: str, action: str) -> List[Dict[str, Any]]:
    action = (action or "").strip()

    if action in {"開始預測", "開始分析", "開始", "START"}:
        reset_session(user_id)
        return [build_guide_message()]

    if action == "FLOW:START":
        update_session(user_id, step="PLATFORM_SELECT", running=False)
        return [build_platform_message()]

    if action.startswith("PLATFORM:"):
        platform = action.split(":", 1)[1].strip().upper()
        update_session(user_id, platform=platform, step="HALL_SELECT")
        try:
            return [build_hall_message(platform)]
        except Exception as e:
            return [text_message(f"平台設定錯誤：{e}")]

    if action.startswith("HALL:"):
        hall = action.split(":", 1)[1].strip().upper()
        session = get_session(user_id)
        platform = session.get("platform")
        if not platform:
            return [text_message("請先選擇平台。")]
        update_session(user_id, hall=hall, step="TABLE_LOADING")
        asyncio.create_task(push_tables_after_loading(user_id, platform, hall))
        return [build_loading_message("正在讀取真實桌號與荷官資料，請稍候...")]

    if action.startswith("TABLE:"):
        table_id = action.split(":", 1)[1].strip().upper()
        session = get_session(user_id)
        platform = session.get("platform")
        hall = session.get("hall")
        if not platform or not hall:
            return [text_message("請先選擇平台與遊戲廳。")]
        update_session(user_id, table_id=table_id, step="ANALYZING", running=True)
        await monitor_manager.start(user_id, platform, hall, table_id, on_update=push_analysis)
        return [build_loading_message(f"已選擇桌號 {table_id}，正在建立監控並讀取第一筆分析...")]

    if action in {"ANALYZE:CONTINUE", "繼續分析"}:
        session = get_session(user_id)
        data = session.get("last_data")
        prediction = session.get("last_prediction")
        if data and prediction:
            return [build_analysis_message(data, prediction)]
        refreshed = await monitor_manager.refresh_once(user_id)
        if refreshed:
            return [build_analysis_message(refreshed["data"], refreshed["prediction"])]
        return [text_message("資料仍在讀取中，請稍候 3～5 秒後再按繼續分析。")]

    if action in {"ANALYZE:STOP", "結束分析", "停止分析"}:
        await monitor_manager.stop(user_id)
        reset_session(user_id)
        return [text_message("已結束分析，已回到主選單。"), build_guide_message()]

    if action.startswith("桌號") or action.startswith("桌号"):
        table_id = action.replace("桌號", "").replace("桌号", "").strip().upper()
        if not table_id:
            return [text_message("請輸入：桌號 你的桌號")]
        session = get_session(user_id)
        platform = session.get("platform")
        hall = session.get("hall")
        if not platform or not hall:
            return [text_message("請先選擇平台與遊戲廳。")]
        update_session(user_id, table_id=table_id, step="ANALYZING", running=True)
        await monitor_manager.start(user_id, platform, hall, table_id, on_update=push_analysis)
        return [build_loading_message(f"已手動指定桌號 {table_id}，正在讀取分析...")]

    return [text_message("請輸入「開始預測」，或使用按鈕操作。")]


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    if not verify_signature(body, signature):
        raise HTTPException(status_code=400, detail="Invalid LINE signature")

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid json"})

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


# ---------- LIFF / Web API ----------

@app.get("/api/config")
async def api_config():
    return {
        "ok": True,
        "liffId": LIFF_ID,
        "autoPollMs": FRONTEND_AUTO_POLL_MS,
        "platforms": [{"key": p.key, "name": p.name} for p in enabled_platforms()],
        "halls": [
            {"key": "BACCARAT", "name": "經典百家樂"},
            {"key": "DRAGON_TIGER", "name": "龍虎門"},
        ],
    }


@app.get("/api/session/current")
async def api_current(user_id: str):
    return {"ok": True, "session": get_session(user_id)}


@app.post("/api/flow/start")
async def api_flow_start(payload: Dict[str, Any]):
    user_id = payload.get("user_id") or "local"
    reset_session(user_id)
    return {"ok": True, "session": get_session(user_id), "platforms": [{"key": p.key, "name": p.name} for p in enabled_platforms()]}


@app.post("/api/flow/platform")
async def api_flow_platform(payload: Dict[str, Any]):
    user_id = payload.get("user_id") or "local"
    platform = str(payload.get("platform") or "").upper()
    update_session(user_id, platform=platform, step="HALL_SELECT")
    return {"ok": True, "session": get_session(user_id)}


@app.post("/api/flow/hall")
async def api_flow_hall(payload: Dict[str, Any]):
    user_id = payload.get("user_id") or "local"
    hall = str(payload.get("hall") or "BACCARAT").upper()
    session = get_session(user_id)
    platform = session.get("platform")
    if not platform:
        raise HTTPException(status_code=400, detail="請先選擇平台")
    update_session(user_id, hall=hall, step="TABLE_LOADING")
    tables = await reader.list_tables(platform, hall)
    update_session(user_id, step="TABLE_SELECT", tables=tables)
    return {"ok": True, "session": get_session(user_id), "tables": tables}


@app.post("/api/flow/table")
async def api_flow_table(payload: Dict[str, Any]):
    user_id = payload.get("user_id") or "local"
    table_id = str(payload.get("table_id") or "").upper()
    session = get_session(user_id)
    platform = session.get("platform")
    hall = session.get("hall")
    if not platform or not hall or not table_id:
        raise HTTPException(status_code=400, detail="請先選擇平台、遊戲廳與桌號")
    update_session(user_id, table_id=table_id, step="ANALYZING", running=True)
    await monitor_manager.start(user_id, platform, hall, table_id, on_update=push_analysis)
    return {"ok": True, "session": get_session(user_id), "message": "正在建立監控"}


@app.post("/api/analyze/continue")
async def api_analyze_continue(payload: Dict[str, Any]):
    user_id = payload.get("user_id") or "local"
    session = get_session(user_id)
    if session.get("last_data") and session.get("last_prediction"):
        return {"ok": True, "session": session, "data": session.get("last_data"), "prediction": session.get("last_prediction")}
    refreshed = await monitor_manager.refresh_once(user_id)
    if refreshed:
        return {"ok": True, "session": get_session(user_id), **refreshed}
    return {"ok": False, "session": session, "message": "資料讀取中"}


@app.post("/api/analyze/stop")
async def api_analyze_stop(payload: Dict[str, Any]):
    user_id = payload.get("user_id") or "local"
    await monitor_manager.stop(user_id)
    session = reset_session(user_id)
    return {"ok": True, "session": session}


# ---------- Debug ----------

@app.get("/api/debug/playwright")
async def debug_playwright():
    from playwright.async_api import async_playwright
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = await browser.new_page()
            await page.goto("https://example.com", wait_until="domcontentloaded", timeout=30000)
            title = await page.title()
            await browser.close()
            return {"ok": True, "stage": "playwright_browser_ok", "title": title}
    except Exception as e:
        return {"ok": False, "stage": "playwright_browser_failed", "error": str(e)}


@app.get("/api/debug/baccarat")
async def debug_baccarat(platform: str = "DG", hall: str = "BACCARAT"):
    playwright = browser = None
    try:
        playwright, browser, page, collector = await reader.prepare_page(platform, hall)
        text = await reader.extract_visible_text(page)
        return {
            "ok": True,
            "stage": "baccarat_page_connected",
            "platform": platform,
            "hall": hall,
            "title": await page.title(),
            "current_url": page.url,
            "text_length": len(text),
            "text_preview": text[:1200],
            "network_chunks": len(collector.text_chunks),
            "json_chunks": len(collector.json_chunks),
        }
    except Exception as e:
        return {"ok": False, "stage": "baccarat_page_failed", "error": str(e)}
    finally:
        await reader.close_browser(playwright, browser)


@app.get("/api/debug/tables")
async def debug_tables(platform: str = "DG", hall: str = "BACCARAT"):
    try:
        tables = await reader.list_tables(platform, hall)
        return {"ok": True, "stage": "table_list_result", "table_count": len(tables), "tables": tables}
    except Exception as e:
        return {"ok": False, "stage": "table_list_failed", "error": str(e)}


@app.get("/api/debug/table")
async def debug_table(platform: str = "DG", hall: str = "BACCARAT", table_id: str = ""):
    try:
        if not table_id:
            raise HTTPException(status_code=400, detail="請帶 table_id")
        data = await reader.read_table_data(platform, hall, table_id)
        prediction = predict(data.get("road", []))
        return {
            "ok": True,
            "stage": "table_data_result",
            "data": data,
            "road_length": len(data.get("road", [])),
            "road_text": "".join(data.get("road", [])),
            "prediction": prediction,
        }
    except Exception as e:
        return {"ok": False, "stage": "table_data_failed", "error": str(e)}
