# app.py
# -*- coding: utf-8 -*-

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from baccarat_reader import BaccaratReader, LoginExpiredError
from config import AUTO_SCAN_TABLES_ON_HALL, FRONTEND_AUTO_POLL_MS, LIFF_ID, MANUAL_FIRST_TABLE_MODE, enabled_platforms
from line_client import push_line, reply_line, show_loading, verify_signature
from line_messages import (
    build_analysis_message,
    build_guide_message,
    build_hall_message,
    build_loading_message,
    build_platform_message,
    build_table_message,
    build_manual_table_prompt_message,
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
    if data.get("login_expired"):
        push_line(user_id, [build_analysis_message(data, prediction)])
        return
    if data.get("error"):
        push_line(user_id, [text_message(f"系統讀取失敗：{data.get('error')}\n可稍後再試，或重新選擇平台/桌號。")])
        return
    push_line(user_id, [build_analysis_message(data, prediction)])


async def push_tables_after_loading(user_id: str, platform: str, hall: str) -> None:
    try:
        tables = await reader.list_tables(platform, hall)
        update_session(user_id, step="TABLE_SELECT", tables=tables)
        if tables:
            push_line(user_id, [build_table_message(tables)])
        else:
            push_line(user_id, [text_message("自動掃描沒有讀取到牌桌。可再按一次重新掃描，或直接輸入：房號 E5 / 桌號 R5037。")])
    except LoginExpiredError as e:
        reset_session(user_id)
        push_line(user_id, [text_message(f"平台登入已失效：{e}\n請先到 Render 更新該平台 BACCARAT_URL 的完整登入網址/token，再重新開始預測。"), build_guide_message()])
    except Exception as e:
        push_line(user_id, [text_message(f"讀取桌號失敗：{e}\n建議確認 token 是否有效，或直接輸入：房號 E5 / 桌號 R5037。")])


def parse_manual_table_id(action: str, session: Dict[str, Any]) -> str:
    text = (action or "").strip().upper()
    if not text:
        return ""
    # 支援：桌號 E5、房號 E5、房間 E5、桌廳 E5、ROOM E5、TABLE E5
    prefixes = ["桌號", "桌号", "房號", "房号", "房間", "房间", "桌廳", "桌厅", "ROOM", "TABLE", "NO.", "NO"]
    for prefix in prefixes:
        if text.startswith(prefix.upper()):
            text = text[len(prefix):].strip(" ：:=#-_")
            break
    # 在等待桌號階段，允許直接輸入 E5/R5037/RB05 這種短碼
    step = str(session.get("step") or "")
    if step not in {"TABLE_MANUAL_WAIT", "TABLE_SELECT", "TABLE_LOADING", "ANALYZING"} and action == text:
        return ""
    text = text.replace(" ", "").replace("－", "-")
    if 1 <= len(text) <= 30 and any(ch.isdigit() for ch in text):
        return text
    return ""


async def start_manual_table_monitor(user_id: str, table_id: str) -> List[Dict[str, Any]]:
    session = get_session(user_id)
    platform = session.get("platform")
    hall = session.get("hall")
    if not platform or not hall:
        return [text_message("請先選擇平台與遊戲廳。")]
    update_session(user_id, table_id=table_id, step="ANALYZING", running=True, table_input_mode="manual")
    await monitor_manager.start(user_id, platform, hall, table_id, on_update=push_analysis)
    return [build_loading_message(f"已指定房號/桌號 {table_id}，正在只針對此桌建立監控並讀取牌路...")]


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
        update_session(user_id, hall=hall, step="TABLE_MANUAL_WAIT")
        if AUTO_SCAN_TABLES_ON_HALL and not MANUAL_FIRST_TABLE_MODE:
            update_session(user_id, step="TABLE_LOADING")
            asyncio.create_task(push_tables_after_loading(user_id, platform, hall))
            return [build_loading_message("正在自動掃描真實桌號與荷官資料，請稍候...\n若讀取不到，請直接輸入：房號 E5")]
        return [build_manual_table_prompt_message(platform, hall)]

    if action == "TABLE:AUTO_SCAN":
        session = get_session(user_id)
        platform = session.get("platform")
        hall = session.get("hall")
        if not platform or not hall:
            return [text_message("請先選擇平台與遊戲廳。")]
        update_session(user_id, step="TABLE_LOADING")
        asyncio.create_task(push_tables_after_loading(user_id, platform, hall))
        return [build_loading_message("正在重新掃描目前遊戲廳牌桌，請稍候...")]

    if action.startswith("TABLE:"):
        table_id = action.split(":", 1)[1].strip().upper()
        session = get_session(user_id)
        platform = session.get("platform")
        hall = session.get("hall")
        if not platform or not hall:
            return [text_message("請先選擇平台與遊戲廳。")]
        update_session(user_id, table_id=table_id, step="ANALYZING", running=True)
        await monitor_manager.start(user_id, platform, hall, table_id, on_update=push_analysis)
        return [build_loading_message(f"已選擇牌桌 {table_id}，正在點入指定桌並只針對該桌紅藍綠牌路建立監控...")]

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

    session = get_session(user_id)
    manual_table_id = parse_manual_table_id(action, session)
    if manual_table_id:
        return await start_manual_table_monitor(user_id, manual_table_id)

    return [text_message("請輸入「開始預測」，或使用按鈕操作。若已選遊戲廳，可直接輸入：房號 E5")]


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

        # LINE 官方 loading animation：先顯示讀取中，再執行較慢的爬蟲/AI流程。
        # 只支援一對一 userId；group/room 會在 line_client.show_loading 裡自動略過。
        show_loading(source.get("userId") or user_id)

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
    update_session(user_id, hall=hall, step="TABLE_MANUAL_WAIT", tables=[])
    if AUTO_SCAN_TABLES_ON_HALL and not MANUAL_FIRST_TABLE_MODE:
        tables = await reader.list_tables(platform, hall)
        update_session(user_id, step="TABLE_SELECT", tables=tables)
        return {"ok": True, "session": get_session(user_id), "tables": tables, "manual_required": False}
    return {"ok": True, "session": get_session(user_id), "tables": [], "manual_required": True, "message": "請手動輸入房號或桌號"}


@app.post("/api/flow/tablescan")
async def api_flow_tablescan(payload: Dict[str, Any]):
    user_id = payload.get("user_id") or "local"
    session = get_session(user_id)
    platform = session.get("platform")
    hall = session.get("hall")
    if not platform or not hall:
        raise HTTPException(status_code=400, detail="請先選擇平台與遊戲廳")
    update_session(user_id, step="TABLE_LOADING")
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
    except LoginExpiredError as e:
        return {"ok": False, "stage": "baccarat_login_expired", "login_expired": True, "error": str(e), "diagnostics": e.diagnostics}
    except Exception as e:
        return {"ok": False, "stage": "baccarat_page_failed", "error": str(e)}
    finally:
        await reader.close_browser(playwright, browser)


@app.get("/api/debug/tables")
async def debug_tables(platform: str = "DG", hall: str = "BACCARAT"):
    try:
        tables = await reader.list_tables(platform, hall)
        return {"ok": True, "stage": "table_list_result", "table_count": len(tables), "tables": tables}
    except LoginExpiredError as e:
        return {"ok": False, "stage": "table_list_login_expired", "login_expired": True, "error": str(e), "diagnostics": e.diagnostics}
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
    except LoginExpiredError as e:
        return {"ok": False, "stage": "table_data_login_expired", "login_expired": True, "error": str(e), "diagnostics": e.diagnostics}
    except Exception as e:
        return {"ok": False, "stage": "table_data_failed", "error": str(e)}
