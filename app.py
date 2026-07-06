import json
import re
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from baccarat_reader import BaccaratReader
from line_client import reply_message, verify_signature
from line_messages import (
    build_analysis_message,
    build_guide_message,
    build_hall_message,
    build_platform_message,
    build_table_message,
    text_message,
)
from monitor import MonitorManager
from predictor import predict
from session_store import get_session, reset_session, update_session

app = FastAPI(title="富百家 AI Pro Bot")
reader = BaccaratReader()
monitor_manager = MonitorManager(reader)


@app.get("/")
async def root():
    return PlainTextResponse("富百家 AI Pro Bot is running.")


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


async def analyze_current_session(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id)
    platform = session.get("platform")
    hall = session.get("hall")
    table_id = session.get("table_id")

    if not platform or not hall or not table_id:
        return {"error": "請先選擇平台、遊戲廳與桌號。"}

    data = await reader.read_table_data(platform_key=platform, hall_key=hall, table_id=table_id)
    road = data.get("road", [])
    prediction = predict(road)

    update_session(
        user_id,
        game_no=data.get("game_no"),
        dealer=data.get("dealer"),
        online_count=data.get("online_count"),
        road=road,
        last_round_key=data.get("round_key"),
        running=True,
        step="ANALYZING",
    )

    return {"data": data, "prediction": prediction}


async def handle_action(user_id: str, action: str) -> List[Dict[str, Any]]:
    action = (action or "").strip()

    if action in ("開始預測", "開始分析", "開始", "START"):
        reset_session(user_id)
        return [build_guide_message()]

    if action == "FLOW:START":
        update_session(user_id, step="PLATFORM_SELECT", running=False)
        return [build_platform_message()]

    if action.startswith("PLATFORM:"):
        platform_key = action.split(":", 1)[1].strip().upper()
        update_session(user_id, platform=platform_key, step="HALL_SELECT")
        try:
            return [build_hall_message(platform_key)]
        except Exception as exc:
            return [text_message(f"平台設定錯誤：{exc}")]

    if action.startswith("HALL:"):
        hall_key = action.split(":", 1)[1].strip().upper()
        session = get_session(user_id)
        platform_key = session.get("platform")

        if not platform_key:
            return [text_message("請先選擇平台。")]

        update_session(user_id, hall=hall_key, step="TABLE_SELECT")

        try:
            tables = await reader.list_tables(platform_key=platform_key, hall_key=hall_key)
            return [build_table_message(tables)]
        except Exception as exc:
            return [text_message(f"讀取桌號失敗：{exc}\n可先手動輸入：桌號 RB05")]

    if action.startswith("TABLE:"):
        table_id = action.split(":", 1)[1].strip().upper()
        update_session(user_id, table_id=table_id, step="ANALYZING", running=True)

        result = await analyze_current_session(user_id)
        await monitor_manager.start(user_id)

        if "error" in result:
            return [text_message(result["error"])]

        return [build_analysis_message(result["data"], result["prediction"])]

    table_match = re.match(r"^(?:桌號|桌号)\s*([A-Za-z0-9_-]+)$", action)
    if table_match:
        table_id = table_match.group(1).strip().upper()
        update_session(user_id, table_id=table_id, step="ANALYZING", running=True)

        result = await analyze_current_session(user_id)
        await monitor_manager.start(user_id)

        if "error" in result:
            return [text_message(result["error"])]

        return [build_analysis_message(result["data"], result["prediction"])]

    if action in ("ANALYZE:CONTINUE", "繼續分析"):
        result = await analyze_current_session(user_id)
        if "error" in result:
            return [text_message(result["error"])]
        return [build_analysis_message(result["data"], result["prediction"])]

    if action in ("ANALYZE:STOP", "結束分析"):
        await monitor_manager.stop(user_id)
        update_session(user_id, running=False, step="STOPPED")
        return [text_message("已結束分析。需要重新開始時，請輸入：開始預測")]

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
        return JSONResponse({"ok": False, "error": "Invalid JSON"})

    events = payload.get("events", [])

    for event in events:
        reply_token = event.get("replyToken")
        source = event.get("source", {})
        user_id = source.get("userId") or source.get("groupId") or source.get("roomId")

        if not reply_token or not user_id:
            continue

        action = ""
        if event.get("type") == "message":
            message = event.get("message", {})
            if message.get("type") == "text":
                action = message.get("text", "").strip()
        elif event.get("type") == "postback":
            action = event.get("postback", {}).get("data", "").strip()

        if not action:
            reply_message(reply_token, [text_message("請輸入「開始預測」。")])
            continue

        try:
            messages = await handle_action(user_id, action)
            reply_message(reply_token, messages)
        except Exception as exc:
            print("webhook handle error:", exc)
            reply_message(reply_token, [text_message(f"系統讀取失敗：{exc}")])

    return JSONResponse({"ok": True})


@app.post("/api/test-action")
async def api_test_action(payload: Dict[str, Any]):
    user_id = payload.get("user_id", "TEST_USER")
    action = payload.get("action", "開始預測")
    messages = await handle_action(user_id, action)
    return {"ok": True, "messages": messages, "session": get_session(user_id)}


@app.post("/api/test-analyze")
async def api_test_analyze(payload: Dict[str, Any]):
    platform = payload.get("platform", "DG")
    hall = payload.get("hall", "BACCARAT")
    table_id = payload.get("table_id", "RB05")

    data = await reader.read_table_data(platform_key=platform, hall_key=hall, table_id=table_id)
    prediction = predict(data.get("road", []))

    return {"ok": True, "data": data, "prediction": prediction}
