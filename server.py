import base64
import hashlib
import hmac
import os
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from config import LINE_CHANNEL_SECRET, ENABLE_SIGNATURE_VERIFY, GAME_MAP, DEFAULT_GAME, DEFAULT_TABLE
from session_store import store
from predictor import predict
from gemini_helper import explain
from line_api import reply_messages, text_message
from message_builder import (
    game_menu_text,
    table_connecting_text,
    table_connected_text,
    ask_points_text,
    start_text,
    end_text,
    read_done_text,
    prediction_text,
    help_text,
)
from parser_utils import parse_points, looks_like_table_id
from point_db import point_db_meta
from pattern_db import pattern_db_meta

app = Flask(__name__)
CORS(app)

def verify_line_signature(body: bytes, signature: str) -> bool:
    if not ENABLE_SIGNATURE_VERIFY:
        return True
    if not LINE_CHANNEL_SECRET:
        return False

    digest = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)

@app.get("/")
def home():
    return "OK - BGS Dual 3M DB LINE BOT is running"

@app.get("/health")
def health():
    pm = point_db_meta()
    rm = pattern_db_meta()
    return jsonify({
        "ok": True,
        "service": "BGS_DUAL_3M_DB_LINE_BOT",
        "sessions": store.all_count(),
        "mode": "dual_3m_point_and_result_pattern_no_observe",
        "point_db_samples": pm.get("total_simulated_samples"),
        "pattern_db_samples": rm.get("total_simulated_samples"),
    })

@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True)
    player_point = int(data.get("player_point"))
    banker_point = int(data.get("banker_point"))
    rounds = data.get("rounds", [])
    result = predict(player_point, banker_point, rounds)
    ai_text = explain(result)
    return jsonify({**result, "ai_text": ai_text})

@app.post("/webhook")
def webhook():
    body = request.get_data()
    signature = request.headers.get("X-Line-Signature", "")

    if not verify_line_signature(body, signature):
        abort(400)

    payload = request.get_json(force=True)
    for event in payload.get("events", []):
        handle_event(event)

    return "OK"

def handle_event(event: dict):
    reply_token = event.get("replyToken")
    if not reply_token:
        return

    source = event.get("source", {})
    user_id = source.get("userId") or source.get("groupId") or source.get("roomId") or "anonymous"

    if event.get("type") == "postback":
        data = event.get("postback", {}).get("data", "")
        return handle_text(user_id, reply_token, data)

    if event.get("type") != "message":
        return

    msg = event.get("message", {})
    if msg.get("type") != "text":
        reply_messages(reply_token, [text_message("目前只支援文字輸入點數，例如：65")])
        return

    return handle_text(user_id, reply_token, msg.get("text", "").strip())

def handle_text(user_id: str, reply_token: str, text: str):
    sess = store.get(user_id)
    raw = text.strip()

    if raw in {"help", "幫助", "說明", "指令"}:
        reply_messages(reply_token, [text_message(help_text())])
        return

    if raw in {"遊戲設定", "設定遊戲", "遊戲館別", "館別設定"}:
        sess.phase = "choose_game"
        reply_messages(reply_token, [text_message(game_menu_text())])
        return

    if sess.phase == "choose_game" and raw in GAME_MAP:
        sess.game = GAME_MAP[raw]
        sess.phase = "need_table"
        reply_messages(reply_token, [
            text_message(f"✅ 已設定遊戲類別【{sess.game}】"),
            text_message("🎯 請輸入需預測桌號 (Ex:DG01)")
        ])
        return

    if sess.phase == "need_table" and looks_like_table_id(raw):
        sess.table = raw.upper()
        sess.phase = "idle"
        sess.active = True
        reply_messages(reply_token, [
            text_message(table_connecting_text()),
            text_message(table_connected_text()),
            text_message(ask_points_text()),
        ])
        return

    if raw in {"開始分析", "開始", "啟動分析"}:
        sess.active = True
        if not sess.game:
            sess.game = DEFAULT_GAME
        if not sess.table:
            sess.table = DEFAULT_TABLE
        reply_messages(reply_token, [text_message(start_text(sess.game, sess.table))])
        return

    if raw in {"結束分析", "結束", "停止分析", "停止"}:
        sess.active = False
        reply_messages(reply_token, [text_message(end_text())])
        return

    if raw in {"重置", "清空", "reset"}:
        store.reset(user_id, keep_setting=True)
        reply_messages(reply_token, [text_message("♻️ 已重置本輪資料\n請直接輸入點數，例如：65")])
        return

    points = parse_points(raw)
    if points:
        player_point, banker_point = points
        sess.active = True

        # 先把本局結果放入臨時rounds，讓pattern能包含最新一局。
        last_result = "閒" if player_point > banker_point else ("莊" if banker_point > player_point else "和")
        temp_rounds = sess.rounds + [{
            "player_point": player_point,
            "banker_point": banker_point,
            "last_result": last_result,
        }]

        pred = predict(player_point, banker_point, temp_rounds)
        ai_text = explain(pred)

        round_data = {
            "player_point": player_point,
            "banker_point": banker_point,
            "last_result": pred["last_result"],
            "recommend": pred["recommend"],
            "player_prob": pred["player_prob"],
            "banker_prob": pred["banker_prob"],
        }

        # 只保留最近30局供莊閒pattern查詢；不是用來累計點數權重。
        sess.last_round = round_data
        sess.rounds.append(round_data)
        sess.rounds = sess.rounds[-30:]

        reply_messages(reply_token, [
            text_message(read_done_text(player_point, banker_point)),
            text_message(prediction_text(pred, ai_text)),
        ])
        return

    reply_messages(reply_token, [
        text_message(
            "⚠️ 格式錯誤\n"
            "請直接輸入點數，例如：65\n"
            "規則：先輸入閒，再輸入莊。\n"
            "也可輸入「遊戲設定」重新設定館別。"
        )
    ])

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
