# line_messages.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List

from config import enabled_platforms, get_platform, public_url


def text_message(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text[:4500]}


def postback_button(label: str, data: str, color: str = "#FFCC00") -> Dict[str, Any]:
    return {
        "type": "button",
        "style": "primary",
        "height": "sm",
        "color": color,
        "action": {
            "type": "postback",
            "label": label,
            "data": data,
            "displayText": label,
        },
    }


def uri_button(label: str, uri: str, color: str = "#FFCC00") -> Dict[str, Any]:
    return {
        "type": "button",
        "style": "primary",
        "height": "sm",
        "color": color,
        "action": {"type": "uri", "label": label, "uri": uri},
    }


def build_guide_message() -> Dict[str, Any]:
    contents: List[Dict[str, Any]] = [
        {"type": "text", "text": "富百家 AI Pro", "weight": "bold", "size": "xl", "align": "center", "color": "#D9A300"},
        {"type": "separator", "color": "#FFCC00"},
        {"type": "text", "text": "📍 操作流程\n開始預測 → 選平台 → 選遊戲廳 → 選桌號 → 自動分析", "wrap": True, "size": "md"},
        {"type": "text", "text": "✅ 本地牌路模型 + DeepSeek AI 校準\n會判斷長龍、單跳、雙跳、成對、散盤等型態。", "wrap": True, "size": "md"},
        postback_button("開始預測", "FLOW:START"),
    ]
    liff_url = public_url("/static/index.html")
    if liff_url.startswith("http"):
        contents.append(uri_button("開啟 LIFF 面板", liff_url))
    return {
        "type": "flex",
        "altText": "富百家 AI Pro 使用指南",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {"type": "box", "layout": "vertical", "spacing": "md", "contents": contents},
        },
    }


def build_platform_message() -> Dict[str, Any]:
    platforms = enabled_platforms()
    if not platforms:
        return text_message("目前尚未設定平台網址，請先到 Render 新增 BACCARAT_URL_GSA / BACCARAT_URL_DG / BACCARAT_URL_REBIRTH。")
    bubbles = []
    for p in platforms:
        bubbles.append({
            "type": "bubble",
            "size": "micro",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {"type": "text", "text": p.name, "weight": "bold", "size": "lg", "align": "center", "color": "#D9A300", "wrap": True},
                    {"type": "separator", "color": "#FFCC00"},
                    postback_button("選擇", f"PLATFORM:{p.key}"),
                ],
            },
        })
    return {"type": "flex", "altText": "請選擇平台", "contents": {"type": "carousel", "contents": bubbles}}


def build_hall_message(platform_key: str) -> Dict[str, Any]:
    p = get_platform(platform_key)
    return {
        "type": "flex",
        "altText": "請選擇遊戲廳",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "md",
                "contents": [
                    {"type": "text", "text": p.name, "weight": "bold", "size": "xl", "align": "center", "color": "#D9A300"},
                    {"type": "text", "text": "請選擇遊戲廳", "weight": "bold", "align": "center"},
                    postback_button("經典百家樂", "HALL:BACCARAT"),
                    postback_button("龍虎門", "HALL:DRAGON_TIGER"),
                ],
            },
        },
    }


def build_loading_message(text: str = "正在讀取資料，請稍候...") -> Dict[str, Any]:
    return text_message(text)


def build_table_message(tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not tables:
        return text_message("目前沒有讀取到真實桌號。請確認平台網址 token 是否有效，或用 /api/debug/tables 檢查資料來源。")
    bubbles = []
    for t in tables[:12]:
        tid = str(t.get("table_id") or "").strip()
        game_no = str(t.get("game_no") or "讀取中")
        dealer = str(t.get("dealer") or "讀取中")
        online = str(t.get("online_count") or 0)
        source = str(t.get("source") or "")
        bubbles.append({
            "type": "bubble",
            "size": "micro",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {"type": "text", "text": tid or "未命名", "weight": "bold", "size": "xl", "align": "center", "color": "#D9A300", "wrap": True},
                    {"type": "separator", "color": "#FFCC00"},
                    {"type": "text", "text": f"遊戲編號\n{game_no}", "size": "xs", "wrap": True},
                    {"type": "text", "text": f"荷官姓名：{dealer}", "size": "xs", "wrap": True},
                    {"type": "text", "text": f"在線人數：{online}", "size": "xs", "wrap": True},
                    {"type": "text", "text": f"來源：{source}", "size": "xxs", "color": "#888888", "wrap": True},
                    postback_button("選擇", f"TABLE:{tid}"),
                ],
            },
        })
    return {"type": "flex", "altText": "請選擇桌號", "contents": {"type": "carousel", "contents": bubbles}}


def build_analysis_message(data: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    table_id = data.get("table_id") or "讀取中"
    game_no = data.get("game_no") or "讀取中"
    countdown = data.get("countdown", 0)
    status = data.get("status") or "讀取中"
    road = data.get("road") or []
    road_text = "".join(road[-30:]) if isinstance(road, list) else str(road)[-30:]
    ai_state = "DeepSeek 已校準" if prediction.get("ai_used") else "DeepSeek 未啟用/未參與"

    return {
        "type": "flex",
        "altText": "分析數據",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "md",
                "contents": [
                    {"type": "box", "layout": "vertical", "backgroundColor": "#FFCC00", "cornerRadius": "md", "paddingAll": "10px", "contents": [
                        {"type": "text", "text": "分析數據", "weight": "bold", "size": "xl", "align": "center", "color": "#333333"}
                    ]},
                    {"type": "text", "text": f"桌號：{table_id}", "wrap": True},
                    {"type": "text", "text": f"遊戲編號：{game_no}", "wrap": True},
                    {"type": "text", "text": f"倒數計時：{countdown} 秒", "wrap": True},
                    {"type": "text", "text": f"遊戲狀態：{status}", "wrap": True},
                    {"type": "text", "text": f"近局牌路：{road_text or '尚未讀取'}", "wrap": True, "size": "sm", "color": "#666666"},
                    {"type": "separator", "color": "#FFCC00"},
                    {"type": "text", "text": f"莊　{prediction.get('banker_percent', 33)}%", "weight": "bold", "size": "xl", "color": "#D60000"},
                    {"type": "text", "text": f"閒　{prediction.get('player_percent', 33)}%", "weight": "bold", "size": "xl", "color": "#0047FF"},
                    {"type": "text", "text": f"和　{prediction.get('tie_percent', 34)}%", "weight": "bold", "size": "xl", "color": "#00A12A"},
                    {"type": "box", "layout": "horizontal", "backgroundColor": "#FFCC00", "cornerRadius": "md", "paddingAll": "10px", "contents": [
                        {"type": "text", "text": "推薦", "weight": "bold", "size": "xl", "color": "#333333"},
                        {"type": "text", "text": str(prediction.get('recommend', '觀望')), "weight": "bold", "size": "xl", "align": "end", "color": "#0047FF" if prediction.get('recommend') == "閒" else "#D60000" if prediction.get('recommend') == "莊" else "#333333"},
                    ]},
                    {"type": "text", "text": f"牌路型態：{prediction.get('pattern_detail', '讀取中')}", "size": "xs", "wrap": True, "color": "#555555"},
                    {"type": "text", "text": f"AI狀態：{ai_state}", "size": "xs", "wrap": True, "color": "#555555"},
                    {"type": "text", "text": str(prediction.get("reason", ""))[:500], "size": "xs", "wrap": True, "color": "#666666"},
                    postback_button("繼續分析", "ANALYZE:CONTINUE"),
                    postback_button("結束分析", "ANALYZE:STOP", color="#222222"),
                ],
            },
        },
    }
