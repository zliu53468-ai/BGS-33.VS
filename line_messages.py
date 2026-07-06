# line_messages.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List

from config import enabled_platforms, get_platform, PUBLIC_BASE_URL


def text_message(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text[:4500]}


def postback_button(label: str, data: str, color: str = "#FFCC00", style: str = "primary") -> Dict[str, Any]:
    return {
        "type": "button",
        "style": style,
        "color": color,
        "height": "sm",
        "action": {"type": "postback", "label": label, "data": data, "displayText": label},
    }


def uri_button(label: str, uri: str, color: str = "#FFCC00") -> Dict[str, Any]:
    return {
        "type": "button",
        "style": "primary",
        "color": color,
        "height": "sm",
        "action": {"type": "uri", "label": label, "uri": uri},
    }


def build_guide_message() -> Dict[str, Any]:
    contents = [
        {"type": "text", "text": "富百家 AI Pro", "weight": "bold", "size": "xl", "color": "#D9A300", "align": "center"},
        {"type": "separator", "color": "#FFCC00"},
        {"type": "text", "text": "📍 操作流程\n開始預測 → 選平台 → 選遊戲廳 → 選桌號 → 自動分析", "wrap": True, "size": "md"},
        {"type": "text", "text": "✅ 本地牌路模型 + DeepSeek AI 校準\n會判斷長龍、單跳、雙跳、成對、散盤等型態。", "wrap": True, "size": "md"},
        postback_button("開始預測", "FLOW:START"),
    ]
    if PUBLIC_BASE_URL:
        contents.append(uri_button("開啟 LIFF 面板", f"{PUBLIC_BASE_URL}/static/index.html"))
    return {"type": "flex", "altText": "富百家 AI Pro", "contents": {"type": "bubble", "size": "mega", "body": {"type": "box", "layout": "vertical", "spacing": "md", "contents": contents}}}


def build_platform_message() -> Dict[str, Any]:
    platforms = enabled_platforms()
    if not platforms:
        return text_message("目前尚未設定平台網址，請先到 Render 新增 BACCARAT_URL_GSA / BACCARAT_URL_DG / BACCARAT_URL_REBIRTH。")
    bubbles = []
    for p in platforms:
        bubbles.append({
            "type": "bubble",
            "size": "micro",
            "body": {"type": "box", "layout": "vertical", "spacing": "sm", "contents": [
                {"type": "text", "text": p.name, "weight": "bold", "size": "lg", "align": "center", "color": "#D9A300", "wrap": True},
                {"type": "separator", "color": "#FFCC00"},
                postback_button("選擇", f"PLATFORM:{p.key}"),
            ]},
        })
    return {"type": "flex", "altText": "請選擇平台", "contents": {"type": "carousel", "contents": bubbles}}


def build_hall_message(platform_key: str) -> Dict[str, Any]:
    p = get_platform(platform_key)
    return {"type": "flex", "altText": "請選擇遊戲廳", "contents": {"type": "bubble", "size": "mega", "body": {"type": "box", "layout": "vertical", "spacing": "md", "contents": [
        {"type": "text", "text": p.name, "weight": "bold", "size": "xl", "color": "#D9A300", "align": "center"},
        {"type": "text", "text": "請選擇遊戲廳", "weight": "bold", "size": "md", "align": "center"},
        postback_button("經典百家樂", "HALL:BACCARAT"),
        postback_button("龍虎門", "HALL:DRAGON_TIGER"),
    ]}}}


def build_table_message(tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not tables:
        return text_message("目前沒有讀取到平台真實桌號。這代表網頁文字/API 尚未抓到，請先測 /api/debug/baccarat 或檢查 token 是否進入大廳。")
    bubbles = []
    for t in tables[:10]:
        table_id = t.get("table_id", "")
        game_no = t.get("game_no", "") or "讀取中"
        dealer = t.get("dealer", "") or "讀取中"
        online = t.get("online_count", 0)
        source = t.get("source", "")
        bubbles.append({
            "type": "bubble",
            "size": "micro",
            "body": {"type": "box", "layout": "vertical", "spacing": "sm", "contents": [
                {"type": "text", "text": table_id, "weight": "bold", "size": "xl", "align": "center", "color": "#D9A300"},
                {"type": "separator", "color": "#FFCC00"},
                {"type": "text", "text": f"遊戲編號\n{game_no}", "size": "xs", "wrap": True},
                {"type": "text", "text": f"荷官姓名：{dealer}", "size": "xs", "wrap": True},
                {"type": "text", "text": f"在線人數：{online}", "size": "xs", "wrap": True},
                {"type": "text", "text": f"來源：{source}", "size": "xxs", "color": "#888888", "wrap": True},
                postback_button("選擇", f"TABLE:{table_id}"),
            ]},
        })
    return {"type": "flex", "altText": "請選擇桌號", "contents": {"type": "carousel", "contents": bubbles}}


def build_analysis_message(data: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    recommend = prediction.get("recommend", "觀望")
    rec_color = "#0047FF" if recommend == "閒" else "#D60000" if recommend == "莊" else "#333333"
    ai_txt = "已啟用" if prediction.get("ai_used") else "未啟用/未成功"
    contents = [
        {"type": "box", "layout": "vertical", "backgroundColor": "#FFCC00", "cornerRadius": "md", "paddingAll": "10px", "contents": [{"type": "text", "text": "分析數據", "weight": "bold", "size": "xl", "align": "center", "color": "#333333"}]},
        {"type": "text", "text": f"桌號：{data.get('table_id', '讀取中')}", "size": "md", "wrap": True},
        {"type": "text", "text": f"遊戲編號：{data.get('game_no') or '讀取中'}", "size": "md", "wrap": True},
        {"type": "text", "text": f"荷官姓名：{data.get('dealer') or '讀取中'}", "size": "md", "wrap": True},
        {"type": "text", "text": f"倒數計時：{data.get('countdown', 0)} 秒｜狀態：{data.get('status', '讀取中')}", "size": "md", "wrap": True},
        {"type": "separator", "color": "#FFCC00"},
        {"type": "text", "text": f"莊　{prediction.get('banker_percent', 33)}%", "weight": "bold", "size": "xl", "color": "#D60000"},
        {"type": "text", "text": f"閒　{prediction.get('player_percent', 33)}%", "weight": "bold", "size": "xl", "color": "#0047FF"},
        {"type": "text", "text": f"和　{prediction.get('tie_percent', 34)}%", "weight": "bold", "size": "xl", "color": "#00A12A"},
        {"type": "box", "layout": "horizontal", "backgroundColor": "#FFCC00", "cornerRadius": "md", "paddingAll": "10px", "contents": [
            {"type": "text", "text": "推薦", "weight": "bold", "size": "xl", "color": "#333333"},
            {"type": "text", "text": recommend, "weight": "bold", "size": "xl", "align": "end", "color": rec_color},
        ]},
        {"type": "text", "text": f"牌路：{prediction.get('pattern_detail', '')}", "size": "xs", "wrap": True, "color": "#666666"},
        {"type": "text", "text": f"原因：{prediction.get('reason', '')}", "size": "xs", "wrap": True, "color": "#666666"},
        {"type": "text", "text": f"DeepSeek：{ai_txt}｜{prediction.get('ai_reason', '')}", "size": "xs", "wrap": True, "color": "#666666"},
        postback_button("繼續分析", "ANALYZE:CONTINUE"),
        postback_button("結束分析", "ANALYZE:STOP", color="#222222"),
    ]
    return {"type": "flex", "altText": "分析數據", "contents": {"type": "bubble", "size": "mega", "body": {"type": "box", "layout": "vertical", "spacing": "md", "contents": contents}}}
