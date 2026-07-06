from typing import Any, Dict, List

from config import enabled_platforms, get_platform


def text_message(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def postback_button(label: str, data: str, color: str = "#FFCC00") -> Dict[str, Any]:
    return {
        "type": "button",
        "style": "primary",
        "color": color,
        "height": "sm",
        "action": {
            "type": "postback",
            "label": label,
            "data": data,
            "displayText": label,
        },
    }


def build_guide_message() -> Dict[str, Any]:
    return {
        "type": "flex",
        "altText": "富百家使用指南",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "md",
                "contents": [
                    {
                        "type": "text",
                        "text": "富百家使用指南",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#444444",
                    },
                    {"type": "separator", "color": "#FFCC00"},
                    {
                        "type": "text",
                        "text": "📍 操作 3 步驟\n同步桌號 → 選擇平台 → 自動分析",
                        "wrap": True,
                        "size": "md",
                    },
                    {
                        "type": "text",
                        "text": "⚠️ 玩家 4 守則\n非合作平台不分析、嚴禁梭哈、紀律停利、程式綁定。",
                        "wrap": True,
                        "size": "md",
                    },
                    {
                        "type": "text",
                        "text": "✅ 本金規劃\n請將本金分成 20–30 等份，穩定分析。",
                        "wrap": True,
                        "size": "md",
                    },
                    postback_button("開始預測", "FLOW:START"),
                ],
            },
        },
    }


def build_platform_message() -> Dict[str, Any]:
    platforms = enabled_platforms()

    if not platforms:
        return text_message("尚未設定平台網址，請先到 Render Environment Variables 設定 BACCARAT_URL_GSA / BACCARAT_URL_DG / BACCARAT_URL_REBIRTH。")

    bubbles: List[Dict[str, Any]] = []

    for platform in platforms:
        bubbles.append(
            {
                "type": "bubble",
                "size": "micro",
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "text",
                            "text": platform.name,
                            "weight": "bold",
                            "size": "lg",
                            "color": "#D9A300",
                            "align": "center",
                            "wrap": True,
                        },
                        {"type": "separator", "color": "#FFCC00"},
                        postback_button("選擇", f"PLATFORM:{platform.key}"),
                    ],
                },
            }
        )

    return {
        "type": "flex",
        "altText": "請選擇平台",
        "contents": {"type": "carousel", "contents": bubbles},
    }


def build_hall_message(platform_key: str) -> Dict[str, Any]:
    platform = get_platform(platform_key)

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
                    {
                        "type": "text",
                        "text": platform.name,
                        "weight": "bold",
                        "size": "xl",
                        "color": "#D9A300",
                        "align": "center",
                    },
                    {
                        "type": "text",
                        "text": "請選擇遊戲廳",
                        "weight": "bold",
                        "size": "md",
                        "align": "center",
                    },
                    postback_button("經典百家樂", "HALL:BACCARAT"),
                    postback_button("龍虎門", "HALL:DRAGON_TIGER"),
                ],
            },
        },
    }


def build_table_message(tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    bubbles: List[Dict[str, Any]] = []

    for table in tables[:10]:
        table_id = str(table.get("table_id") or "").strip() or "TABLE"
        game_no = str(table.get("game_no") or "讀取中")
        dealer = str(table.get("dealer") or "讀取中")
        online_count = table.get("online_count", 0)

        bubbles.append(
            {
                "type": "bubble",
                "size": "micro",
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "text",
                            "text": table_id,
                            "weight": "bold",
                            "size": "xl",
                            "align": "center",
                            "color": "#D9A300",
                        },
                        {"type": "separator", "color": "#FFCC00"},
                        {"type": "text", "text": f"遊戲編號\n{game_no}", "size": "xs", "wrap": True},
                        {"type": "text", "text": f"荷官姓名：{dealer}", "size": "xs", "wrap": True},
                        {"type": "text", "text": f"在線人數：{online_count}", "size": "xs", "wrap": True},
                        postback_button("選擇", f"TABLE:{table_id}"),
                    ],
                },
            }
        )

    return {
        "type": "flex",
        "altText": "請選擇桌號",
        "contents": {"type": "carousel", "contents": bubbles},
    }


def build_analysis_message(data: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    table_id = data.get("table_id", "讀取中") or "讀取中"
    game_no = data.get("game_no", "讀取中") or "讀取中"
    countdown = data.get("countdown", 0)
    status = data.get("status", "讀取中") or "讀取中"
    road = data.get("road", []) or []

    banker_percent = prediction.get("banker_percent", 33)
    player_percent = prediction.get("player_percent", 33)
    tie_percent = prediction.get("tie_percent", 34)
    recommend = prediction.get("recommend", "觀望")
    reason = prediction.get("reason", "")

    recommend_color = "#0047FF" if recommend == "閒" else "#D60000" if recommend == "莊" else "#333333"

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
                    {
                        "type": "box",
                        "layout": "vertical",
                        "backgroundColor": "#FFCC00",
                        "cornerRadius": "md",
                        "paddingAll": "10px",
                        "contents": [
                            {
                                "type": "text",
                                "text": "分析數據",
                                "weight": "bold",
                                "size": "xl",
                                "align": "center",
                                "color": "#333333",
                            }
                        ],
                    },
                    {"type": "text", "text": f"桌號：{table_id}", "size": "md", "wrap": True},
                    {"type": "text", "text": f"遊戲編號：{game_no}", "size": "md", "wrap": True},
                    {"type": "text", "text": f"倒數計時：{countdown} 秒", "size": "md", "wrap": True},
                    {"type": "text", "text": f"遊戲狀態：{status}", "size": "md", "wrap": True},
                    {"type": "text", "text": f"目前牌路：{''.join(road[-30:]) or '讀取中'}", "size": "xs", "wrap": True, "color": "#666666"},
                    {"type": "separator", "color": "#FFCC00"},
                    {"type": "text", "text": f"莊　{banker_percent}%", "weight": "bold", "size": "xl", "color": "#D60000"},
                    {"type": "text", "text": f"閒　{player_percent}%", "weight": "bold", "size": "xl", "color": "#0047FF"},
                    {"type": "text", "text": f"和　{tie_percent}%", "weight": "bold", "size": "xl", "color": "#00A12A"},
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "backgroundColor": "#FFCC00",
                        "cornerRadius": "md",
                        "paddingAll": "10px",
                        "contents": [
                            {"type": "text", "text": "推薦", "weight": "bold", "size": "xl", "color": "#333333"},
                            {"type": "text", "text": recommend, "weight": "bold", "size": "xl", "align": "end", "color": recommend_color},
                        ],
                    },
                    {"type": "text", "text": reason, "size": "xs", "wrap": True, "color": "#666666"},
                    postback_button("繼續分析", "ANALYZE:CONTINUE"),
                    postback_button("結束分析", "ANALYZE:STOP", color="#222222"),
                ],
            },
        },
    }
