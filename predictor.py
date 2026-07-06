# predictor.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import math


VALID_RESULTS = {"B", "P", "T"}


def normalize_road(road: Any) -> List[str]:
    """
    將各種格式統一轉成：
    B = 莊
    P = 閒
    T = 和
    """

    if road is None:
        return []

    # 字串格式：例如 "BPBTTPB"
    if isinstance(road, str):
        raw_items = []
        text = road.strip().upper()

        for ch in text:
            if ch in ("B", "P", "T", "莊", "庄", "閒", "闲", "和"):
                raw_items.append(ch)
    else:
        raw_items = road

    cleaned: List[str] = []

    for item in raw_items:
        if isinstance(item, dict):
            value = (
                item.get("result")
                or item.get("side")
                or item.get("value")
                or item.get("text")
                or ""
            )
        else:
            value = item

        value = str(value).strip().upper()

        if value in ("B", "BANKER", "莊", "庄"):
            cleaned.append("B")
        elif value in ("P", "PLAYER", "閒", "闲"):
            cleaned.append("P")
        elif value in ("T", "TIE", "和"):
            cleaned.append("T")

    return cleaned


def side_text(side: str) -> str:
    if side == "B":
        return "莊"
    if side == "P":
        return "閒"
    if side == "T":
        return "和"
    return "觀望"


def opposite(side: str) -> str:
    if side == "B":
        return "P"
    if side == "P":
        return "B"
    return ""


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def get_non_tie(road: List[str]) -> List[str]:
    return [x for x in road if x in ("B", "P")]


def get_segments(non_tie: List[str]) -> List[Tuple[str, int]]:
    """
    將牌路轉成段落：
    B B B P P B
    => [("B", 3), ("P", 2), ("B", 1)]
    """
    if not non_tie:
        return []

    segments: List[Tuple[str, int]] = []
    current = non_tie[0]
    count = 1

    for x in non_tie[1:]:
        if x == current:
            count += 1
        else:
            segments.append((current, count))
            current = x
            count = 1

    segments.append((current, count))
    return segments


def switch_rate(non_tie: List[str], window: int = 8) -> float:
    recent = non_tie[-window:]

    if len(recent) < 2:
        return 0.0

    switches = 0

    for a, b in zip(recent, recent[1:]):
        if a != b:
            switches += 1

    return switches / max(1, len(recent) - 1)


def repeat_rate(non_tie: List[str], window: int = 8) -> float:
    recent = non_tie[-window:]

    if len(recent) < 2:
        return 0.0

    repeats = 0

    for a, b in zip(recent, recent[1:]):
        if a == b:
            repeats += 1

    return repeats / max(1, len(recent) - 1)


def find_pattern_replay(non_tie: List[str]) -> Optional[Dict[str, Any]]:
    """
    找目前尾段規律，在歷史前面是否出現過。
    例如目前最後 6 手是 BPBBPP，
    去前面的牌路找 BPBBPP 後面通常接 B 還是 P。
    """

    n = len(non_tie)

    if n < 14:
        return None

    best = None

    # 從長規律往短規律找
    for size in range(10, 3, -1):
        if n <= size + 2:
            continue

        suffix = non_tie[-size:]
        next_counter = Counter()

        # 不包含最後這段本身
        for i in range(0, n - size - 1):
            segment = non_tie[i:i + size]

            if segment == suffix:
                next_side = non_tie[i + size]

                if next_side in ("B", "P"):
                    next_counter[next_side] += 1

        total = sum(next_counter.values())

        if total < 2:
            continue

        side, count = next_counter.most_common(1)[0]
        hit_rate = count / total

        if hit_rate >= 0.58:
            best = {
                "name": "PATTERN_REPLAY",
                "side": side,
                "confidence": round(hit_rate, 3),
                "weight": min(2.8, 1.4 + hit_rate),
                "reason": f"找到尾段 {size} 手重複規律，歷史相似段後續偏向「{side_text(side)}」{round(hit_rate * 100)}%。",
                "samples": total,
                "window": size,
            }
            break

    return best


def detect_road_patterns(road: List[str]) -> List[Dict[str, Any]]:
    """
    核心牌路判斷：
    1. 長龍
    2. 長龍疲勞 / 斷龍風險
    3. 單跳 / 一跳一
    4. 雙跳 / 成對
    5. 重複規律
    6. 近局偏向
    7. 散盤觀望
    """

    signals: List[Dict[str, Any]] = []
    non_tie = get_non_tie(road)

    if len(non_tie) < 5:
        signals.append({
            "name": "NOT_ENOUGH_DATA",
            "side": None,
            "confidence": 0.0,
            "weight": 0.0,
            "reason": "有效莊閒資料少於 5 手，先觀望。",
        })
        return signals

    segments = get_segments(non_tie)
    last_side = non_tie[-1]
    last_segment_side, last_segment_len = segments[-1]

    # 1. 長龍判斷
    if last_segment_len >= 4:
        if last_segment_len <= 5:
            conf = min(0.68, 0.56 + last_segment_len * 0.025)
            signals.append({
                "name": "LONG_DRAGON_FOLLOW",
                "side": last_segment_side,
                "confidence": conf,
                "weight": 2.1,
                "reason": f"目前出現 {side_text(last_segment_side)} {last_segment_len} 連，屬於長龍延伸型態，短線偏續龍。",
            })
        else:
            # 超過 6 連時，不硬追，給斷龍風險
            op = opposite(last_segment_side)
            conf = min(0.62, 0.48 + last_segment_len * 0.015)
            signals.append({
                "name": "DRAGON_FATIGUE",
                "side": op,
                "confidence": conf,
                "weight": 1.2,
                "reason": f"{side_text(last_segment_side)} 已連續 {last_segment_len} 手，進入長龍疲勞區，需注意斷龍風險。",
            })

    # 2. 小龍 / 2-3 連續
    elif last_segment_len in (2, 3):
        conf = 0.53 + last_segment_len * 0.02
        signals.append({
            "name": "SHORT_STREAK",
            "side": last_segment_side,
            "confidence": conf,
            "weight": 1.1,
            "reason": f"目前為 {side_text(last_segment_side)} {last_segment_len} 連，小龍偏延續，但信號不算強。",
        })

    # 3. 單跳 / 一跳一
    sr8 = switch_rate(non_tie, 8)
    sr10 = switch_rate(non_tie, 10)

    if sr8 >= 0.78 and len(non_tie) >= 8:
        op = opposite(last_side)
        signals.append({
            "name": "CHOP_PATTERN",
            "side": op,
            "confidence": min(0.72, 0.55 + sr8 * 0.18),
            "weight": 2.3,
            "reason": f"近 8 手切換率 {round(sr8 * 100)}%，呈現單跳 / 一跳一，下一手偏向「{side_text(op)}」。",
        })
    elif sr10 >= 0.65 and len(non_tie) >= 10:
        op = opposite(last_side)
        signals.append({
            "name": "LIGHT_CHOP",
            "side": op,
            "confidence": min(0.64, 0.52 + sr10 * 0.13),
            "weight": 1.5,
            "reason": f"近 10 手有偏跳格型態，下一手略偏「{side_text(op)}」。",
        })

    # 4. 雙跳 / 成對 AABB
    if len(segments) >= 4:
        recent_segments = segments[-5:]
        recent_lengths = [length for _, length in recent_segments]

        # 判斷前面是否多段接近 2
        pair_like_count = sum(1 for x in recent_lengths[:-1] if x in (2, 3))

        if pair_like_count >= 3:
            if last_segment_len == 1:
                # 例如 BB PP BB P => 下一手偏 P 補成 PP
                signals.append({
                    "name": "PAIR_COMPLETION",
                    "side": last_segment_side,
                    "confidence": 0.63,
                    "weight": 2.0,
                    "reason": f"近期多次出現成對節奏，當前 {side_text(last_segment_side)} 只有 1 手，偏向補成一對。",
                })
            elif last_segment_len >= 2:
                # 例如 BB PP BB PP => 下一手偏換邊
                op = opposite(last_segment_side)
                signals.append({
                    "name": "PAIR_SWITCH",
                    "side": op,
                    "confidence": 0.61,
                    "weight": 1.9,
                    "reason": f"近期呈現雙跳 / 成對節奏，當前 {side_text(last_segment_side)} 已成對，下一手偏向換邊「{side_text(op)}」。",
                })

    # 5. 重複規律 Replay
    replay = find_pattern_replay(non_tie)

    if replay:
        signals.append(replay)

    # 6. 近局偏向
    for window, weight in [(20, 1.1), (12, 1.25), (8, 1.35)]:
        recent = non_tie[-window:]

        if len(recent) < max(6, window // 2):
            continue

        b_count = recent.count("B")
        p_count = recent.count("P")
        total = len(recent)

        diff = abs(b_count - p_count) / total

        if diff >= 0.18:
            side = "B" if b_count > p_count else "P"
            signals.append({
                "name": f"RECENT_BIAS_{window}",
                "side": side,
                "confidence": min(0.66, 0.50 + diff),
                "weight": weight,
                "reason": f"近 {len(recent)} 手 {side_text(side)} 明顯較多，形成近期偏向。",
            })

    # 7. 散盤 / 混亂盤
    rep8 = repeat_rate(non_tie, 8)

    # 不是明顯跳，也不是明顯連，容易混亂
    if 0.35 <= sr8 <= 0.65 and 0.35 <= rep8 <= 0.65 and len(non_tie) >= 8:
        signals.append({
            "name": "CHAOS_OBSERVE",
            "side": None,
            "confidence": 0.0,
            "weight": 0.0,
            "reason": "近局跳與連都不明顯，屬於散盤 / 混亂盤，建議降低進場頻率。",
        })

    return signals


def estimate_tie_percent(road: List[str]) -> int:
    if not road:
        return 8

    recent = road[-30:]
    tie_count = recent.count("T")

    raw = round((tie_count / len(recent)) * 100)

    # 和局不建議放太大，避免影響莊閒主判斷
    return int(clamp(raw, 5, 22))


def combine_signals(road: List[str], signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    non_tie = get_non_tie(road)

    score_b = 0.0
    score_p = 0.0
    reason_list: List[str] = []
    pattern_names: List[str] = []
    observe_reasons: List[str] = []

    for signal in signals:
        name = signal.get("name", "")
        side = signal.get("side")
        confidence = float(signal.get("confidence", 0.0))
        weight = float(signal.get("weight", 0.0))
        reason = signal.get("reason", "")

        if reason:
            if side in ("B", "P"):
                reason_list.append(reason)
            else:
                observe_reasons.append(reason)

        if name:
            pattern_names.append(name)

        if side == "B":
            score_b += confidence * weight
        elif side == "P":
            score_p += confidence * weight

    # 如果完全沒有有效信號，用近局基本統計補底
    if score_b == 0 and score_p == 0 and non_tie:
        recent = non_tie[-12:]
        b_count = recent.count("B")
        p_count = recent.count("P")

        if b_count > p_count:
            score_b += 0.55
            reason_list.append("目前無明顯牌路型態，改用近 12 手莊閒比例作為弱判斷。")
        elif p_count > b_count:
            score_p += 0.55
            reason_list.append("目前無明顯牌路型態，改用近 12 手莊閒比例作為弱判斷。")

    tie_percent = estimate_tie_percent(road)
    bp_total = 100 - tie_percent

    # 將分數轉成莊閒百分比
    margin = score_b - score_p

    # tanh 壓縮，避免百分比太誇張
    share_b = 0.5 + math.tanh(margin / 3.2) * 0.23
    share_b = clamp(share_b, 0.28, 0.72)

    banker_percent = int(round(bp_total * share_b))
    player_percent = int(100 - tie_percent - banker_percent)

    gap = abs(banker_percent - player_percent)

    if len(non_tie) < 5:
        recommend = "觀望"
        signal_level = "資料不足"
        entry_allowed = False
    elif gap < 4:
        recommend = "觀望"
        signal_level = "低信號"
        entry_allowed = False
    else:
        recommend = "莊" if banker_percent > player_percent else "閒"

        if gap >= 12:
            signal_level = "強信號"
        elif gap >= 7:
            signal_level = "中信號"
        else:
            signal_level = "低信號"

        entry_allowed = gap >= 6

    # 如果混亂盤訊號存在，而且差距不夠大，強制觀望
    has_chaos = any(s.get("name") == "CHAOS_OBSERVE" for s in signals)

    if has_chaos and gap < 8:
        recommend = "觀望"
        signal_level = "散盤觀望"
        entry_allowed = False

    # 如果長龍疲勞，而且差距不大，避免硬追
    has_dragon_fatigue = any(s.get("name") == "DRAGON_FATIGUE" for s in signals)

    if has_dragon_fatigue and gap < 10:
        recommend = "觀望"
        signal_level = "長龍疲勞觀望"
        entry_allowed = False

    main_reason = "；".join(reason_list[:3]) if reason_list else ""
    observe_reason = "；".join(observe_reasons[:2]) if observe_reasons else ""

    if not main_reason and observe_reason:
        main_reason = observe_reason

    if not main_reason:
        main_reason = "目前牌路規律不明顯，建議先觀察下一局。"

    pattern_type = " / ".join(pattern_names[:4]) if pattern_names else "NO_PATTERN"

    return {
        "banker_percent": banker_percent,
        "player_percent": player_percent,
        "tie_percent": tie_percent,

        # 兼容你舊版前端欄位
        "banker_rate": banker_percent,
        "player_rate": player_percent,
        "tie_rate": tie_percent,

        "recommend": recommend,
        "recommend_text": recommend,
        "side": recommend,
        "signal_level": signal_level,
        "entry_allowed": entry_allowed,

        "pattern_type": pattern_type,
        "pattern_names": pattern_names,
        "reason": main_reason,
        "observe_reason": observe_reason,

        "score_banker": round(score_b, 3),
        "score_player": round(score_p, 3),
        "gap": gap,
    }


def build_road_summary(road: List[str]) -> Dict[str, Any]:
    non_tie = get_non_tie(road)
    segments = get_segments(non_tie)

    if not non_tie:
        return {
            "round_count": len(road),
            "valid_round_count": 0,
            "last_side": "",
            "last_side_text": "",
            "current_streak": 0,
            "switch_rate_8": 0,
            "repeat_rate_8": 0,
            "recent_road": "".join(road[-30:]),
        }

    last_side = non_tie[-1]
    current_streak = segments[-1][1] if segments else 1

    return {
        "round_count": len(road),
        "valid_round_count": len(non_tie),
        "last_side": last_side,
        "last_side_text": side_text(last_side),
        "current_streak": current_streak,
        "switch_rate_8": round(switch_rate(non_tie, 8), 3),
        "repeat_rate_8": round(repeat_rate(non_tie, 8), 3),
        "recent_road": "".join(road[-30:]),
        "recent_non_tie": "".join(non_tie[-30:]),
        "segments": [
            {
                "side": s,
                "side_text": side_text(s),
                "length": l,
            }
            for s, l in segments[-10:]
        ],
    }


def predict(road: Any) -> Dict[str, Any]:
    """
    主入口。
    app.py / monitor.py 只要呼叫：
    prediction = predict(road)

    road 可以是：
    ["B", "P", "B", "T"]
    或
    "BPBT"
    """

    normalized = normalize_road(road)

    if len(normalized) < 5:
        return {
            "banker_percent": 33,
            "player_percent": 33,
            "tie_percent": 34,

            "banker_rate": 33,
            "player_rate": 33,
            "tie_rate": 34,

            "recommend": "觀望",
            "recommend_text": "觀望",
            "side": "觀望",
            "signal_level": "資料不足",
            "entry_allowed": False,
            "pattern_type": "NOT_ENOUGH_DATA",
            "pattern_names": ["NOT_ENOUGH_DATA"],
            "reason": "目前牌路少於 5 局，無法判斷規律，建議先觀察。",
            "observe_reason": "資料不足。",
            "score_banker": 0,
            "score_player": 0,
            "gap": 0,
            "road_summary": build_road_summary(normalized),
        }

    signals = detect_road_patterns(normalized)
    result = combine_signals(normalized, signals)
    result["road_summary"] = build_road_summary(normalized)

    # 給前端顯示更白話的牌路判斷
    result["pattern_detail"] = build_pattern_detail(result)

    return result


def build_pattern_detail(result: Dict[str, Any]) -> str:
    pattern_names = result.get("pattern_names", [])
    signal_level = result.get("signal_level", "")
    recommend = result.get("recommend", "觀望")
    gap = result.get("gap", 0)

    readable = []

    mapping = {
        "LONG_DRAGON_FOLLOW": "長龍延伸",
        "DRAGON_FATIGUE": "長龍疲勞 / 斷龍風險",
        "SHORT_STREAK": "小龍延伸",
        "CHOP_PATTERN": "單跳 / 一跳一",
        "LIGHT_CHOP": "輕微跳格",
        "PAIR_COMPLETION": "成對補齊",
        "PAIR_SWITCH": "雙跳換邊",
        "PATTERN_REPLAY": "歷史重複規律",
        "RECENT_BIAS_20": "近 20 手偏向",
        "RECENT_BIAS_12": "近 12 手偏向",
        "RECENT_BIAS_8": "近 8 手偏向",
        "CHAOS_OBSERVE": "散盤觀望",
        "NOT_ENOUGH_DATA": "資料不足",
    }

    for name in pattern_names:
        if name in mapping and mapping[name] not in readable:
            readable.append(mapping[name])

    if not readable:
        readable.append("無明顯規律")

    return f"{'、'.join(readable)}｜{signal_level}｜推薦：{recommend}｜莊閒差距：{gap}%"


# 本地簡單測試用，Render 不會執行到這段
if __name__ == "__main__":
    samples = [
        "BBBBBP",
        "BPBPBPBP",
        "BBPPBBP",
        "BBPPBBPP",
        "BPBBPPBPBBPP",
        "BBBPBBPBBB",
        "BPTBPPTBBP",
    ]

    for s in samples:
        print("=" * 60)
        print("ROAD:", s)
        print(predict(s))
