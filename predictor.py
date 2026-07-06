# predictor.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import math

from config import DEEPSEEK_ENABLED, DEEPSEEK_API_KEY, DEEPSEEK_WEIGHT, LOCAL_MODEL_WEIGHT
from deepseek_client import DeepSeekClient


def normalize_road(road: Any) -> List[str]:
    if road is None:
        return []
    if isinstance(road, str):
        raw_items = list(road.strip().upper())
    else:
        raw_items = road

    cleaned: List[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            value = item.get("result") or item.get("side") or item.get("value") or item.get("text") or ""
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
    return {"B": "莊", "P": "閒", "T": "和", "OBSERVE": "觀望", "觀望": "觀望", "莊": "莊", "閒": "閒"}.get(side, "觀望")


def opposite(side: str) -> str:
    return "P" if side == "B" else "B" if side == "P" else ""


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def get_non_tie(road: List[str]) -> List[str]:
    return [x for x in road if x in ("B", "P")]


def get_segments(non_tie: List[str]) -> List[Tuple[str, int]]:
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
    return sum(1 for a, b in zip(recent, recent[1:]) if a != b) / max(1, len(recent) - 1)


def repeat_rate(non_tie: List[str], window: int = 8) -> float:
    recent = non_tie[-window:]
    if len(recent) < 2:
        return 0.0
    return sum(1 for a, b in zip(recent, recent[1:]) if a == b) / max(1, len(recent) - 1)


def find_pattern_replay(non_tie: List[str]) -> Optional[Dict[str, Any]]:
    n = len(non_tie)
    if n < 14:
        return None
    for size in range(10, 3, -1):
        if n <= size + 2:
            continue
        suffix = non_tie[-size:]
        counter = Counter()
        for i in range(0, n - size - 1):
            if non_tie[i:i + size] == suffix:
                nxt = non_tie[i + size]
                if nxt in ("B", "P"):
                    counter[nxt] += 1
        total = sum(counter.values())
        if total < 2:
            continue
        side, count = counter.most_common(1)[0]
        hit_rate = count / total
        if hit_rate >= 0.58:
            return {
                "name": "PATTERN_REPLAY",
                "side": side,
                "confidence": round(hit_rate, 3),
                "weight": min(2.8, 1.4 + hit_rate),
                "reason": f"找到尾段 {size} 手重複規律，歷史相似段後續偏向「{side_text(side)}」{round(hit_rate * 100)}%。",
                "samples": total,
            }
    return None


def detect_road_patterns(road: List[str]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    non_tie = get_non_tie(road)
    if len(non_tie) < 5:
        return [{"name": "NOT_ENOUGH_DATA", "side": None, "confidence": 0, "weight": 0, "reason": "有效莊閒資料少於 5 手，先觀望。"}]

    segments = get_segments(non_tie)
    last_side = non_tie[-1]
    last_segment_side, last_segment_len = segments[-1]

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
            op = opposite(last_segment_side)
            conf = min(0.62, 0.48 + last_segment_len * 0.015)
            signals.append({
                "name": "DRAGON_FATIGUE",
                "side": op,
                "confidence": conf,
                "weight": 1.2,
                "reason": f"{side_text(last_segment_side)} 已連續 {last_segment_len} 手，進入長龍疲勞區，需注意斷龍風險。",
            })
    elif last_segment_len in (2, 3):
        conf = 0.53 + last_segment_len * 0.02
        signals.append({
            "name": "SHORT_STREAK",
            "side": last_segment_side,
            "confidence": conf,
            "weight": 1.1,
            "reason": f"目前為 {side_text(last_segment_side)} {last_segment_len} 連，小龍偏延續，但信號不算強。",
        })

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

    if len(segments) >= 4:
        recent_segments = segments[-5:]
        recent_lengths = [length for _, length in recent_segments]
        pair_like_count = sum(1 for x in recent_lengths[:-1] if x in (2, 3))
        if pair_like_count >= 3:
            if last_segment_len == 1:
                signals.append({
                    "name": "PAIR_COMPLETION",
                    "side": last_segment_side,
                    "confidence": 0.63,
                    "weight": 2.0,
                    "reason": f"近期多次出現成對節奏，當前 {side_text(last_segment_side)} 只有 1 手，偏向補成一對。",
                })
            elif last_segment_len >= 2:
                op = opposite(last_segment_side)
                signals.append({
                    "name": "PAIR_SWITCH",
                    "side": op,
                    "confidence": 0.61,
                    "weight": 1.9,
                    "reason": f"近期呈現雙跳 / 成對節奏，當前 {side_text(last_segment_side)} 已成對，下一手偏向換邊「{side_text(op)}」。",
                })

    replay = find_pattern_replay(non_tie)
    if replay:
        signals.append(replay)

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

    rep8 = repeat_rate(non_tie, 8)
    if 0.35 <= sr8 <= 0.65 and 0.35 <= rep8 <= 0.65 and len(non_tie) >= 8:
        signals.append({
            "name": "CHAOS_OBSERVE",
            "side": None,
            "confidence": 0,
            "weight": 0,
            "reason": "近局跳與連都不明顯，屬於散盤 / 混亂盤，建議降低進場頻率。",
        })
    return signals


def estimate_tie_percent(road: List[str]) -> int:
    if not road:
        return 8
    recent = road[-30:]
    raw = round((recent.count("T") / len(recent)) * 100)
    return int(clamp(raw, 5, 22))


def build_road_summary(road: List[str]) -> Dict[str, Any]:
    non_tie = get_non_tie(road)
    segments = get_segments(non_tie)
    if not non_tie:
        return {"round_count": len(road), "valid_round_count": 0, "recent_road": "".join(road[-30:])}
    last_side = non_tie[-1]
    return {
        "round_count": len(road),
        "valid_round_count": len(non_tie),
        "last_side": last_side,
        "last_side_text": side_text(last_side),
        "current_streak": segments[-1][1] if segments else 1,
        "switch_rate_8": round(switch_rate(non_tie, 8), 3),
        "repeat_rate_8": round(repeat_rate(non_tie, 8), 3),
        "recent_road": "".join(road[-30:]),
        "recent_non_tie": "".join(non_tie[-30:]),
        "segments": [{"side": s, "side_text": side_text(s), "length": l} for s, l in segments[-10:]],
    }


def build_pattern_detail(result: Dict[str, Any]) -> str:
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
    names = result.get("pattern_names", [])
    readable = []
    for name in names:
        if name in mapping and mapping[name] not in readable:
            readable.append(mapping[name])
    if not readable:
        readable.append("無明顯規律")
    return f"{'、'.join(readable)}｜{result.get('signal_level', '')}｜推薦：{result.get('recommend', '觀望')}｜莊閒差距：{result.get('gap', 0)}%"


def local_predict(road: List[str]) -> Dict[str, Any]:
    if len(road) < 5:
        base = {
            "banker_percent": 33,
            "player_percent": 33,
            "tie_percent": 34,
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
            "road_summary": build_road_summary(road),
            "ai_used": False,
        }
        base["banker_rate"] = base["banker_percent"]
        base["player_rate"] = base["player_percent"]
        base["tie_rate"] = base["tie_percent"]
        base["pattern_detail"] = build_pattern_detail(base)
        return base

    signals = detect_road_patterns(road)
    score_b = 0.0
    score_p = 0.0
    reasons: List[str] = []
    observes: List[str] = []
    pattern_names: List[str] = []

    for s in signals:
        name = s.get("name", "")
        side = s.get("side")
        conf = float(s.get("confidence", 0))
        weight = float(s.get("weight", 0))
        reason = s.get("reason", "")
        if name:
            pattern_names.append(name)
        if side == "B":
            score_b += conf * weight
            if reason:
                reasons.append(reason)
        elif side == "P":
            score_p += conf * weight
            if reason:
                reasons.append(reason)
        elif reason:
            observes.append(reason)

    non_tie = get_non_tie(road)
    if score_b == 0 and score_p == 0 and non_tie:
        recent = non_tie[-12:]
        b = recent.count("B")
        p = recent.count("P")
        if b > p:
            score_b += 0.55
        elif p > b:
            score_p += 0.55
        reasons.append("目前無明顯牌路型態，改用近 12 手莊閒比例作為弱判斷。")

    tie_percent = estimate_tie_percent(road)
    bp_total = 100 - tie_percent
    margin = score_b - score_p
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
        signal_level = "強信號" if gap >= 12 else "中信號" if gap >= 7 else "低信號"
        entry_allowed = gap >= 6

    has_chaos = any(s.get("name") == "CHAOS_OBSERVE" for s in signals)
    if has_chaos and gap < 8:
        recommend = "觀望"
        signal_level = "散盤觀望"
        entry_allowed = False

    has_fatigue = any(s.get("name") == "DRAGON_FATIGUE" for s in signals)
    if has_fatigue and gap < 10:
        recommend = "觀望"
        signal_level = "長龍疲勞觀望"
        entry_allowed = False

    main_reason = "；".join(reasons[:3]) or "；".join(observes[:2]) or "目前牌路規律不明顯，建議先觀察下一局。"
    result = {
        "banker_percent": banker_percent,
        "player_percent": player_percent,
        "tie_percent": tie_percent,
        "banker_rate": banker_percent,
        "player_rate": player_percent,
        "tie_rate": tie_percent,
        "recommend": recommend,
        "recommend_text": recommend,
        "side": recommend,
        "signal_level": signal_level,
        "entry_allowed": entry_allowed,
        "pattern_type": " / ".join(pattern_names[:4]) if pattern_names else "NO_PATTERN",
        "pattern_names": pattern_names,
        "reason": main_reason,
        "observe_reason": "；".join(observes[:2]),
        "score_banker": round(score_b, 3),
        "score_player": round(score_p, 3),
        "gap": gap,
        "road_summary": build_road_summary(road),
        "ai_used": False,
    }
    result["pattern_detail"] = build_pattern_detail(result)
    return result


def combine_with_ai(local: Dict[str, Any], ai: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(local)
    result["deepseek"] = ai
    if not ai.get("ok"):
        result["ai_used"] = False
        result["ai_reason"] = ai.get("reason", "AI 未啟用或呼叫失敗。")
        return result

    ai_side = ai.get("side")
    ai_conf = int(ai.get("confidence", 0))
    local_side = local.get("recommend")

    def pct_to_side_score(side: str) -> float:
        if side == "B":
            return float(local.get("banker_percent", 33))
        if side == "P":
            return float(local.get("player_percent", 33))
        return 50.0

    local_b_score = pct_to_side_score("B")
    local_p_score = pct_to_side_score("P")
    ai_b_score = 50.0
    ai_p_score = 50.0
    if ai_side == "B":
        ai_b_score = 50.0 + ai_conf / 2
        ai_p_score = 50.0 - ai_conf / 2
    elif ai_side == "P":
        ai_p_score = 50.0 + ai_conf / 2
        ai_b_score = 50.0 - ai_conf / 2

    lw = max(0.0, LOCAL_MODEL_WEIGHT)
    aw = max(0.0, DEEPSEEK_WEIGHT)
    total_w = lw + aw if (lw + aw) > 0 else 1.0
    final_b_raw = (local_b_score * lw + ai_b_score * aw) / total_w
    final_p_raw = (local_p_score * lw + ai_p_score * aw) / total_w

    tie_percent = int(local.get("tie_percent", 8))
    bp_total = 100 - tie_percent
    denom = final_b_raw + final_p_raw
    banker_percent = int(round(bp_total * final_b_raw / denom)) if denom > 0 else int(local.get("banker_percent", 33))
    player_percent = int(100 - tie_percent - banker_percent)
    gap = abs(banker_percent - player_percent)

    if ai_side == "OBSERVE" and ai_conf >= 60 and gap < 10:
        recommend = "觀望"
        signal_level = "AI觀望校準"
        entry_allowed = False
    elif gap < 4:
        recommend = "觀望"
        signal_level = "低信號"
        entry_allowed = False
    else:
        recommend = "莊" if banker_percent > player_percent else "閒"
        same_side = (local_side == "莊" and ai_side == "B") or (local_side == "閒" and ai_side == "P")
        disagree = ai_side in ("B", "P") and ((recommend == "莊" and ai_side != "B") or (recommend == "閒" and ai_side != "P"))
        if same_side:
            signal_level = "AI共振強信號" if gap >= 7 else "AI共振中信號"
            entry_allowed = gap >= 6
        elif disagree:
            signal_level = "AI分歧觀察"
            if gap < 10:
                recommend = "觀望"
                entry_allowed = False
            else:
                entry_allowed = True
        else:
            signal_level = local.get("signal_level", "中信號")
            entry_allowed = gap >= 6

    result.update({
        "banker_percent": banker_percent,
        "player_percent": player_percent,
        "tie_percent": tie_percent,
        "banker_rate": banker_percent,
        "player_rate": player_percent,
        "tie_rate": tie_percent,
        "recommend": recommend,
        "recommend_text": recommend,
        "side": recommend,
        "signal_level": signal_level,
        "entry_allowed": entry_allowed,
        "gap": gap,
        "ai_used": True,
        "ai_reason": ai.get("reason", ""),
        "ai_pattern": ai.get("pattern", ""),
        "reason": f"本地牌路：{local.get('reason', '')}｜AI校準：{ai.get('reason', '')}",
    })
    result["pattern_detail"] = f"{local.get('pattern_detail', '')}｜AI：{ai.get('pattern', '')}｜{signal_level}"
    return result


def predict(road: Any) -> Dict[str, Any]:
    normalized = normalize_road(road)
    local = local_predict(normalized)
    if DEEPSEEK_ENABLED and DEEPSEEK_API_KEY and len(normalized) >= 5:
        try:
            ai = DeepSeekClient().analyze_road(normalized, local)
            return combine_with_ai(local, ai)
        except Exception as e:
            local["deepseek"] = {"ok": False, "side": "OBSERVE", "confidence": 0, "reason": f"DeepSeek 例外：{e}", "pattern": "EXCEPTION"}
            local["ai_used"] = False
            local["ai_reason"] = f"DeepSeek 例外：{e}"
            return local
    local["deepseek"] = {
        "ok": False,
        "side": "OBSERVE",
        "confidence": 0,
        "reason": "DeepSeek 未啟用。",
        "pattern": "DISABLED",
    }
    return local


if __name__ == "__main__":
    samples = ["BBBBBP", "BPBPBPBP", "BBPPBBP", "BBPPBBPP", "BPBBPPBPBBPP", "BPTBPPTBBP"]
    for s in samples:
        print("=" * 60)
        print(s)
        print(predict(s))
