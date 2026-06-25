import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from deepseek_client import DeepSeekClient

B_PRIOR = float(os.getenv("B_PRIOR", "0.4586"))
P_PRIOR = float(os.getenv("P_PRIOR", "0.4462"))
T_PRIOR = float(os.getenv("T_PRIOR", "0.0952"))

MARKOV_WEIGHT = float(os.getenv("MARKOV_WEIGHT", "0.26"))
ROAD_WEIGHT = float(os.getenv("ROAD_WEIGHT", "0.24"))
STREAK_WEIGHT = float(os.getenv("STREAK_WEIGHT", "0.18"))
BALANCE_WEIGHT = float(os.getenv("BALANCE_WEIGHT", "0.12"))
RECENT_WEIGHT = float(os.getenv("RECENT_WEIGHT", "0.10"))
TIE_WEIGHT = float(os.getenv("TIE_WEIGHT", "0.04"))
AI_BLEND = float(os.getenv("AI_BLEND", "0.16"))

TIE_SHRINK = float(os.getenv("TIE_SHRINK", "0.35"))
TIE_MAX_PROB = float(os.getenv("TIE_MAX_PROB", "0.18"))
ALLOW_TIE_RECOMMEND = os.getenv("ALLOW_TIE_RECOMMEND", "0") == "1"
TIE_RECOMMEND_MIN = float(os.getenv("TIE_RECOMMEND_MIN", "0.165"))
MIN_HISTORY_FOR_AI = int(os.getenv("MIN_HISTORY_FOR_AI", "6"))
MIN_HISTORY_FOR_SIGNAL = int(os.getenv("MIN_HISTORY_FOR_SIGNAL", "4"))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def _normalize_three(b: float, p: float, t: float) -> Tuple[float, float, float]:
    b = max(0.001, b)
    p = max(0.001, p)
    t = max(0.001, min(TIE_MAX_PROB, t))
    s = b + p + t
    return b / s, p / s, t / s


def _last_non_tie(history: List[str]) -> List[str]:
    return [x for x in history if x in {"B", "P"}]


def _streak(non_tie: List[str]) -> Tuple[str, int]:
    if not non_tie:
        return "", 0
    last = non_tie[-1]
    n = 1
    for x in reversed(non_tie[:-1]):
        if x == last:
            n += 1
        else:
            break
    return last, n


def _transition_prob(non_tie: List[str]) -> Dict[str, float]:
    # current shoe Markov transition; shrink heavily to 50/50 when samples are few.
    counts = defaultdict(lambda: Counter())
    for a, b in zip(non_tie, non_tie[1:]):
        counts[a][b] += 1
    if not non_tie:
        return {"B": 0.5, "P": 0.5, "sample": 0}
    last = non_tie[-1]
    c = counts[last]
    sample = c["B"] + c["P"]
    alpha = float(os.getenv("MARKOV_ALPHA", "2.4"))
    b = (c["B"] + alpha) / (sample + 2 * alpha)
    p = (c["P"] + alpha) / (sample + 2 * alpha)
    # more shrink under 8 transitions
    shrink = min(1.0, sample / float(os.getenv("MARKOV_FULL_SAMPLE", "14")))
    b = 0.5 * (1 - shrink) + b * shrink
    p = 0.5 * (1 - shrink) + p * shrink
    return {"B": b, "P": p, "sample": sample}


def _road_pattern_score(non_tie: List[str]) -> Dict[str, Any]:
    # Recognizes: long dragon, chop road, double road, last-4 repetition.
    if len(non_tie) < 3:
        return {"B": 0.5, "P": 0.5, "label": "資料不足", "strength": 0.0}

    last, streak_n = _streak(non_tie)
    opp = "P" if last == "B" else "B"
    recent = non_tie[-12:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)

    b = p = 0.5
    label = "混合盤"
    strength = 0.08

    if streak_n >= 5:
        # Long dragon usually increases continuation signal but with anti-overfit shrink.
        cont = 0.53 + min(0.05, (streak_n - 5) * 0.008)
        b = cont if last == "B" else 1 - cont
        p = cont if last == "P" else 1 - cont
        label = f"長龍{last}{streak_n}"
        strength = 0.18
    elif switch_rate >= 0.72 and len(recent) >= 6:
        # Chop road: alternating pressure.
        b = 0.57 if opp == "B" else 0.43
        p = 0.57 if opp == "P" else 0.43
        label = "跳路偏強"
        strength = 0.16
    elif len(non_tie) >= 6 and non_tie[-6:] in [list("BBPPBB"), list("PPBBPP")]:
        # double chop continuation.
        next_side = non_tie[-2]
        b = 0.56 if next_side == "B" else 0.44
        p = 0.56 if next_side == "P" else 0.44
        label = "雙跳/兩房型"
        strength = 0.15
    elif len(non_tie) >= 8:
        # last 4 pattern matching in current shoe
        key = "".join(non_tie[-4:])
        follows = []
        seq = "".join(non_tie)
        for i in range(0, len(seq) - 4):
            if seq[i:i + 4] == key and i + 4 < len(seq):
                follows.append(seq[i + 4])
        if follows:
            c = Counter(follows)
            total = c["B"] + c["P"]
            b_raw = c["B"] / total
            shrink = min(0.75, total / 10)
            b = 0.5 * (1 - shrink) + b_raw * shrink
            p = 1 - b
            label = f"四碼回測{key}"
            strength = min(0.20, 0.08 + total * 0.015)
        else:
            # mild bias against over-clustered side
            b_count = recent.count("B")
            p_count = recent.count("P")
            if abs(b_count - p_count) >= 4:
                scarce = "B" if b_count < p_count else "P"
                b = 0.54 if scarce == "B" else 0.46
                p = 0.54 if scarce == "P" else 0.46
                label = "短窗均衡修正"
                strength = 0.10

    return {"B": b, "P": p, "label": label, "strength": strength, "switch_rate": switch_rate, "streak": streak_n}


def _recent_score(non_tie: List[str]) -> Dict[str, float]:
    if not non_tie:
        return {"B": 0.5, "P": 0.5}
    recent = non_tie[-10:]
    # Exponential emphasis on latest outcomes; continuation when low switch, reversal when high switch.
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)
    last, n = _streak(non_tie)
    opp = "P" if last == "B" else "B"
    if switch_rate > 0.66:
        side = opp
        edge = 0.055
    elif n >= 3:
        side = last
        edge = 0.045 + min(0.025, (n - 3) * 0.008)
    else:
        # small balance correction
        b_count = recent.count("B")
        p_count = recent.count("P")
        side = "B" if b_count < p_count else "P"
        edge = min(0.035, abs(b_count - p_count) * 0.006)
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}


def _balance_score(non_tie: List[str]) -> Dict[str, float]:
    if len(non_tie) < 8:
        return {"B": 0.5, "P": 0.5}
    b = non_tie.count("B")
    p = non_tie.count("P")
    diff = b - p
    edge = min(0.055, abs(diff) / max(1, len(non_tie)) * 0.16)
    # scarce side gets a mild correction only, never strong.
    side = "B" if diff < 0 else "P"
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}


def _streak_score(non_tie: List[str]) -> Dict[str, float]:
    last, n = _streak(non_tie)
    if not last:
        return {"B": 0.5, "P": 0.5}
    opp = "P" if last == "B" else "B"
    if n == 1:
        side, edge = opp, 0.025
    elif n == 2:
        side, edge = last, 0.030
    elif n == 3:
        side, edge = last, 0.045
    elif n >= 4:
        side, edge = last, min(0.075, 0.050 + (n - 4) * 0.008)
    else:
        side, edge = last, 0.0
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}


def _tie_score(history: List[str]) -> float:
    if not history:
        return T_PRIOR
    recent = history[-18:]
    t_rate = recent.count("T") / len(recent)
    gap_since_tie = 0
    for x in reversed(history):
        if x == "T":
            break
        gap_since_tie += 1
    # tie is always shrunk; it is only a side warning unless enabled.
    pressure = T_PRIOR * (1 - TIE_SHRINK) + t_rate * TIE_SHRINK
    if gap_since_tie >= 18:
        pressure += 0.012
    if recent[-4:].count("T") >= 2:
        pressure += 0.018
    return _clamp(pressure, 0.055, TIE_MAX_PROB)


def _confidence(b: float, p: float, t: float, history_len: int, agreement: float) -> Tuple[float, str]:
    gap = abs(b - p)
    base = gap * 3.6 + agreement * 0.22 + min(0.16, history_len / 80)
    conf = _clamp(base, 0.08, 0.94)
    if history_len < MIN_HISTORY_FOR_SIGNAL:
        return min(conf, 0.35), "冷啟動"
    if conf >= 0.68:
        return conf, "強訊號"
    if conf >= 0.48:
        return conf, "中訊號"
    return conf, "弱訊號"


def predict(history: List[str], venue: str = "", room: str = "", shoe_id: str = "") -> Dict[str, Any]:
    history = [x.upper() for x in history if x.upper() in {"B", "P", "T"}]
    non_tie = _last_non_tie(history)

    markov = _transition_prob(non_tie)
    road = _road_pattern_score(non_tie)
    recent = _recent_score(non_tie)
    balance = _balance_score(non_tie)
    streak = _streak_score(non_tie)

    # Weighted side model, then add tie separately.
    total_w = MARKOV_WEIGHT + ROAD_WEIGHT + STREAK_WEIGHT + BALANCE_WEIGHT + RECENT_WEIGHT
    b_side = (
        markov["B"] * MARKOV_WEIGHT
        + road["B"] * ROAD_WEIGHT
        + streak["B"] * STREAK_WEIGHT
        + balance["B"] * BALANCE_WEIGHT
        + recent["B"] * RECENT_WEIGHT
    ) / total_w
    p_side = 1 - b_side

    tie_prob = _tie_score(history)
    b_prob = b_side * (1 - tie_prob)
    p_prob = p_side * (1 - tie_prob)

    feature_payload = {
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "history_len": len(history),
        "history_tail": "".join(history[-36:]),
        "non_tie_tail": "".join(non_tie[-36:]),
        "markov": markov,
        "road": road,
        "recent": recent,
        "balance": balance,
        "streak": streak,
        "local_probs": {"B": round(b_prob, 5), "P": round(p_prob, 5), "T": round(tie_prob, 5)},
    }

    ai_result = None
    if len(history) >= MIN_HISTORY_FOR_AI and AI_BLEND > 0:
        ai_result = DeepSeekClient().calibrate(feature_payload)
        if ai_result and not ai_result.get("error"):
            try:
                ba = _clamp(float(ai_result.get("banker_adjust", 0)), -0.035, 0.035)
                pa = _clamp(float(ai_result.get("player_adjust", 0)), -0.035, 0.035)
                ta = _clamp(float(ai_result.get("tie_adjust", 0)), -0.020, 0.020)
                ai_conf = _clamp(float(ai_result.get("confidence", 0.4)), 0, 1)
                blend = AI_BLEND * (0.45 + ai_conf * 0.55)
                b_prob += ba * blend
                p_prob += pa * blend
                tie_prob += ta * blend
            except Exception:
                pass

    b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)

    votes = [
        "B" if markov["B"] >= markov["P"] else "P",
        "B" if road["B"] >= road["P"] else "P",
        "B" if streak["B"] >= streak["P"] else "P",
        "B" if balance["B"] >= balance["P"] else "P",
        "B" if recent["B"] >= recent["P"] else "P",
    ]
    main_pick = "B" if b_prob >= p_prob else "P"
    agreement = votes.count(main_pick) / len(votes)

    if ALLOW_TIE_RECOMMEND and tie_prob >= TIE_RECOMMEND_MIN and tie_prob > max(b_prob, p_prob) * 0.55:
        recommend = "T"
    else:
        recommend = main_pick

    conf, level = _confidence(b_prob, p_prob, tie_prob, len(history), agreement)
    reason_parts = [road.get("label", "牌路"), f"模型一致{int(agreement * 100)}%"]
    if ai_result and ai_result.get("pattern_label"):
        reason_parts.append(f"AI:{ai_result.get('pattern_label')}")
    elif ai_result and ai_result.get("error"):
        reason_parts.append("AI離線改本地判斷")

    return {
        "ok": True,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "round_no": len(history) + 1,
        "history_len": len(history),
        "banker_rate": round(b_prob * 100, 1),
        "player_rate": round(p_prob * 100, 1),
        "tie_rate": round(tie_prob * 100, 1),
        "recommend": recommend,
        "recommend_text": {"B": "莊", "P": "閒", "T": "和"}[recommend],
        "confidence": round(conf, 3),
        "signal_level": level,
        "pattern_label": road.get("label", ""),
        "reason": " / ".join(reason_parts),
        "ai_used": bool(ai_result and not ai_result.get("error")),
        "ai_result": ai_result if os.getenv("DEBUG_AI_RESULT", "0") == "1" else None,
        "debug": feature_payload if os.getenv("DEBUG_PREDICTOR", "0") == "1" else None,
    }
