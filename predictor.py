import os
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional

from deepseek_client import DeepSeekClient

# -----------------------------
# Base priors / environment
# -----------------------------
B_PRIOR = float(os.getenv("B_PRIOR", "0.4586"))
P_PRIOR = float(os.getenv("P_PRIOR", "0.4462"))
T_PRIOR = float(os.getenv("T_PRIOR", "0.0952"))

# Main ensemble weights. These are still used, but v3 adds a road-state router on top.
MARKOV_WEIGHT = float(os.getenv("MARKOV_WEIGHT", "0.20"))
ROAD_WEIGHT = float(os.getenv("ROAD_WEIGHT", "0.36"))
STREAK_WEIGHT = float(os.getenv("STREAK_WEIGHT", "0.14"))
BALANCE_WEIGHT = float(os.getenv("BALANCE_WEIGHT", "0.08"))
RECENT_WEIGHT = float(os.getenv("RECENT_WEIGHT", "0.10"))
AI_BLEND = float(os.getenv("AI_BLEND", "0.14"))

# Dynamic router controls.
ROAD_STATE_ROUTER = os.getenv("ROAD_STATE_ROUTER", "1") == "1"
DYNAMIC_WEIGHT_MODE = os.getenv("DYNAMIC_WEIGHT_MODE", "1") == "1"
FRONT_PATTERN_MATCH = os.getenv("FRONT_PATTERN_MATCH", "1") == "1"

# Tie handling. Tie is a warning layer by default, not a main recommendation.
TIE_SHRINK = float(os.getenv("TIE_SHRINK", "0.35"))
TIE_MAX_PROB = float(os.getenv("TIE_MAX_PROB", "0.18"))
ALLOW_TIE_RECOMMEND = os.getenv("ALLOW_TIE_RECOMMEND", "0") == "1"
TIE_RECOMMEND_MIN = float(os.getenv("TIE_RECOMMEND_MIN", "0.165"))

MIN_HISTORY_FOR_AI = int(os.getenv("MIN_HISTORY_FOR_AI", "6"))
MIN_HISTORY_FOR_SIGNAL = int(os.getenv("MIN_HISTORY_FOR_SIGNAL", "4"))

# Dragon / run-length memory.
DRAGON_MEMORY_LOOKBACK = int(os.getenv("DRAGON_MEMORY_LOOKBACK", "18"))
DRAGON_BREAK_REPEAT_MIN = int(os.getenv("DRAGON_BREAK_REPEAT_MIN", "2"))
DRAGON_FOLLOW_MIN_LEN = int(os.getenv("DRAGON_FOLLOW_MIN_LEN", "3"))
DRAGON_MIN_LEN = int(os.getenv("DRAGON_MIN_LEN", "3"))
DRAGON_STRONG_LEN = int(os.getenv("DRAGON_STRONG_LEN", "5"))
DRAGON_FATIGUE_START = int(os.getenv("DRAGON_FATIGUE_START", "8"))
DRAGON_MAX_EDGE = float(os.getenv("DRAGON_MAX_EDGE", "0.118"))
DRAGON_BREAK_EDGE = float(os.getenv("DRAGON_BREAK_EDGE", "0.100"))

# Single chop / double chop / room rhythm.
ROAD_PATTERN_WINDOW = int(os.getenv("ROAD_PATTERN_WINDOW", "18"))
SINGLE_CHOP_CONFIRM_WINDOW = int(os.getenv("SINGLE_CHOP_CONFIRM_WINDOW", "7"))
SINGLE_CHOP_MIN_RATE = float(os.getenv("SINGLE_CHOP_MIN_RATE", "0.66"))
DOUBLE_CHOP_LOOKBACK = int(os.getenv("DOUBLE_CHOP_LOOKBACK", "12"))
DOUBLE_CHOP_MIN_HITS = int(os.getenv("DOUBLE_CHOP_MIN_HITS", "3"))
ROOM_PATTERN_MODE = os.getenv("ROOM_PATTERN_MODE", "1") == "1"
ONE_TWO_PATTERN_WEIGHT = float(os.getenv("ONE_TWO_PATTERN_WEIGHT", "0.22"))
TWO_ONE_PATTERN_WEIGHT = float(os.getenv("TWO_ONE_PATTERN_WEIGHT", "0.22"))
RUN_PATTERN_LOOKBACK = int(os.getenv("RUN_PATTERN_LOOKBACK", "12"))
RUN_PATTERN_MIN_MATCH = int(os.getenv("RUN_PATTERN_MIN_MATCH", "3"))

# Front / n-gram memory.
PATTERN_LOOKBACK = int(os.getenv("PATTERN_LOOKBACK", "6"))
FRONT_PATTERN_LOOKBACK = int(os.getenv("FRONT_PATTERN_LOOKBACK", "7"))
FRONT_PATTERN_MIN_SAMPLE = int(os.getenv("FRONT_PATTERN_MIN_SAMPLE", "2"))

# Markov smoothing.
MARKOV_ALPHA = float(os.getenv("MARKOV_ALPHA", "2.8"))
MARKOV_FULL_SAMPLE = float(os.getenv("MARKOV_FULL_SAMPLE", "18"))

# Output limits.
MAX_SIDE_PROB = float(os.getenv("MAX_SIDE_PROB", "0.62"))
MIN_SIDE_PROB = float(os.getenv("MIN_SIDE_PROB", "0.38"))


# -----------------------------
# Utility
# -----------------------------
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


def _opposite(side: str) -> str:
    return "P" if side == "B" else "B"


def _side_name(side: str) -> str:
    return {"B": "莊", "P": "閒", "T": "和"}.get(side, side)


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


def _runs(non_tie: List[str]) -> List[Tuple[str, int]]:
    """Compress B/P sequence into runs, e.g. BBPBBB -> [(B,2),(P,1),(B,3)]."""
    if not non_tie:
        return []
    out: List[Tuple[str, int]] = []
    side = non_tie[0]
    n = 1
    for x in non_tie[1:]:
        if x == side:
            n += 1
        else:
            out.append((side, n))
            side, n = x, 1
    out.append((side, n))
    return out


def _bp_score(side: str, prob: float) -> Tuple[float, float]:
    prob = _clamp(prob, MIN_SIDE_PROB, MAX_SIDE_PROB)
    return (prob, 1 - prob) if side == "B" else (1 - prob, prob)


def _edge_of(score: Dict[str, Any]) -> float:
    return abs(float(score.get("B", 0.5)) - 0.5)


def _score(side: str, prob: float, label: str, strength: float, action: str = "", **extra: Any) -> Dict[str, Any]:
    b, p = _bp_score(side, prob)
    data = {
        "B": b,
        "P": p,
        "pick": side,
        "label": label,
        "action": action,
        "strength": _clamp(strength, 0.0, 0.35),
        "edge": abs(prob - 0.5),
    }
    data.update(extra)
    return data


def _neutral(label: str = "資料不足") -> Dict[str, Any]:
    return {"B": 0.5, "P": 0.5, "pick": "", "label": label, "action": "", "strength": 0.0, "edge": 0.0}


# -----------------------------
# Base statistical layers
# -----------------------------
def _transition_prob(non_tie: List[str]) -> Dict[str, float]:
    counts = defaultdict(lambda: Counter())
    for a, b in zip(non_tie, non_tie[1:]):
        counts[a][b] += 1
    if not non_tie:
        return {"B": 0.5, "P": 0.5, "sample": 0}
    last = non_tie[-1]
    c = counts[last]
    sample = c["B"] + c["P"]
    b = (c["B"] + MARKOV_ALPHA) / (sample + 2 * MARKOV_ALPHA)
    p = (c["P"] + MARKOV_ALPHA) / (sample + 2 * MARKOV_ALPHA)
    shrink = min(1.0, sample / MARKOV_FULL_SAMPLE)
    b = 0.5 * (1 - shrink) + b * shrink
    p = 0.5 * (1 - shrink) + p * shrink
    return {"B": b, "P": p, "sample": sample}


def _balance_score(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 8:
        return _neutral("均衡資料不足")
    b = non_tie.count("B")
    p = non_tie.count("P")
    diff = b - p
    # This is intentionally small. Balance cannot override clear road states.
    edge = min(0.040, abs(diff) / max(1, len(non_tie)) * 0.11)
    side = "B" if diff < 0 else "P"
    return _score(side, 0.5 + edge, "短靴均衡修正", 0.06 + min(0.05, edge * 1.5), "均衡")


def _recent_score(non_tie: List[str]) -> Dict[str, Any]:
    if not non_tie:
        return _neutral("近期資料不足")
    recent = non_tie[-10:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)
    last, n = _streak(non_tie)
    if switch_rate >= 0.70 and len(recent) >= 7:
        return _score(_opposite(last), 0.552, "近期單跳偏強", 0.12, "單跳")
    if n >= DRAGON_MIN_LEN:
        edge = 0.035 + min(0.030, (n - DRAGON_MIN_LEN) * 0.008)
        if n >= DRAGON_FATIGUE_START:
            edge *= 0.72
        return _score(last, 0.5 + edge, f"近期{_side_name(last)}{n}連", 0.10 + min(0.05, edge), "跟龍")
    b_count = recent.count("B")
    p_count = recent.count("P")
    if abs(b_count - p_count) >= 3:
        side = "B" if b_count < p_count else "P"
        return _score(side, 0.525, "近期偏態小修正", 0.07, "修正")
    return _neutral("近期混合")


def _streak_score(non_tie: List[str]) -> Dict[str, Any]:
    last, n = _streak(non_tie)
    if not last:
        return _neutral("連段資料不足")
    opp = _opposite(last)
    if n == 1:
        return _score(opp, 0.518, "單顆後反邊微修正", 0.05, "反邊")
    if n == 2:
        return _score(last, 0.526, "短龍2連補足", 0.07, "補足")
    if n == 3:
        return _score(last, 0.540, "中龍3連續龍", 0.09, "跟龍")
    if n == 4:
        return _score(last, 0.552, "中龍4連續龍", 0.11, "跟龍")
    if n < DRAGON_FATIGUE_START:
        return _score(last, min(0.585, 0.558 + (n - 4) * 0.008), f"長龍{_side_name(last)}{n}連", 0.13, "跟龍")
    return _score(last, 0.548, f"超長龍{_side_name(last)}{n}連疲勞", 0.10, "保守跟龍")


# -----------------------------
# Advanced road-state router
# -----------------------------
def _run_follow_stats(non_tie: List[str], current_len: int) -> Dict[str, Any]:
    completed = _runs(non_tie)[:-1]
    cont = 0
    brk = 0
    for _side, length in completed:
        if length > current_len:
            cont += 1
        elif length == current_len:
            brk += 1
    return {"cont": cont, "break": brk, "sample": cont + brk}


def _recent_completed_lengths(runs: List[Tuple[str, int]], n: int = DRAGON_MEMORY_LOOKBACK) -> List[int]:
    return [length for _side, length in runs[:-1]][-n:]


def _length_rhythm_score(non_tie: List[str]) -> Dict[str, Any]:
    """Detects 1-2 / 2-1 / 2-2 / 3-1 run-length rhythms and decides 補足 vs 轉邊."""
    runs = _runs(non_tie)
    if len(runs) < 5:
        return _neutral("長短節奏資料不足")

    current_side, current_len = runs[-1]
    completed = _recent_completed_lengths(runs, RUN_PATTERN_LOOKBACK)
    if len(completed) < 4:
        return _neutral("長短節奏資料不足")

    # Fixed cut length, e.g. 2-2-2-2 or 3-3-3-3.
    c = Counter(completed[-8:])
    mode_len, mode_count = c.most_common(1)[0]
    fixed_consistency = mode_count / max(1, min(8, len(completed)))
    if mode_count >= RUN_PATTERN_MIN_MATCH and fixed_consistency >= 0.45:
        if current_len < mode_len:
            side = current_side
            prob = 0.552 + min(0.035, (fixed_consistency - 0.45) * 0.10)
            return _score(side, prob, f"固定{mode_len}連節奏｜補足{_side_name(side)}", 0.15 + fixed_consistency * 0.08, "補足", target_len=mode_len)
        side = _opposite(current_side)
        prob = 0.555 + min(0.040, (fixed_consistency - 0.45) * 0.12)
        return _score(side, prob, f"固定{mode_len}連節奏｜到點轉邊", 0.16 + fixed_consistency * 0.08, "斷龍/轉邊", target_len=mode_len)

    # Alternating run rhythm, e.g. 1-2-1-2, 2-1-2-1, 1-3-1-3.
    recent = completed[-6:]
    if len(recent) >= 6:
        a_hits = recent[0] == recent[2] == recent[4]
        b_hits = recent[1] == recent[3] == recent[5]
        if a_hits and b_hits and recent[0] != recent[1]:
            # Current run target is usually the length two runs ago.
            target_len = recent[-2]
            rhythm = f"{recent[0]}-{recent[1]}"
            if current_len < target_len:
                side = current_side
                prob = 0.562 if rhythm in {"1-2", "2-1"} else 0.555
                label = f"一房兩廳/兩房一廳{rhythm}｜補足{_side_name(side)}"
                strength = 0.19 if rhythm in {"1-2", "2-1"} else 0.17
                return _score(side, prob, label, strength, "補足", target_len=target_len, rhythm=rhythm)
            side = _opposite(current_side)
            prob = 0.568 if rhythm in {"1-2", "2-1"} else 0.558
            label = f"一房兩廳/兩房一廳{rhythm}｜到點轉邊"
            strength = 0.20 if rhythm in {"1-2", "2-1"} else 0.18
            return _score(side, prob, label, strength, "斷龍/轉邊", target_len=target_len, rhythm=rhythm)

    # Last four may already show A-B-A pattern; predict B target.
    if len(completed) >= 4:
        last4 = completed[-4:]
        if last4[0] == last4[2] and last4[1] != last4[0]:
            target_len = last4[1]
            if current_len < target_len:
                return _score(current_side, 0.548, f"長短龍交替預判｜補到{target_len}連", 0.13, "補足", target_len=target_len)
            return _score(_opposite(current_side), 0.552, f"長短龍交替預判｜{target_len}連轉邊", 0.14, "轉邊", target_len=target_len)

    return _neutral("未見長短固定節奏")


def _single_chop_score(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 6:
        return _neutral("單跳資料不足")
    recent = non_tie[-ROAD_PATTERN_WINDOW:]
    last = recent[-1]
    opp = _opposite(last)
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)

    # exact alternating suffix length
    alt_len = 1
    for i in range(len(non_tie) - 1, 0, -1):
        if non_tie[i] != non_tie[i - 1]:
            alt_len += 1
        else:
            break

    if alt_len >= SINGLE_CHOP_CONFIRM_WINDOW:
        prob = 0.568 + min(0.035, (alt_len - SINGLE_CHOP_CONFIRM_WINDOW) * 0.006)
        return _score(opp, prob, f"單跳成型｜連跳{alt_len}手", 0.20 + min(0.06, alt_len * 0.006), "單跳", alt_len=alt_len, switch_rate=round(switch_rate, 3))

    if switch_rate >= SINGLE_CHOP_MIN_RATE and alt_len >= 4:
        prob = 0.550 + min(0.030, (switch_rate - SINGLE_CHOP_MIN_RATE) * 0.10)
        return _score(opp, prob, f"單跳偏強｜跳率{int(switch_rate * 100)}%", 0.15 + min(0.05, switch_rate - SINGLE_CHOP_MIN_RATE), "單跳", alt_len=alt_len, switch_rate=round(switch_rate, 3))

    # broken chop: when it was alternating then repeats, do not keep forcing chop.
    if len(non_tie) >= 7:
        prev = non_tie[-7:-1]
        prev_alt = all(prev[i] != prev[i - 1] for i in range(1, len(prev)))
        if prev_alt and non_tie[-1] == non_tie[-2]:
            return _score(non_tie[-1], 0.538, "單跳破壞後補同邊", 0.10, "破跳補邊")

    return _neutral("非單跳")


def _double_chop_score(non_tie: List[str]) -> Dict[str, Any]:
    runs = _runs(non_tie)
    if len(runs) < 4:
        return _neutral("雙跳資料不足")
    current_side, current_len = runs[-1]
    recent_runs = runs[-DOUBLE_CHOP_LOOKBACK:]
    recent_lengths = [n for _s, n in recent_runs]
    two_hits = sum(1 for n in recent_lengths[-6:] if n == 2)

    # Clear BB/PP rhythm.
    if len(recent_lengths) >= 4 and two_hits >= DOUBLE_CHOP_MIN_HITS:
        if current_len < 2:
            return _score(current_side, 0.562, f"雙跳/兩房型｜補足{_side_name(current_side)}2連", 0.18, "雙跳補足", target_len=2)
        return _score(_opposite(current_side), 0.566, "雙跳/兩房型｜2連到點轉邊", 0.19, "雙跳轉邊", target_len=2)

    # 2-1-2-1 is not double chop, but a room-rhythm; still give a soft signal here.
    if len(recent_lengths) >= 5:
        tail = recent_lengths[-5:]
        if tail[0] == tail[2] == tail[4] == 2 and tail[1] == tail[3] == 1:
            if current_len >= 2:
                return _score(_opposite(current_side), 0.554, "2-1-2-1節奏｜2連轉邊", 0.15, "轉邊", target_len=2)
            return _score(current_side, 0.548, "2-1-2-1節奏｜補到2連", 0.14, "補足", target_len=2)

    return _neutral("非雙跳")


def _dragon_score(non_tie: List[str]) -> Dict[str, Any]:
    runs = _runs(non_tie)
    if len(non_tie) < 4 or not runs:
        return _neutral("龍型資料不足")
    current_side, n = runs[-1]
    if n < 2:
        return _neutral("未成龍")

    completed_lengths = _recent_completed_lengths(runs, DRAGON_MEMORY_LOOKBACK)
    stats = _run_follow_stats(non_tie, n)

    # base continuation curve: not blindly chase; after fatigue it starts to decline.
    if n == 2:
        cont_prob = 0.522
    elif n == 3:
        cont_prob = 0.545
    elif n == 4:
        cont_prob = 0.565
    elif n == 5:
        cont_prob = 0.582
    elif n == 6:
        cont_prob = 0.590
    elif n == 7:
        cont_prob = 0.584
    else:
        cont_prob = max(0.535, 0.582 - (n - 7) * 0.010)

    # Current-shoe learning: did previous runs break at this exact length or continue beyond it?
    sample = stats["sample"]
    if sample > 0:
        hist_prob = (stats["cont"] + cont_prob * 2.5) / (sample + 2.5)
        hist_weight = min(0.65, sample / 7)
        cont_prob = cont_prob * (1 - hist_weight) + hist_prob * hist_weight

    same_cut_count = sum(1 for x in completed_lengths[-10:] if x == n)
    shorter_count = sum(1 for x in completed_lengths[-10:] if x < n)
    longer_count = sum(1 for x in completed_lengths[-10:] if x > n)

    # If this shoe repeatedly cuts at the current length, switch to break mode.
    if same_cut_count >= DRAGON_BREAK_REPEAT_MIN and n >= DRAGON_MIN_LEN:
        cont_prob -= min(0.070, 0.024 * same_cut_count)

    # If current dragon exceeds most recent completed run lengths, add fatigue.
    if n >= DRAGON_FATIGUE_START and shorter_count >= 5 and longer_count == 0:
        cont_prob -= 0.040

    cont_prob = _clamp(cont_prob, 0.405, 0.625)
    if cont_prob >= 0.5:
        side = current_side
        edge = min(DRAGON_MAX_EDGE, cont_prob - 0.5)
        action = "跟龍" if n >= DRAGON_FOLLOW_MIN_LEN else "補足"
    else:
        side = _opposite(current_side)
        edge = min(DRAGON_BREAK_EDGE, 0.5 - cont_prob)
        action = "斷龍"

    if n >= DRAGON_STRONG_LEN:
        length_name = "長莊" if current_side == "B" else "長閒"
        label = f"{length_name}{n}連｜{action}"
        strength = 0.17 + min(0.08, (n - DRAGON_STRONG_LEN) * 0.012)
    elif n >= DRAGON_MIN_LEN:
        label = f"中龍{_side_name(current_side)}{n}連｜{action}"
        strength = 0.145
    else:
        label = f"短龍{_side_name(current_side)}{n}連｜{action}"
        strength = 0.105

    # Learning evidence boosts strength.
    strength += min(0.045, sample * 0.006)
    if same_cut_count >= DRAGON_BREAK_REPEAT_MIN:
        strength += 0.025

    return _score(side, 0.5 + edge, label, strength, action, streak=n, cont_prob=round(cont_prob, 4), run_stats=stats, same_cut_count=same_cut_count)


def _front_pattern_score(non_tie: List[str]) -> Dict[str, Any]:
    if not FRONT_PATTERN_MATCH or len(non_tie) < 8:
        return _neutral("前排回測資料不足")
    seq = "".join(non_tie)
    max_k = min(FRONT_PATTERN_LOOKBACK, len(seq) - 1)
    best: Optional[Dict[str, Any]] = None
    for k in range(max_k, 2, -1):
        key = seq[-k:]
        follows: List[str] = []
        for i in range(0, len(seq) - k):
            if seq[i:i + k] == key and i + k < len(seq):
                follows.append(seq[i + k])
        if len(follows) >= FRONT_PATTERN_MIN_SAMPLE:
            c = Counter(follows)
            total = c["B"] + c["P"]
            side = "B" if c["B"] >= c["P"] else "P"
            raw = max(c["B"], c["P"]) / total
            shrink = min(0.72, total / 9)
            prob = 0.5 * (1 - shrink) + raw * shrink
            prob = _clamp(prob, 0.525, 0.600)
            strength = min(0.20, 0.075 + total * 0.018 + k * 0.006)
            best = _score(side, prob, f"前排{k}碼回測｜樣本{total}", strength, "前排回測", sample=total, key=key)
            break
    return best or _neutral("前排無重複")


def _road_state_router(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 3:
        return _neutral("資料不足")

    candidates = [
        _single_chop_score(non_tie),
        _double_chop_score(non_tie),
        _length_rhythm_score(non_tie),
        _dragon_score(non_tie),
        _front_pattern_score(non_tie),
    ]

    # Minor short-window scarcity correction, low strength fallback only.
    recent = non_tie[-12:]
    if len(recent) >= 10:
        b_count = recent.count("B")
        p_count = recent.count("P")
        if abs(b_count - p_count) >= 5:
            side = "B" if b_count < p_count else "P"
            candidates.append(_score(side, 0.528, "短窗偏態小修正", 0.065, "均衡"))

    candidates = sorted(candidates, key=lambda x: (float(x.get("strength", 0)), float(x.get("edge", 0))), reverse=True)
    best = dict(candidates[0])
    second = candidates[1] if len(candidates) > 1 else _neutral()

    # Blend compatible second signal, but do not let weak mixed signals neutralize a clear state.
    if second.get("strength", 0) >= 0.13 and second.get("label") != best.get("label"):
        best["secondary_label"] = second.get("label")
        same_pick = (best.get("B", 0.5) >= best.get("P", 0.5)) == (second.get("B", 0.5) >= second.get("P", 0.5))
        blend = 0.20 if same_pick else 0.12
        best["B"] = best["B"] * (1 - blend) + second["B"] * blend
        best["P"] = 1 - best["B"]
        best["strength"] = min(0.35, best.get("strength", 0) + (0.025 if same_pick else 0.0))

    best["candidates"] = [
        {"label": c.get("label"), "B": round(c.get("B", 0.5), 4), "P": round(c.get("P", 0.5), 4), "strength": round(c.get("strength", 0), 3), "action": c.get("action", "")}
        for c in candidates[:5]
    ]
    return best


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
    pressure = T_PRIOR * (1 - TIE_SHRINK) + t_rate * TIE_SHRINK
    if gap_since_tie >= 18:
        pressure += 0.012
    if recent[-4:].count("T") >= 2:
        pressure += 0.018
    return _clamp(pressure, 0.055, TIE_MAX_PROB)


def _dynamic_weights(road: Dict[str, Any], history_len: int) -> Dict[str, float]:
    weights = {
        "markov": MARKOV_WEIGHT,
        "road": ROAD_WEIGHT,
        "streak": STREAK_WEIGHT,
        "balance": BALANCE_WEIGHT,
        "recent": RECENT_WEIGHT,
    }
    if not DYNAMIC_WEIGHT_MODE or history_len < 5:
        return weights

    strength = float(road.get("strength", 0))
    action = str(road.get("action", ""))
    label = str(road.get("label", ""))

    # Clear road state: let road router dominate more.
    if strength >= 0.18:
        weights.update({"road": 0.46, "markov": 0.16, "streak": 0.14, "recent": 0.10, "balance": 0.05})
    elif strength >= 0.13:
        weights.update({"road": 0.40, "markov": 0.18, "streak": 0.15, "recent": 0.10, "balance": 0.07})

    # If router detected a cut point or turn-side rhythm, avoid old streak layer over-following.
    if any(k in action for k in ["斷", "轉邊", "單跳", "雙跳轉邊"]):
        weights["streak"] *= 0.55
        weights["road"] += 0.06
        weights["recent"] += 0.03

    # If following clear dragon, streak can support it.
    if "跟龍" in action or "補足" in action:
        weights["streak"] += 0.03
        weights["road"] += 0.03
        weights["balance"] *= 0.70

    s = sum(weights.values())
    return {k: v / s for k, v in weights.items()}


def _confidence(b: float, p: float, t: float, history_len: int, agreement: float, road_strength: float, road_edge: float) -> Tuple[float, str]:
    gap = abs(b - p)
    base = gap * 3.35 + agreement * 0.18 + road_strength * 0.42 + road_edge * 0.70 + min(0.14, history_len / 90)
    conf = _clamp(base, 0.08, 0.94)
    if history_len < MIN_HISTORY_FOR_SIGNAL:
        return min(conf, 0.35), "冷啟動"
    if conf >= 0.70:
        return conf, "強訊號"
    if conf >= 0.50:
        return conf, "中訊號"
    return conf, "弱訊號"


# -----------------------------
# Public prediction function
# -----------------------------
def predict(history: List[str], venue: str = "", room: str = "", shoe_id: str = "") -> Dict[str, Any]:
    history = [str(x).upper() for x in history if str(x).upper() in {"B", "P", "T"}]
    non_tie = _last_non_tie(history)
    run_data = _runs(non_tie)

    markov = _transition_prob(non_tie)
    road = _road_state_router(non_tie) if ROAD_STATE_ROUTER else _front_pattern_score(non_tie)
    recent = _recent_score(non_tie)
    balance = _balance_score(non_tie)
    streak = _streak_score(non_tie)
    weights = _dynamic_weights(road, len(history))

    b_side = (
        markov["B"] * weights["markov"]
        + road["B"] * weights["road"]
        + streak["B"] * weights["streak"]
        + balance["B"] * weights["balance"]
        + recent["B"] * weights["recent"]
    )
    p_side = 1 - b_side

    tie_prob = _tie_score(history)
    b_prob = b_side * (1 - tie_prob)
    p_prob = p_side * (1 - tie_prob)

    feature_payload = {
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "history_len": len(history),
        "history_tail": "".join(history[-54:]),
        "non_tie_tail": "".join(non_tie[-54:]),
        "runs_tail": run_data[-14:],
        "current_streak": _streak(non_tie),
        "weights": weights,
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

    conf, level = _confidence(
        b_prob,
        p_prob,
        tie_prob,
        len(history),
        agreement,
        float(road.get("strength", 0)),
        float(road.get("edge", 0)),
    )

    reason_parts = [road.get("label", "牌路"), f"模型一致{int(agreement * 100)}%"]
    if road.get("action"):
        reason_parts.append(f"動作:{road.get('action')}")
    if road.get("secondary_label"):
        reason_parts.append(f"副路:{road.get('secondary_label')}")
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
        "road_action": road.get("action", ""),
        "reason": " / ".join(reason_parts),
        "dragon": {
            "current_streak": _streak(non_tie),
            "runs_tail": run_data[-12:],
            "road_strength": round(float(road.get("strength", 0)), 3),
            "road_edge": round(float(road.get("edge", 0)), 3),
            "road_candidates": road.get("candidates", []),
        },
        "ai_used": bool(ai_result and not ai_result.get("error")),
        "ai_result": ai_result if os.getenv("DEBUG_AI_RESULT", "0") == "1" else None,
        "debug": feature_payload if os.getenv("DEBUG_PREDICTOR", "0") == "1" else None,
    }
