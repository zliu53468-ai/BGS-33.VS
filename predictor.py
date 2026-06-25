import math
import os
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Dict, List, Tuple

from deepseek_client import DeepSeekClient

# Base baccarat long-run priors, used only as soft priors for display calibration.
B_PRIOR = float(os.getenv("B_PRIOR", "0.4586"))
P_PRIOR = float(os.getenv("P_PRIOR", "0.4462"))
T_PRIOR = float(os.getenv("T_PRIOR", "0.0952"))

# Main ensemble weights.
MARKOV_WEIGHT = float(os.getenv("MARKOV_WEIGHT", "0.22"))
ROAD_WEIGHT = float(os.getenv("ROAD_WEIGHT", "0.30"))
STREAK_WEIGHT = float(os.getenv("STREAK_WEIGHT", "0.16"))
BALANCE_WEIGHT = float(os.getenv("BALANCE_WEIGHT", "0.10"))
RECENT_WEIGHT = float(os.getenv("RECENT_WEIGHT", "0.10"))
TIE_WEIGHT = float(os.getenv("TIE_WEIGHT", "0.04"))
AI_BLEND = float(os.getenv("AI_BLEND", "0.14"))

# Tie handling. Tie should usually be a warning layer, not a main recommendation.
TIE_SHRINK = float(os.getenv("TIE_SHRINK", "0.35"))
TIE_MAX_PROB = float(os.getenv("TIE_MAX_PROB", "0.18"))
ALLOW_TIE_RECOMMEND = os.getenv("ALLOW_TIE_RECOMMEND", "0") == "1"
TIE_RECOMMEND_MIN = float(os.getenv("TIE_RECOMMEND_MIN", "0.165"))

MIN_HISTORY_FOR_AI = int(os.getenv("MIN_HISTORY_FOR_AI", "6"))
MIN_HISTORY_FOR_SIGNAL = int(os.getenv("MIN_HISTORY_FOR_SIGNAL", "4"))

# Advanced road / dragon controls.
DRAGON_MIN_LEN = int(os.getenv("DRAGON_MIN_LEN", "3"))
DRAGON_STRONG_LEN = int(os.getenv("DRAGON_STRONG_LEN", "5"))
DRAGON_FATIGUE_START = int(os.getenv("DRAGON_FATIGUE_START", "8"))
DRAGON_MAX_EDGE = float(os.getenv("DRAGON_MAX_EDGE", "0.105"))
DRAGON_BREAK_EDGE = float(os.getenv("DRAGON_BREAK_EDGE", "0.080"))
RUN_CYCLE_MIN_HITS = int(os.getenv("RUN_CYCLE_MIN_HITS", "3"))
ROAD_PATTERN_WINDOW = int(os.getenv("ROAD_PATTERN_WINDOW", "16"))
PATTERN_LOOKBACK = int(os.getenv("PATTERN_LOOKBACK", "5"))
MARKOV_ALPHA = float(os.getenv("MARKOV_ALPHA", "2.6"))
MARKOV_FULL_SAMPLE = float(os.getenv("MARKOV_FULL_SAMPLE", "16"))

# Breakout Dragon Mode:
# Handles shoes where a Banker/Player dragon suddenly exceeds previous run lengths.
# It protects 1~2 hands after a true breakout so the model does not force-break too early.
DRAGON_MEMORY_LOOKBACK = int(os.getenv("DRAGON_MEMORY_LOOKBACK", "28"))
DRAGON_BREAK_REPEAT_MIN = int(os.getenv("DRAGON_BREAK_REPEAT_MIN", "3"))
BREAKOUT_DRAGON_MODE = os.getenv("BREAKOUT_DRAGON_MODE", "1") == "1"
BREAKOUT_MIN_LEN = int(os.getenv("BREAKOUT_MIN_LEN", str(DRAGON_STRONG_LEN)))
BREAKOUT_PROTECT_STEPS = int(os.getenv("BREAKOUT_PROTECT_STEPS", "2"))
BREAKOUT_EXTEND_STEPS = int(os.getenv("BREAKOUT_EXTEND_STEPS", "2"))
BREAKOUT_CONT_EDGE = float(os.getenv("BREAKOUT_CONT_EDGE", "0.038"))
BREAKOUT_NEW_HIGH_BONUS = float(os.getenv("BREAKOUT_NEW_HIGH_BONUS", "0.014"))
BREAKOUT_OVERHEAT_PENALTY = float(os.getenv("BREAKOUT_OVERHEAT_PENALTY", "0.018"))

# Chaos / broken-road regime controls.
# These protect the model from forcing dragon/chop/double-chop logic on unstable shoes.
CHAOS_MODE = os.getenv("CHAOS_MODE", "1") == "1"
PATTERN_FAILURE_COUNTER = os.getenv("PATTERN_FAILURE_COUNTER", "1") == "1"
FAKE_DRAGON_DETECTOR = os.getenv("FAKE_DRAGON_DETECTOR", "1") == "1"
CHOP_BREAK_DETECTOR = os.getenv("CHOP_BREAK_DETECTOR", "1") == "1"
LOW_CONFIDENCE_MINBET = os.getenv("LOW_CONFIDENCE_MINBET", "1") == "1"

CHAOS_WINDOW = int(os.getenv("CHAOS_WINDOW", "18"))
CHAOS_TRIGGER = float(os.getenv("CHAOS_TRIGGER", "0.58"))
CHAOS_STRONG_TRIGGER = float(os.getenv("CHAOS_STRONG_TRIGGER", "0.72"))
CHAOS_MAX_EDGE = float(os.getenv("CHAOS_MAX_EDGE", "0.045"))
CHAOS_RECENT_BLEND = float(os.getenv("CHAOS_RECENT_BLEND", "0.26"))
CHAOS_CONF_CAP = float(os.getenv("CHAOS_CONF_CAP", "0.46"))
CHAOS_STRONG_CONF_CAP = float(os.getenv("CHAOS_STRONG_CONF_CAP", "0.38"))
CHAOS_AI_BLEND_FACTOR = float(os.getenv("CHAOS_AI_BLEND_FACTOR", "0.55"))

# Dynamic weight factors used only when chaos mode is active.
CHAOS_MARKOV_WEIGHT_FACTOR = float(os.getenv("CHAOS_MARKOV_WEIGHT_FACTOR", "1.22"))
CHAOS_ROAD_WEIGHT_FACTOR = float(os.getenv("CHAOS_ROAD_WEIGHT_FACTOR", "0.45"))
CHAOS_STREAK_WEIGHT_FACTOR = float(os.getenv("CHAOS_STREAK_WEIGHT_FACTOR", "0.48"))
CHAOS_BALANCE_WEIGHT_FACTOR = float(os.getenv("CHAOS_BALANCE_WEIGHT_FACTOR", "0.70"))
CHAOS_RECENT_WEIGHT_FACTOR = float(os.getenv("CHAOS_RECENT_WEIGHT_FACTOR", "1.60"))

# Chaos scoring knobs.
CHAOS_MIN_RUN_VARIETY = int(os.getenv("CHAOS_MIN_RUN_VARIETY", "3"))
CHAOS_MODE_CONSISTENCY_MAX = float(os.getenv("CHAOS_MODE_CONSISTENCY_MAX", "0.48"))
CHAOS_SWITCH_DIFF_TRIGGER = float(os.getenv("CHAOS_SWITCH_DIFF_TRIGGER", "0.28"))
ROUTE_SWITCH_PENALTY = float(os.getenv("ROUTE_SWITCH_PENALTY", "0.11"))
FAKE_DRAGON_MIN_LEN = int(os.getenv("FAKE_DRAGON_MIN_LEN", "4"))
FAKE_DRAGON_SHORT_RUN_RATE = float(os.getenv("FAKE_DRAGON_SHORT_RUN_RATE", "0.62"))
CHOP_BREAK_MIN_PRE_RATE = float(os.getenv("CHOP_BREAK_MIN_PRE_RATE", "0.72"))
CHOP_BREAK_DROP = float(os.getenv("CHOP_BREAK_DROP", "0.22"))


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
    prob = _clamp(prob, 0.35, 0.65)
    return (prob, 1 - prob) if side == "B" else (1 - prob, prob)


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


def _run_follow_stats(non_tie: List[str], current_len: int) -> Dict[str, Any]:
    """
    Estimate whether a run tends to continue or break when it reaches current_len.
    A completed run with length > current_len means it continued at this length.
    A completed run with length == current_len means it broke at this length.
    """
    completed = _runs(non_tie)[:-1]
    cont = 0
    brk = 0
    nearby_cont = 0
    nearby_brk = 0
    for _side, length in completed:
        if length > current_len:
            cont += 1
        elif length == current_len:
            brk += 1
        if length > max(1, current_len - 1):
            nearby_cont += 1
        elif length == max(1, current_len - 1):
            nearby_brk += 1
    return {
        "cont": cont,
        "break": brk,
        "nearby_cont": nearby_cont,
        "nearby_break": nearby_brk,
        "sample": cont + brk,
        "nearby_sample": nearby_cont + nearby_brk,
    }


def _dragon_prior_cont_prob(n: int) -> float:
    """
    Conservative continuation prior.
    Important change: fatigue now follows DRAGON_FATIGUE_START instead of hard-stopping too early.
    This avoids forcing a break when a dragon is just starting to exceed earlier history.
    """
    if n <= 1:
        return 0.500
    if n == 2:
        return 0.522
    if n == 3:
        return 0.548
    if n == 4:
        return 0.568
    if n == 5:
        return 0.586
    if n == 6:
        return 0.596
    if n == 7:
        return 0.594

    # Between 8 and fatigue start, keep continuation alive with only mild decay.
    if n < DRAGON_FATIGUE_START:
        return max(0.570, 0.594 - max(0, n - 7) * 0.004)

    # True fatigue starts only after DRAGON_FATIGUE_START.
    return max(0.540, 0.585 - (n - DRAGON_FATIGUE_START + 1) * 0.010)


def _breakout_dragon_context(non_tie: List[str], side: str, n: int) -> Dict[str, Any]:
    """
    Breakout Dragon Mode.
    If the current dragon length exceeds recent historical dragon lengths,
    do not immediately force a break. Protect the first 1~2 hands after breakout,
    then gradually switch to fatigue / break-risk checking.

    Example:
    previous max P run = 4, current P run = 5 or 6 -> continuation protection
    current P run = 8~10 -> reduce continuation and check fatigue
    """
    if not BREAKOUT_DRAGON_MODE or not side or n < BREAKOUT_MIN_LEN:
        return {"active": False}

    runs = _runs(non_tie)
    if len(runs) < 4:
        return {"active": False}

    completed = runs[:-1]
    recent = completed[-DRAGON_MEMORY_LOOKBACK:] if DRAGON_MEMORY_LOOKBACK > 0 else completed
    same_side_lengths = [length for s, length in recent if s == side]
    all_lengths = [length for _s, length in recent]

    if not all_lengths:
        return {"active": False}

    max_same = max(same_side_lengths) if same_side_lengths else 0
    max_all = max(all_lengths)
    baseline = max_same if max_same > 0 else max_all

    # Not a breakout yet.
    if n <= baseline:
        return {
            "active": False,
            "max_same": max_same,
            "max_all": max_all,
            "baseline": baseline,
        }

    over = n - baseline
    is_new_shoe_high = n > max_all

    # Protect the first hands after the run breaks prior history.
    if over <= BREAKOUT_PROTECT_STEPS:
        cont_adjust = BREAKOUT_CONT_EDGE + (over - 1) * 0.010
        phase = "breakout_protect"
        label = f"突破龍{side}{n}｜突破前高續龍保護"
        strength_bonus = 0.045
    elif over <= BREAKOUT_PROTECT_STEPS + BREAKOUT_EXTEND_STEPS:
        # Still can continue, but confidence should decay.
        cont_adjust = max(0.010, BREAKOUT_CONT_EDGE * 0.55 - (over - BREAKOUT_PROTECT_STEPS - 1) * 0.006)
        phase = "breakout_extend"
        label = f"突破龍{side}{n}｜延伸續龍但降信心"
        strength_bonus = 0.025
    else:
        # Overextended beyond the protected zone. Start allowing fatigue pressure.
        cont_adjust = -BREAKOUT_OVERHEAT_PENALTY if n >= DRAGON_FATIGUE_START else 0.004
        phase = "breakout_overheat" if n >= DRAGON_FATIGUE_START else "breakout_late"
        label = f"突破龍{side}{n}｜突破過熱觀察斷點"
        strength_bonus = 0.010

    if is_new_shoe_high and phase in {"breakout_protect", "breakout_extend"}:
        cont_adjust += BREAKOUT_NEW_HIGH_BONUS
        label += "｜本靴新高"

    # If the shoe repeatedly broke at this length before, reduce breakout protection.
    same_cut_count = sum(1 for length in all_lengths[-min(len(all_lengths), 12):] if length == n)
    if same_cut_count >= DRAGON_BREAK_REPEAT_MIN:
        cont_adjust -= min(0.030, same_cut_count * 0.010)
        label += "｜同長度斷點壓力"

    return {
        "active": True,
        "phase": phase,
        "side": side,
        "length": n,
        "over": over,
        "max_same": max_same,
        "max_all": max_all,
        "baseline": baseline,
        "is_new_shoe_high": is_new_shoe_high,
        "cont_adjust": round(cont_adjust, 5),
        "strength_bonus": strength_bonus,
        "label": label,
    }

def _dragon_score(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 4:
        return {"B": 0.5, "P": 0.5, "label": "龍型資料不足", "strength": 0.0}

    last, n = _streak(non_tie)
    if not last or n < 2:
        return {"B": 0.5, "P": 0.5, "label": "未成龍", "strength": 0.0, "streak": n}

    stats = _run_follow_stats(non_tie, n)
    prior = _dragon_prior_cont_prob(n)

    # Current-shoe run cutoff learning, shrunk heavily when samples are low.
    sample = stats["sample"]
    hist_prob = (stats["cont"] + prior * 3.0) / (sample + 3.0) if sample else prior
    hist_weight = min(0.62, sample / 7.0)
    cont_prob = prior * (1 - hist_weight) + hist_prob * hist_weight

    completed = _runs(non_tie)[:-1]
    recent_completed = completed[-DRAGON_MEMORY_LOOKBACK:] if DRAGON_MEMORY_LOOKBACK > 0 else completed
    recent_lengths = [length for _s, length in recent_completed]
    same_side_lengths = [length for s, length in recent_completed if s == last]
    same_cut_count = sum(1 for length in recent_lengths[-12:] if length == n)
    below_cut_count = sum(1 for length in recent_lengths[-12:] if length < n)
    max_same = max(same_side_lengths) if same_side_lengths else 0
    max_all = max(recent_lengths) if recent_lengths else 0

    breakout = _breakout_dragon_context(non_tie, last, n)
    breakout_phase = breakout.get("phase", "") if breakout.get("active") else ""

    # Breakout Dragon Mode: if this dragon just exceeded history, protect continuation first.
    if breakout.get("active"):
        cont_prob += float(breakout.get("cont_adjust", 0.0))

    # If many recent runs ended exactly at this length, add break pressure.
    # During breakout protection, reduce this penalty so the model does not break too early.
    if same_cut_count >= DRAGON_BREAK_REPEAT_MIN and n >= DRAGON_MIN_LEN:
        penalty = min(0.055, same_cut_count * 0.014)
        if breakout_phase in {"breakout_protect", "breakout_extend"}:
            penalty *= 0.35
        cont_prob -= penalty

    # If many recent runs could not reach this length, very long dragons should be chased with lower confidence.
    # But do not punish the first breakout hands too heavily.
    if n >= DRAGON_FATIGUE_START and below_cut_count >= 4:
        fatigue_penalty = 0.030
        if breakout_phase in {"breakout_protect", "breakout_extend"}:
            fatigue_penalty *= 0.40
        cont_prob -= fatigue_penalty

    # If current dragon is above same-side historical max but not in protected breakout mode,
    # avoid hard reverse: keep it near neutral instead of forcing break.
    if max_same and n > max_same and not breakout.get("active"):
        cont_prob = max(cont_prob, 0.515)

    cont_prob = _clamp(cont_prob, 0.405, 0.642)
    side = last if cont_prob >= 0.5 else _opposite(last)
    prob = 0.5 + min(DRAGON_MAX_EDGE, abs(cont_prob - 0.5))
    if cont_prob < 0.5:
        prob = 0.5 + min(DRAGON_BREAK_EDGE, abs(cont_prob - 0.5))
    b, p = _bp_score(side, prob)

    if breakout.get("active"):
        label = str(breakout.get("label", f"突破龍{last}{n}"))
        strength = 0.18 + float(breakout.get("strength_bonus", 0.0))
        if side != last:
            label += "｜轉斷"
    elif n >= DRAGON_STRONG_LEN:
        label = f"長龍{last}{n}｜{'續龍' if side == last else '斷龍壓力'}"
        strength = 0.18 + min(0.06, (n - DRAGON_STRONG_LEN) * 0.012)
    elif n >= DRAGON_MIN_LEN:
        label = f"中龍{last}{n}｜{'續龍' if side == last else '斷龍壓力'}"
        strength = 0.145
    else:
        label = f"短龍{last}{n}｜{'續龍' if side == last else '斷龍壓力'}"
        strength = 0.105

    strength *= 0.85 + min(0.25, sample * 0.03)
    if breakout.get("active"):
        strength += min(0.035, float(breakout.get("over", 0)) * 0.010)

    action = "續龍" if side == last else "斷龍/轉邊"
    return {
        "B": b,
        "P": p,
        "label": label,
        "strength": _clamp(strength, 0.05, 0.30),
        "streak": n,
        "dragon_side": last,
        "cont_prob": round(cont_prob, 4),
        "road_action": action,
        "breakout": breakout,
        "run_stats": {
            **stats,
            "max_same": max_same,
            "max_all": max_all,
            "same_cut_count": same_cut_count,
            "below_cut_count": below_cut_count,
        },
    }

def _run_cycle_score(non_tie: List[str]) -> Dict[str, Any]:
    runs = _runs(non_tie)
    if len(runs) < 5:
        return {"B": 0.5, "P": 0.5, "label": "跑法資料不足", "strength": 0.0}

    current_side, current_len = runs[-1]
    completed_lengths = [n for _s, n in runs[:-1]]
    recent = completed_lengths[-8:]

    # Fixed cut length: e.g. BB PP BB PP, or BBB PPP BBB PPP.
    c = Counter(recent)
    mode_len, mode_count = c.most_common(1)[0]
    if mode_count >= RUN_CYCLE_MIN_HITS and len(recent) >= 5:
        consistency = mode_count / len(recent)
        if consistency >= 0.48:
            side = _opposite(current_side) if current_len >= mode_len else current_side
            prob = 0.545 + min(0.045, (consistency - 0.48) * 0.12)
            b, p = _bp_score(side, prob)
            return {
                "B": b,
                "P": p,
                "label": f"固定{mode_len}連節奏｜{'轉邊' if side != current_side else '補足'}",
                "strength": _clamp(0.12 + consistency * 0.09, 0.10, 0.21),
                "target_len": mode_len,
                "consistency": round(consistency, 3),
            }

    # Period-2 run rhythm: e.g. 1,2,1,2 or 2,3,2,3.
    if len(recent) >= 6:
        last6 = recent[-6:]
        if last6[0] == last6[2] == last6[4] and last6[1] == last6[3] == last6[5] and last6[0] != last6[1]:
            target = last6[-2]  # current run usually mirrors the run two positions ago.
            side = _opposite(current_side) if current_len >= target else current_side
            prob = 0.565
            b, p = _bp_score(side, prob)
            return {
                "B": b,
                "P": p,
                "label": f"長短龍交替{last6[0]}-{last6[1]}｜{'轉邊' if side != current_side else '補足'}",
                "strength": 0.18,
                "target_len": target,
            }

    return {"B": 0.5, "P": 0.5, "label": "未見固定龍節奏", "strength": 0.0}


def _chop_score(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 6:
        return {"B": 0.5, "P": 0.5, "label": "跳路資料不足", "strength": 0.0}
    recent = non_tie[-ROAD_PATTERN_WINDOW:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)
    last = recent[-1]
    opp = _opposite(last)

    # pure chop / single jump.
    if switch_rate >= 0.72 and len(recent) >= 7:
        prob = 0.555 + min(0.035, (switch_rate - 0.72) * 0.12)
        b, p = _bp_score(opp, prob)
        return {"B": b, "P": p, "label": "跳路偏強", "strength": 0.16 + min(0.04, switch_rate - 0.72), "switch_rate": switch_rate}

    # double chop / two-room pattern, detected by recent run lengths mostly 2.
    run_lengths = [n for _s, n in _runs(recent)]
    if len(run_lengths) >= 4 and sum(1 for n in run_lengths[-5:] if n == 2) >= 3:
        current_side, current_len = _streak(non_tie)
        side = _opposite(current_side) if current_len >= 2 else current_side
        b, p = _bp_score(side, 0.56)
        return {"B": b, "P": p, "label": "雙跳/兩房型", "strength": 0.165, "switch_rate": switch_rate}

    return {"B": 0.5, "P": 0.5, "label": "非跳路", "strength": 0.0, "switch_rate": switch_rate}


def _pattern_memory_score(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 8:
        return {"B": 0.5, "P": 0.5, "label": "回測資料不足", "strength": 0.0}

    seq = "".join(non_tie)
    best = None
    for k in range(PATTERN_LOOKBACK, 2, -1):
        key = seq[-k:]
        follows: List[str] = []
        for i in range(0, len(seq) - k):
            if seq[i:i + k] == key and i + k < len(seq):
                follows.append(seq[i + k])
        if follows:
            c = Counter(follows)
            total = c["B"] + c["P"]
            b_raw = c["B"] / total
            shrink = min(0.70, total / 10)
            b = 0.5 * (1 - shrink) + b_raw * shrink
            p = 1 - b
            strength = min(0.18, 0.07 + total * 0.014 + k * 0.006)
            best = {"B": b, "P": p, "label": f"{k}碼回測{key}", "strength": strength, "sample": total}
            break
    return best or {"B": 0.5, "P": 0.5, "label": "無回測重複", "strength": 0.0}



def _run_category(length: int) -> str:
    if length <= 1:
        return "S1"
    if length == 2:
        return "S2"
    if length <= 4:
        return "MID"
    return "LONG"


def _window_switch_rate(seq: List[str]) -> float:
    if len(seq) < 2:
        return 0.5
    return _safe_div(sum(1 for a, b in zip(seq, seq[1:]) if a != b), len(seq) - 1, 0.5)


def _chaos_regime_score(non_tie: List[str], history: List[str]) -> Dict[str, Any]:
    """
    Detect broken-road / fake-pattern shoes.

    Goal:
    - If the shoe has stable road state, let road/dragon/chop models work.
    - If every pattern breaks after 1~3 hands, lower ROAD/STREAK confidence and
      switch to short-window Markov + Recent with weak signal.
    """
    if not CHAOS_MODE or len(non_tie) < 10:
        return {"active": False, "score": 0.0, "label": "混亂偵測資料不足", "B": 0.5, "P": 0.5}

    window = max(10, CHAOS_WINDOW)
    recent = non_tie[-window:]
    runs = _runs(recent)
    lengths = [n for _s, n in runs]
    current_side, current_len = _streak(non_tie)

    score = 0.0
    reasons: List[str] = []
    metrics: Dict[str, Any] = {}

    switch_rate = _window_switch_rate(recent)
    half = max(4, len(recent) // 2)
    first_rate = _window_switch_rate(recent[:half])
    second_rate = _window_switch_rate(recent[-half:])
    switch_diff = abs(first_rate - second_rate)
    metrics.update({
        "switch_rate": round(switch_rate, 3),
        "first_switch_rate": round(first_rate, 3),
        "second_switch_rate": round(second_rate, 3),
        "switch_diff": round(switch_diff, 3),
    })

    # 1) Mixed switching: not a clean chop, not a stable dragon.
    if 0.38 <= switch_rate <= 0.72:
        score += 0.13
        reasons.append("切換率混合")

    # 2) Switch-rate changed too much between early and late windows.
    if switch_diff >= CHAOS_SWITCH_DIFF_TRIGGER:
        score += 0.16
        reasons.append("路態前後切換")

    if lengths:
        capped = [min(5, n) for n in lengths]
        variety = len(set(capped))
        c = Counter(capped)
        mode_len, mode_count = c.most_common(1)[0]
        mode_consistency = mode_count / len(capped)
        short_rate = sum(1 for n in lengths if n <= 2) / len(lengths)
        metrics.update({
            "run_lengths": lengths[-10:],
            "run_variety": variety,
            "mode_len": mode_len,
            "mode_consistency": round(mode_consistency, 3),
            "short_run_rate": round(short_rate, 3),
        })

        # 3) Run lengths are too scattered; no stable 1/2/3 rhythm.
        if variety >= CHAOS_MIN_RUN_VARIETY and mode_consistency <= CHAOS_MODE_CONSISTENCY_MAX:
            score += 0.18
            reasons.append("龍長分散")

        # 4) Frequent route category switching: S1 -> LONG -> S2 -> MID...
        cats = [_run_category(n) for n in lengths[-8:]]
        if len(cats) >= 5:
            cat_switches = sum(1 for a, b in zip(cats, cats[1:]) if a != b)
            cat_switch_rate = cat_switches / max(1, len(cats) - 1)
            metrics["category_switch_rate"] = round(cat_switch_rate, 3)
            if cat_switch_rate >= 0.62:
                score += ROUTE_SWITCH_PENALTY
                reasons.append("規律類型頻繁切換")

        # 5) Fake dragon: many short runs, then a dragon appears in a messy regime.
        if FAKE_DRAGON_DETECTOR and current_len >= FAKE_DRAGON_MIN_LEN and short_rate >= FAKE_DRAGON_SHORT_RUN_RATE:
            breakout = _breakout_dragon_context(non_tie, current_side, current_len)
            if breakout.get("active") and breakout.get("phase") in {"breakout_protect", "breakout_extend"}:
                # True breakout should not be punished too hard.
                score += 0.04
                reasons.append("短路後突破龍觀察")
            else:
                score += 0.14
                reasons.append("假長龍風險")

        # 6) Double-chop was expected but got broken.
        if PATTERN_FAILURE_COUNTER and len(lengths) >= 5:
            prev5 = lengths[-6:-1]
            if prev5 and sum(1 for n in prev5 if n == 2) >= 3 and lengths[-1] != 2:
                score += 0.10
                reasons.append("雙跳破壞")

    # 7) Single chop got broken: previous window was jumpy, latest window no longer is.
    if CHOP_BREAK_DETECTOR and len(recent) >= 10:
        pre = recent[-10:-3]
        post = recent[-5:]
        pre_rate = _window_switch_rate(pre)
        post_rate = _window_switch_rate(post)
        metrics.update({"pre_chop_rate": round(pre_rate, 3), "post_chop_rate": round(post_rate, 3)})
        if pre_rate >= CHOP_BREAK_MIN_PRE_RATE and (pre_rate - post_rate) >= CHOP_BREAK_DROP:
            score += 0.12
            reasons.append("單跳破壞")

    # 8) Alternating tail failed at the end.
    if PATTERN_FAILURE_COUNTER and len(recent) >= 8:
        before = recent[-8:-2]
        before_rate = _window_switch_rate(before)
        if before_rate >= 0.80 and recent[-1] == recent[-2]:
            score += 0.10
            reasons.append("單跳尾端失敗")

    # If a clean breakout dragon is active, reduce chaos slightly so it can continue 1~2 hands.
    if current_side and current_len >= BREAKOUT_MIN_LEN:
        breakout = _breakout_dragon_context(non_tie, current_side, current_len)
        if breakout.get("active") and breakout.get("phase") in {"breakout_protect", "breakout_extend"}:
            score -= 0.06
            metrics["breakout_guard"] = breakout.get("phase")

    score = _clamp(score, 0.0, 1.0)
    active = score >= CHAOS_TRIGGER
    strong = score >= CHAOS_STRONG_TRIGGER

    # Chaos recommendation source: short-window transition + recent trend, capped to a small edge.
    tail = recent[-min(12, len(recent)):]
    short_markov = _transition_prob(tail)
    short_recent = _recent_score(tail)
    b_raw = short_markov["B"] * 0.55 + short_recent["B"] * 0.45
    edge_cap = CHAOS_MAX_EDGE * (0.70 if strong else 1.0)
    b = 0.5 + _clamp(b_raw - 0.5, -edge_cap, edge_cap)
    p = 1 - b

    if not reasons:
        reasons.append("路態穩定")
    label = "混亂盤/破路盤｜" + "+".join(reasons[:3]) if active else "路態尚可｜" + "+".join(reasons[:2])

    return {
        "active": active,
        "strong": strong,
        "score": round(score, 3),
        "B": b,
        "P": p,
        "label": label,
        "reasons": reasons,
        "metrics": metrics,
        "short_markov": short_markov,
        "short_recent": short_recent,
    }


def _effective_weights(chaos: Dict[str, Any]) -> Dict[str, float]:
    weights = {
        "markov": MARKOV_WEIGHT,
        "road": ROAD_WEIGHT,
        "streak": STREAK_WEIGHT,
        "balance": BALANCE_WEIGHT,
        "recent": RECENT_WEIGHT,
    }
    if chaos.get("active"):
        strong = bool(chaos.get("strong"))
        road_factor = CHAOS_ROAD_WEIGHT_FACTOR * (0.72 if strong else 1.0)
        streak_factor = CHAOS_STREAK_WEIGHT_FACTOR * (0.80 if strong else 1.0)
        recent_factor = CHAOS_RECENT_WEIGHT_FACTOR * (1.08 if strong else 1.0)
        markov_factor = CHAOS_MARKOV_WEIGHT_FACTOR * (1.05 if strong else 1.0)
        weights = {
            "markov": MARKOV_WEIGHT * markov_factor,
            "road": ROAD_WEIGHT * road_factor,
            "streak": STREAK_WEIGHT * streak_factor,
            "balance": BALANCE_WEIGHT * CHAOS_BALANCE_WEIGHT_FACTOR,
            "recent": RECENT_WEIGHT * recent_factor,
        }
    return weights

def _road_pattern_score(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 3:
        return {"B": 0.5, "P": 0.5, "label": "資料不足", "strength": 0.0}

    candidates = [
        _dragon_score(non_tie),
        _run_cycle_score(non_tie),
        _chop_score(non_tie),
        _pattern_memory_score(non_tie),
    ]

    # Short-window balance is low priority fallback.
    recent = non_tie[-12:]
    b_count = recent.count("B")
    p_count = recent.count("P")
    if abs(b_count - p_count) >= 4:
        scarce = "B" if b_count < p_count else "P"
        b, p = _bp_score(scarce, 0.535)
        candidates.append({"B": b, "P": p, "label": "短窗均衡修正", "strength": 0.09})

    # Choose strongest road mode, but keep second mode for reason/debug.
    candidates = sorted(candidates, key=lambda x: float(x.get("strength", 0)), reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    best = dict(best)
    if second and second.get("strength", 0) >= 0.12 and second.get("label") != best.get("label"):
        best["secondary_label"] = second.get("label")
        # Small blend to avoid one pattern totally dominating another valid road mode.
        blend = 0.25
        best["B"] = best["B"] * (1 - blend) + second["B"] * blend
        best["P"] = 1 - best["B"]
    return best


def _recent_score(non_tie: List[str]) -> Dict[str, float]:
    if not non_tie:
        return {"B": 0.5, "P": 0.5}
    recent = non_tie[-10:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)
    last, n = _streak(non_tie)
    opp = _opposite(last) if last else "B"
    if switch_rate > 0.68:
        side = opp
        edge = 0.052
    elif n >= DRAGON_MIN_LEN:
        # Use dragon length-sensitive continuation, but dampen after fatigue point.
        side = last
        edge = 0.040 + min(0.040, (n - DRAGON_MIN_LEN) * 0.010)
        if n >= DRAGON_FATIGUE_START:
            edge *= 0.75
    else:
        b_count = recent.count("B")
        p_count = recent.count("P")
        side = "B" if b_count < p_count else "P"
        edge = min(0.032, abs(b_count - p_count) * 0.0055)
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}


def _balance_score(non_tie: List[str]) -> Dict[str, float]:
    if len(non_tie) < 8:
        return {"B": 0.5, "P": 0.5}
    b = non_tie.count("B")
    p = non_tie.count("P")
    diff = b - p
    edge = min(0.050, abs(diff) / max(1, len(non_tie)) * 0.14)
    side = "B" if diff < 0 else "P"
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}


def _streak_score(non_tie: List[str]) -> Dict[str, float]:
    last, n = _streak(non_tie)
    if not last:
        return {"B": 0.5, "P": 0.5}
    opp = _opposite(last)
    if n == 1:
        side, edge = opp, 0.022
    elif n == 2:
        side, edge = last, 0.030
    elif n == 3:
        side, edge = last, 0.045
    elif n == 4:
        side, edge = last, 0.058
    elif n < DRAGON_FATIGUE_START:
        side, edge = last, min(0.085, 0.062 + (n - 4) * 0.008)
    else:
        # Very long dragon: still can continue, but do not keep increasing confidence blindly.
        side, edge = last, 0.068
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
    pressure = T_PRIOR * (1 - TIE_SHRINK) + t_rate * TIE_SHRINK
    if gap_since_tie >= 18:
        pressure += 0.012
    if recent[-4:].count("T") >= 2:
        pressure += 0.018
    return _clamp(pressure, 0.055, TIE_MAX_PROB)


def _confidence(b: float, p: float, t: float, history_len: int, agreement: float, road_strength: float) -> Tuple[float, str]:
    gap = abs(b - p)
    base = gap * 3.4 + agreement * 0.20 + road_strength * 0.34 + min(0.15, history_len / 85)
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
    run_data = _runs(non_tie)
    chaos = _chaos_regime_score(non_tie, history)
    weights = _effective_weights(chaos)

    total_w = weights["markov"] + weights["road"] + weights["streak"] + weights["balance"] + weights["recent"]
    b_side = (
        markov["B"] * weights["markov"]
        + road["B"] * weights["road"]
        + streak["B"] * weights["streak"]
        + balance["B"] * weights["balance"]
        + recent["B"] * weights["recent"]
    ) / total_w
    p_side = 1 - b_side

    # In chaos mode, blend toward short-window Markov/Recent and cap confidence later.
    if chaos.get("active"):
        chaos_blend = _clamp(CHAOS_RECENT_BLEND + max(0.0, float(chaos.get("score", 0)) - CHAOS_TRIGGER) * 0.22, 0.12, 0.42)
        b_side = b_side * (1 - chaos_blend) + float(chaos.get("B", 0.5)) * chaos_blend
        p_side = 1 - b_side

    tie_prob = _tie_score(history)
    b_prob = b_side * (1 - tie_prob)
    p_prob = p_side * (1 - tie_prob)

    feature_payload = {
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "history_len": len(history),
        "history_tail": "".join(history[-48:]),
        "non_tie_tail": "".join(non_tie[-48:]),
        "runs_tail": run_data[-12:],
        "current_streak": _streak(non_tie),
        "markov": markov,
        "road": road,
        "recent": recent,
        "balance": balance,
        "streak": streak,
        "chaos": chaos,
        "effective_weights": {k: round(v, 5) for k, v in weights.items()},
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
                if chaos.get("active"):
                    blend *= CHAOS_AI_BLEND_FACTOR
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
    if chaos.get("active"):
        votes.append("B" if chaos.get("B", 0.5) >= chaos.get("P", 0.5) else "P")
    main_pick = "B" if b_prob >= p_prob else "P"
    agreement = votes.count(main_pick) / len(votes)

    if ALLOW_TIE_RECOMMEND and tie_prob >= TIE_RECOMMEND_MIN and tie_prob > max(b_prob, p_prob) * 0.55:
        recommend = "T"
    else:
        recommend = main_pick

    road_strength_for_conf = float(road.get("strength", 0))
    if chaos.get("active"):
        road_strength_for_conf *= 0.45 if chaos.get("strong") else 0.62
    conf, level = _confidence(b_prob, p_prob, tie_prob, len(history), agreement, road_strength_for_conf)
    if chaos.get("active"):
        cap = CHAOS_STRONG_CONF_CAP if chaos.get("strong") else CHAOS_CONF_CAP
        conf = min(conf, cap)
        level = "混亂盤低信心" if chaos.get("strong") else "混亂盤弱訊號"
    reason_parts = [road.get("label", "牌路"), f"模型一致{int(agreement * 100)}%"]
    if chaos.get("active"):
        reason_parts.insert(0, f"{chaos.get('label')}({int(float(chaos.get('score', 0))*100)}%)")
        if LOW_CONFIDENCE_MINBET:
            reason_parts.append("建議最小注")
    if road.get("road_action"):
        reason_parts.append(f"動作:{road.get('road_action')}")
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
        "bet_mode": "最小注" if chaos.get("active") and LOW_CONFIDENCE_MINBET else "信心分級",
        "pattern_label": road.get("label", ""),
        "chaos_label": chaos.get("label", ""),
        "chaos_score": chaos.get("score", 0),
        "reason": " / ".join(reason_parts),
        "dragon": {
            "current_streak": _streak(non_tie),
            "runs_tail": run_data[-10:],
            "road_strength": round(float(road.get("strength", 0)), 3),
            "breakout": road.get("breakout"),
            "road_action": road.get("road_action", ""),
        },
        "chaos": chaos,
        "effective_weights": {k: round(v, 4) for k, v in weights.items()},
        "ai_used": bool(ai_result and not ai_result.get("error")),
        "ai_result": ai_result if os.getenv("DEBUG_AI_RESULT", "0") == "1" else None,
        "debug": feature_payload if os.getenv("DEBUG_PREDICTOR", "0") == "1" else None,
    }
