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


# First-side Dragon Mode:
# Handles the first Banker/Player dragon in a shoe. If this side has not shown
# a same-side dragon earlier, do not treat the absence of history as evidence
# against continuation.
FIRST_SIDE_DRAGON_MODE = os.getenv("FIRST_SIDE_DRAGON_MODE", "1") == "1"
FIRST_SIDE_DRAGON_MIN_LEN = int(os.getenv("FIRST_SIDE_DRAGON_MIN_LEN", "4"))
FIRST_SIDE_DRAGON_BASELINE = int(os.getenv("FIRST_SIDE_DRAGON_BASELINE", "3"))
FIRST_SIDE_DRAGON_PROTECT_STEPS = int(os.getenv("FIRST_SIDE_DRAGON_PROTECT_STEPS", "2"))
FIRST_SIDE_DRAGON_EDGE = float(os.getenv("FIRST_SIDE_DRAGON_EDGE", "0.034"))
FIRST_SIDE_DRAGON_MAX_EDGE = float(os.getenv("FIRST_SIDE_DRAGON_MAX_EDGE", "0.055"))
SIDE_AWARE_BREAKOUT = os.getenv("SIDE_AWARE_BREAKOUT", "1") == "1"

# Dragon reversal / turning-point controls.
# This layer asks whether an active dragon is likely entering a turning zone.
# It uses length fatigue, repeated same-side cut lengths, imbalance, and prior
# max length instead of blindly following every long dragon.
DRAGON_REVERSAL_MODE = os.getenv("DRAGON_REVERSAL_MODE", "1") == "1"
REVERSAL_MIN_LEN = int(os.getenv("REVERSAL_MIN_LEN", "6"))
REVERSAL_FATIGUE_LEN = int(os.getenv("REVERSAL_FATIGUE_LEN", "8"))
REVERSAL_TRIGGER = float(os.getenv("REVERSAL_TRIGGER", "0.46"))
REVERSAL_STRONG_TRIGGER = float(os.getenv("REVERSAL_STRONG_TRIGGER", "0.62"))
REVERSAL_REPEAT_NEAR_MIN = int(os.getenv("REVERSAL_REPEAT_NEAR_MIN", "2"))
REVERSAL_OVER_MAX_LEN = int(os.getenv("REVERSAL_OVER_MAX_LEN", "2"))
REVERSAL_IMBALANCE_TRIGGER = float(os.getenv("REVERSAL_IMBALANCE_TRIGGER", "0.62"))
REVERSAL_EDGE = float(os.getenv("REVERSAL_EDGE", "0.040"))
REVERSAL_HARD_EDGE = float(os.getenv("REVERSAL_HARD_EDGE", "0.070"))
REVERSAL_PROTECT_FIRST_STEPS = int(os.getenv("REVERSAL_PROTECT_FIRST_STEPS", "1"))
REVERSAL_FINAL_OVERRIDE = os.getenv("REVERSAL_FINAL_OVERRIDE", "1") == "1"
REVERSAL_OVERRIDE_EDGE = float(os.getenv("REVERSAL_OVERRIDE_EDGE", "0.058"))

# Single-chop to dragon transition controls.
# Handles shoes that start as short single-jump / alternating road, then suddenly
# stop jumping and begin connecting one side into a dragon. This prevents the
# model from continuing to force chop after the first 2~3 connected hands.
CHOP_TO_DRAGON_MODE = os.getenv("CHOP_TO_DRAGON_MODE", "1") == "1"
CHOP_TO_DRAGON_WINDOW = int(os.getenv("CHOP_TO_DRAGON_WINDOW", "12"))
CHOP_TO_DRAGON_PRE_MIN_RATE = float(os.getenv("CHOP_TO_DRAGON_PRE_MIN_RATE", "0.72"))
CHOP_TO_DRAGON_MIN_RUN = int(os.getenv("CHOP_TO_DRAGON_MIN_RUN", "2"))
CHOP_TO_DRAGON_CONFIRM_RUN = int(os.getenv("CHOP_TO_DRAGON_CONFIRM_RUN", "3"))
CHOP_TO_DRAGON_PROTECT_STEPS = int(os.getenv("CHOP_TO_DRAGON_PROTECT_STEPS", "2"))
CHOP_TO_DRAGON_EDGE = float(os.getenv("CHOP_TO_DRAGON_EDGE", "0.038"))
CHOP_TO_DRAGON_MAX_EDGE = float(os.getenv("CHOP_TO_DRAGON_MAX_EDGE", "0.068"))
CHOP_TO_DRAGON_STRENGTH = float(os.getenv("CHOP_TO_DRAGON_STRENGTH", "0.195"))
CHOP_TO_DRAGON_CHAOS_RELIEF = float(os.getenv("CHOP_TO_DRAGON_CHAOS_RELIEF", "0.10"))
CHOP_TO_DRAGON_FINAL_OVERRIDE = os.getenv("CHOP_TO_DRAGON_FINAL_OVERRIDE", "1") == "1"
CHOP_TO_DRAGON_OVERRIDE_EDGE = float(os.getenv("CHOP_TO_DRAGON_OVERRIDE_EDGE", "0.040"))

# Mirror Run Mode:
# Handles short mirror-length road behavior, e.g. B B B -> P P (expect P to fill to 3),
# then after P P P is completed, prepare for B instead of blindly chasing P4.
MIRROR_RUN_MODE = os.getenv("MIRROR_RUN_MODE", "1") == "1"
MIRROR_RUN_LOOKBACK = int(os.getenv("MIRROR_RUN_LOOKBACK", "8"))
MIRROR_RUN_MIN_LEN = int(os.getenv("MIRROR_RUN_MIN_LEN", "2"))
MIRROR_RUN_CURRENT_MIN = int(os.getenv("MIRROR_RUN_CURRENT_MIN", "2"))
MIRROR_RUN_MAX_TARGET = int(os.getenv("MIRROR_RUN_MAX_TARGET", "5"))
MIRROR_RUN_MATCH_TOLERANCE = int(os.getenv("MIRROR_RUN_MATCH_TOLERANCE", "0"))
MIRROR_RUN_EDGE = float(os.getenv("MIRROR_RUN_EDGE", "0.036"))
MIRROR_RUN_MAX_EDGE = float(os.getenv("MIRROR_RUN_MAX_EDGE", "0.060"))
MIRROR_RUN_STRENGTH = float(os.getenv("MIRROR_RUN_STRENGTH", "0.185"))
MIRROR_RUN_COMPLETE_REVERSAL = os.getenv("MIRROR_RUN_COMPLETE_REVERSAL", "1") == "1"
MIRROR_RUN_REVERSAL_EDGE = float(os.getenv("MIRROR_RUN_REVERSAL_EDGE", "0.042"))
MIRROR_RUN_REVERSAL_MAX_EDGE = float(os.getenv("MIRROR_RUN_REVERSAL_MAX_EDGE", "0.066"))
MIRROR_RUN_FINAL_OVERRIDE = os.getenv("MIRROR_RUN_FINAL_OVERRIDE", "1") == "1"
MIRROR_RUN_OVERRIDE_EDGE = float(os.getenv("MIRROR_RUN_OVERRIDE_EDGE", "0.042"))
MIRROR_RUN_CHAOS_RELIEF = float(os.getenv("MIRROR_RUN_CHAOS_RELIEF", "0.085"))


# Majority Chase Guard:
# Handles shoes where Banker/Player counts are very imbalanced and the model
# keeps chasing the already-dominant side. It does NOT blindly force the minority;
# it first checks whether the dominant side is truly still in a valid dragon, or
# whether it is overheated and the minority side has started to revive.
MAJORITY_CHASE_GUARD = os.getenv("MAJORITY_CHASE_GUARD", "1") == "1"
MAJORITY_MIN_HISTORY = int(os.getenv("MAJORITY_MIN_HISTORY", "14"))
MAJORITY_IMBALANCE_TRIGGER = float(os.getenv("MAJORITY_IMBALANCE_TRIGGER", "0.62"))
MAJORITY_SIDE_RATE_WINDOW = int(os.getenv("MAJORITY_SIDE_RATE_WINDOW", "32"))
MAJORITY_RECENT_WINDOW = int(os.getenv("MAJORITY_RECENT_WINDOW", "14"))
MAJORITY_RECENT_STRENGTH_RATE = float(os.getenv("MAJORITY_RECENT_STRENGTH_RATE", "0.64"))
MAJORITY_CHASE_DAMPEN = float(os.getenv("MAJORITY_CHASE_DAMPEN", "0.034"))
MAJORITY_OVERHEAT_LEN = int(os.getenv("MAJORITY_OVERHEAT_LEN", "4"))
MAJORITY_VALID_DRAGON_LEN = int(os.getenv("MAJORITY_VALID_DRAGON_LEN", "5"))
MAJORITY_VALID_DRAGON_PROTECT = float(os.getenv("MAJORITY_VALID_DRAGON_PROTECT", "0.45"))
MINORITY_REVIVE_LEN = int(os.getenv("MINORITY_REVIVE_LEN", "2"))
MINORITY_REVIVE_EDGE = float(os.getenv("MINORITY_REVIVE_EDGE", "0.030"))
MAJORITY_REVERSAL_TRIGGER = float(os.getenv("MAJORITY_REVERSAL_TRIGGER", "0.56"))
MAJORITY_FORCE_REVERSAL_SCORE = float(os.getenv("MAJORITY_FORCE_REVERSAL_SCORE", "0.64"))
MAJORITY_MAX_EDGE = float(os.getenv("MAJORITY_MAX_EDGE", "0.052"))
MAJORITY_CONF_CAP = float(os.getenv("MAJORITY_CONF_CAP", "0.48"))
MAJORITY_STRONG_CONF_CAP = float(os.getenv("MAJORITY_STRONG_CONF_CAP", "0.42"))
MAJORITY_FINAL_OVERRIDE = os.getenv("MAJORITY_FINAL_OVERRIDE", "1") == "1"

# Global Reversal Mode:
# Final reversal radar for ordinary road states, complex roads, and after-tie turns.
# This is not limited to long dragons. It checks whether the current final pick is
# still chasing the old direction while recent momentum, minority revival, chaos,
# mirror/chop transition, or after-tie behavior already points to the other side.
GLOBAL_REVERSAL_MODE = os.getenv("GLOBAL_REVERSAL_MODE", "1") == "1"
GLOBAL_REVERSAL_MIN_HISTORY = int(os.getenv("GLOBAL_REVERSAL_MIN_HISTORY", "10"))
GLOBAL_REVERSAL_WINDOW = int(os.getenv("GLOBAL_REVERSAL_WINDOW", "10"))
GLOBAL_REVERSAL_TRIGGER = float(os.getenv("GLOBAL_REVERSAL_TRIGGER", "0.50"))
GLOBAL_REVERSAL_STRONG_TRIGGER = float(os.getenv("GLOBAL_REVERSAL_STRONG_TRIGGER", "0.64"))
GLOBAL_REVERSAL_EDGE = float(os.getenv("GLOBAL_REVERSAL_EDGE", "0.034"))
GLOBAL_REVERSAL_HARD_EDGE = float(os.getenv("GLOBAL_REVERSAL_HARD_EDGE", "0.060"))
GLOBAL_REVERSAL_FINAL_OVERRIDE = os.getenv("GLOBAL_REVERSAL_FINAL_OVERRIDE", "1") == "1"
GLOBAL_REVERSAL_CONF_CAP = float(os.getenv("GLOBAL_REVERSAL_CONF_CAP", "0.44"))
GLOBAL_REVERSAL_STRONG_CONF_CAP = float(os.getenv("GLOBAL_REVERSAL_STRONG_CONF_CAP", "0.38"))
GLOBAL_REVERSAL_VALID_DRAGON_PROTECT = float(os.getenv("GLOBAL_REVERSAL_VALID_DRAGON_PROTECT", "0.55"))
GLOBAL_REVERSAL_RECENT_SHIFT = float(os.getenv("GLOBAL_REVERSAL_RECENT_SHIFT", "0.58"))
GLOBAL_REVERSAL_MIN_TARGET_SCORE = float(os.getenv("GLOBAL_REVERSAL_MIN_TARGET_SCORE", "0.26"))

# Momentum shift detector.
# Looks for the road changing from one side dominance to the opposite side in the
# last few non-tie results, even when no clear long dragon exists.
MOMENTUM_SHIFT_MODE = os.getenv("MOMENTUM_SHIFT_MODE", "1") == "1"
MOMENTUM_SHIFT_WINDOW = int(os.getenv("MOMENTUM_SHIFT_WINDOW", "8"))
MOMENTUM_SHIFT_TRIGGER = float(os.getenv("MOMENTUM_SHIFT_TRIGGER", "0.58"))
MOMENTUM_SHIFT_EDGE = float(os.getenv("MOMENTUM_SHIFT_EDGE", "0.030"))

# After-tie reversal assistant.
# Tie is not recommended directly by default, but a tie often makes the next
# 1~2 hands less suitable for blindly chasing the old side.
AFTER_TIE_REVERSAL_MODE = os.getenv("AFTER_TIE_REVERSAL_MODE", "1") == "1"
AFTER_TIE_WINDOW = int(os.getenv("AFTER_TIE_WINDOW", "2"))
AFTER_TIE_CHASE_DAMPEN = float(os.getenv("AFTER_TIE_CHASE_DAMPEN", "0.030"))
AFTER_TIE_REVERSAL_EDGE = float(os.getenv("AFTER_TIE_REVERSAL_EDGE", "0.026"))
AFTER_TIE_SCORE_BONUS = float(os.getenv("AFTER_TIE_SCORE_BONUS", "0.080"))


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
    if SIDE_AWARE_BREAKOUT:
        # Important: Banker and Player dragons must be judged separately.
        # If Player never had a long dragon, do not use Banker's max dragon length
        # to suppress the first Player dragon, and vice versa.
        baseline = max_same if max_same > 0 else FIRST_SIDE_DRAGON_BASELINE
    else:
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



def _first_side_dragon_context(non_tie: List[str], side: str, n: int) -> Dict[str, Any]:
    """
    First-side Dragon Mode.

    Some shoes do not show a same-side dragon early. If the current side reaches
    4+ for the first time, the old model may be too slow because it reads
    "no same-side history" as "this side does not pull dragons." This context
    treats the first same-side dragon as a possible new road state and protects
    the first 1~2 continuation hands.
    """
    if not FIRST_SIDE_DRAGON_MODE or not side or n < FIRST_SIDE_DRAGON_MIN_LEN:
        return {"active": False}

    runs = _runs(non_tie)
    if len(runs) < 3:
        return {"active": False}

    completed = runs[:-1]
    recent = completed[-DRAGON_MEMORY_LOOKBACK:] if DRAGON_MEMORY_LOOKBACK > 0 else completed
    same_side_lengths = [length for s, length in recent if s == side]
    opp_side_lengths = [length for s, length in recent if s != side]

    max_same = max(same_side_lengths) if same_side_lengths else 0
    max_opp = max(opp_side_lengths) if opp_side_lengths else 0
    same_long_count = sum(1 for length in same_side_lengths if length >= FIRST_SIDE_DRAGON_BASELINE)

    # This side has not really shown a dragon yet, but the current run is now forming one.
    first_side = same_long_count == 0 or max_same < FIRST_SIDE_DRAGON_BASELINE
    if not first_side:
        return {"active": False, "max_same": max_same, "same_long_count": same_long_count}

    # Protect the first side dragon only around its starting zone. If it becomes very long,
    # reversal mode should be allowed to take over.
    over = max(0, n - FIRST_SIDE_DRAGON_MIN_LEN)
    if over <= FIRST_SIDE_DRAGON_PROTECT_STEPS:
        edge = min(FIRST_SIDE_DRAGON_MAX_EDGE, FIRST_SIDE_DRAGON_EDGE + over * 0.008)
        phase = "first_side_protect"
        label = f"首見{side}龍{n}｜同邊首次成龍續龍保護"
        strength_bonus = 0.038
    else:
        edge = max(0.006, FIRST_SIDE_DRAGON_EDGE * 0.45 - (over - FIRST_SIDE_DRAGON_PROTECT_STEPS) * 0.006)
        phase = "first_side_extend"
        label = f"首見{side}龍{n}｜首見龍延伸降信心"
        strength_bonus = 0.018

    return {
        "active": True,
        "phase": phase,
        "side": side,
        "length": n,
        "over": over,
        "max_same": max_same,
        "max_opp": max_opp,
        "same_long_count": same_long_count,
        "cont_adjust": round(edge, 5),
        "strength_bonus": strength_bonus,
        "label": label,
    }


def _dragon_reversal_context(
    non_tie: List[str],
    side: str,
    n: int,
    breakout: Dict[str, Any] | None = None,
    first_side: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Dragon reversal / turning-point detector.

    It does not blindly reverse every long dragon. It waits for several pieces of
    evidence: fatigue length, repeated near cut lengths, over-extension versus
    same-side historical max, heavy side imbalance, or many recent runs below the
    current length.
    """
    if not DRAGON_REVERSAL_MODE or not side or n < REVERSAL_MIN_LEN:
        return {"active": False, "score": 0.0}

    runs = _runs(non_tie)
    if len(runs) < 4:
        return {"active": False, "score": 0.0}

    completed = runs[:-1]
    recent_completed = completed[-DRAGON_MEMORY_LOOKBACK:] if DRAGON_MEMORY_LOOKBACK > 0 else completed
    recent_lengths = [length for _s, length in recent_completed]
    same_side_lengths = [length for s, length in recent_completed if s == side]
    if not recent_lengths:
        return {"active": False, "score": 0.0}

    max_same = max(same_side_lengths) if same_side_lengths else 0
    max_all = max(recent_lengths)
    med_len = median(recent_lengths) if recent_lengths else 1
    near_same_cut = sum(1 for length in same_side_lengths[-12:] if abs(length - n) <= 1)
    near_all_cut = sum(1 for length in recent_lengths[-12:] if abs(length - n) <= 1)
    below_cut_count = sum(1 for length in recent_lengths[-12:] if length < n)

    recent_tail = non_tie[-min(len(non_tie), max(12, CHAOS_WINDOW)):] if non_tie else []
    side_rate = recent_tail.count(side) / len(recent_tail) if recent_tail else 0.5

    score = 0.0
    reasons: List[str] = []

    # Long enough to start looking for a turn.
    if n >= REVERSAL_FATIGUE_LEN:
        score += 0.22 + min(0.14, (n - REVERSAL_FATIGUE_LEN) * 0.035)
        reasons.append("疲勞長度")

    # Same side used to break around this zone.
    if near_same_cut >= REVERSAL_REPEAT_NEAR_MIN:
        score += 0.22 + min(0.08, (near_same_cut - REVERSAL_REPEAT_NEAR_MIN) * 0.035)
        reasons.append("同邊近長度斷點")
    elif near_all_cut >= REVERSAL_REPEAT_NEAR_MIN + 1:
        score += 0.12
        reasons.append("全局近長度斷點")

    # Current run is over its same-side historical high by a meaningful margin.
    if max_same >= DRAGON_STRONG_LEN and n >= max_same + REVERSAL_OVER_MAX_LEN:
        score += 0.18
        reasons.append("超過同邊前高過多")

    # Most recent completed runs were shorter than current length.
    if below_cut_count >= 7 and n >= DRAGON_STRONG_LEN:
        score += 0.10
        reasons.append("近期多數未達此長度")

    # Heavy side imbalance after a long dragon increases turning risk.
    if side_rate >= REVERSAL_IMBALANCE_TRIGGER and n >= REVERSAL_FATIGUE_LEN:
        score += 0.12
        reasons.append("單邊偏重")

    # Very far above median length is also a warning, but keep it small.
    if med_len and n >= max(REVERSAL_FATIGUE_LEN, med_len + 5):
        score += 0.08
        reasons.append("遠高於中位龍長")

    breakout_phase = (breakout or {}).get("phase", "")
    first_phase = (first_side or {}).get("phase", "")

    # Do not let reversal kill a fresh breakout / first-side dragon too early.
    if breakout_phase in {"breakout_protect", "breakout_extend"} and n < REVERSAL_FATIGUE_LEN:
        score *= 0.45
        reasons.append("突破龍保護中")
    if first_phase == "first_side_protect" and int((first_side or {}).get("over", 0)) <= REVERSAL_PROTECT_FIRST_STEPS:
        score *= 0.50
        reasons.append("首見龍保護中")

    score = _clamp(score, 0.0, 1.0)
    active = score >= REVERSAL_TRIGGER
    strong = score >= REVERSAL_STRONG_TRIGGER
    if not active:
        return {
            "active": False,
            "score": round(score, 3),
            "reasons": reasons,
            "max_same": max_same,
            "max_all": max_all,
            "near_same_cut": near_same_cut,
            "side_rate": round(side_rate, 3),
        }

    edge = REVERSAL_EDGE + max(0.0, score - REVERSAL_TRIGGER) * 0.07
    if strong:
        edge += 0.012
    edge = min(REVERSAL_HARD_EDGE, edge)

    return {
        "active": True,
        "strong": strong,
        "score": round(score, 3),
        "side": side,
        "target_side": _opposite(side),
        "length": n,
        "cont_adjust": round(-edge, 5),
        "break_edge": round(edge, 5),
        "label": f"轉龍風險{side}{n}｜" + "+".join(reasons[:3]),
        "reasons": reasons,
        "max_same": max_same,
        "max_all": max_all,
        "median_len": med_len,
        "near_same_cut": near_same_cut,
        "near_all_cut": near_all_cut,
        "below_cut_count": below_cut_count,
        "side_rate": round(side_rate, 3),
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

    first_side = _first_side_dragon_context(non_tie, last, n)
    breakout = _breakout_dragon_context(non_tie, last, n)
    first_phase = first_side.get("phase", "") if first_side.get("active") else ""
    breakout_phase = breakout.get("phase", "") if breakout.get("active") else ""

    # First-side Dragon Mode: when the same side never formed a dragon before,
    # protect the early forming dragon instead of reading missing history as a no-dragon signal.
    if first_side.get("active"):
        cont_prob += float(first_side.get("cont_adjust", 0.0))

    # Breakout Dragon Mode: if this dragon just exceeded history, protect continuation first.
    if breakout.get("active"):
        cont_prob += float(breakout.get("cont_adjust", 0.0))

    reversal = _dragon_reversal_context(non_tie, last, n, breakout=breakout, first_side=first_side)

    # If many recent runs ended exactly at this length, add break pressure.
    # During protected first-side / breakout stages, reduce this penalty so the model does not break too early.
    if same_cut_count >= DRAGON_BREAK_REPEAT_MIN and n >= DRAGON_MIN_LEN:
        penalty = min(0.055, same_cut_count * 0.014)
        if breakout_phase in {"breakout_protect", "breakout_extend"}:
            penalty *= 0.35
        if first_phase == "first_side_protect":
            penalty *= 0.45
        cont_prob -= penalty

    # If many recent runs could not reach this length, very long dragons should be chased with lower confidence.
    # But do not punish the first breakout / first-side hands too heavily.
    if n >= DRAGON_FATIGUE_START and below_cut_count >= 4:
        fatigue_penalty = 0.030
        if breakout_phase in {"breakout_protect", "breakout_extend"}:
            fatigue_penalty *= 0.40
        if first_phase == "first_side_protect":
            fatigue_penalty *= 0.50
        cont_prob -= fatigue_penalty

    # Reversal layer is applied after protections so it can override only when real turning evidence appears.
    if reversal.get("active"):
        cont_prob += float(reversal.get("cont_adjust", 0.0))
        # Strong reversal evidence should be allowed to actually turn the model,
        # otherwise the system only warns but still follows too late.
        if (
            reversal.get("strong")
            and n >= REVERSAL_FATIGUE_LEN
            and first_phase != "first_side_protect"
            and breakout_phase != "breakout_protect"
        ):
            forced_break_gap = min(0.040, float(reversal.get("break_edge", REVERSAL_EDGE)) * 0.55)
            cont_prob = min(cont_prob, 0.5 - forced_break_gap)

    # If current dragon is above same-side historical max but not in protected breakout / first-side mode,
    # avoid hard reverse unless reversal mode has enough evidence.
    if max_same and n > max_same and not breakout.get("active") and not first_side.get("active") and not reversal.get("active"):
        cont_prob = max(cont_prob, 0.515)

    cont_prob = _clamp(cont_prob, 0.385, 0.648)
    side = last if cont_prob >= 0.5 else _opposite(last)
    prob = 0.5 + min(DRAGON_MAX_EDGE, abs(cont_prob - 0.5))
    if cont_prob < 0.5:
        # Reversal can use a slightly stronger break edge, but still capped.
        dynamic_break_edge = DRAGON_BREAK_EDGE
        if reversal.get("active"):
            dynamic_break_edge = min(max(DRAGON_BREAK_EDGE, float(reversal.get("break_edge", DRAGON_BREAK_EDGE))), REVERSAL_HARD_EDGE)
        prob = 0.5 + min(dynamic_break_edge, abs(cont_prob - 0.5))
    b, p = _bp_score(side, prob)

    if reversal.get("active") and side != last:
        label = str(reversal.get("label", f"轉龍風險{last}{n}"))
        strength = 0.18 + (0.045 if reversal.get("strong") else 0.025)
    elif first_side.get("active"):
        label = str(first_side.get("label", f"首見{last}龍{n}"))
        strength = 0.17 + float(first_side.get("strength_bonus", 0.0))
        if reversal.get("active"):
            label += f"｜{reversal.get('label', '轉龍風險')}"
    elif breakout.get("active"):
        label = str(breakout.get("label", f"突破龍{last}{n}"))
        strength = 0.18 + float(breakout.get("strength_bonus", 0.0))
        if side != last:
            label += "｜轉斷"
        elif reversal.get("active"):
            label += f"｜{reversal.get('label', '轉龍風險')}"
    elif n >= DRAGON_STRONG_LEN:
        label = f"長龍{last}{n}｜{'續龍' if side == last else '斷龍壓力'}"
        strength = 0.18 + min(0.06, (n - DRAGON_STRONG_LEN) * 0.012)
        if reversal.get("active"):
            label += f"｜{reversal.get('label', '轉龍風險')}"
    elif n >= DRAGON_MIN_LEN:
        label = f"中龍{last}{n}｜{'續龍' if side == last else '斷龍壓力'}"
        strength = 0.145
    else:
        label = f"短龍{last}{n}｜{'續龍' if side == last else '斷龍壓力'}"
        strength = 0.105

    strength *= 0.85 + min(0.25, sample * 0.03)
    if first_side.get("active"):
        strength += min(0.030, float(first_side.get("over", 0)) * 0.008)
    if breakout.get("active"):
        strength += min(0.035, float(breakout.get("over", 0)) * 0.010)
    if reversal.get("active") and side != last:
        strength += min(0.025, float(reversal.get("score", 0)) * 0.025)

    action = "續龍" if side == last else "斷龍/轉邊"
    return {
        "B": b,
        "P": p,
        "label": label,
        "strength": _clamp(strength, 0.05, 0.32),
        "streak": n,
        "dragon_side": last,
        "cont_prob": round(cont_prob, 4),
        "road_action": action,
        "first_side": first_side,
        "breakout": breakout,
        "reversal": reversal,
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


def _chop_to_dragon_score(non_tie: List[str]) -> Dict[str, Any]:
    """
    Detect single-jump / short-road reversal into a dragon.

    Pattern idea:
    - Previous tail was highly alternating: B P B P B P ...
    - The newest side suddenly repeats: ... B P B P P or ... P B P B B
    - If that repeat reaches 2~3 hands, stop forcing chop and begin treating it
      as a possible connected dragon / road reversal.

    This layer is intentionally side-following, not opposite-following.
    """
    if not CHOP_TO_DRAGON_MODE or len(non_tie) < 7:
        return {"B": 0.5, "P": 0.5, "label": "單跳轉龍資料不足", "strength": 0.0, "active": False}

    current_side, current_len = _streak(non_tie)
    if not current_side or current_len < CHOP_TO_DRAGON_MIN_RUN:
        return {"B": 0.5, "P": 0.5, "label": "未形成單跳轉龍", "strength": 0.0, "active": False}

    before_current = non_tie[:-current_len]
    if len(before_current) < 5:
        return {"B": 0.5, "P": 0.5, "label": "單跳轉龍前段不足", "strength": 0.0, "active": False}

    pre = before_current[-CHOP_TO_DRAGON_WINDOW:]
    pre_rate = _window_switch_rate(pre)
    pre_runs = _runs(pre)
    pre_lengths = [n for _s, n in pre_runs]
    one_run_rate = _safe_div(sum(1 for n in pre_lengths if n == 1), len(pre_lengths), 0.0)

    # The side immediately before the new connected run should usually be opposite.
    # This confirms it was a jump road that just failed by repeating current_side.
    prev_side = before_current[-1] if before_current else ""
    broke_single_jump = prev_side == _opposite(current_side)

    # Tail check catches cases like B P B P B P P / P B P B P B B.
    last_before = before_current[-min(6, len(before_current)):]
    tail_rate = _window_switch_rate(last_before)

    if not broke_single_jump:
        return {
            "B": 0.5,
            "P": 0.5,
            "label": "未破單跳",
            "strength": 0.0,
            "active": False,
            "pre_switch_rate": round(pre_rate, 3),
            "tail_switch_rate": round(tail_rate, 3),
        }

    if max(pre_rate, tail_rate) < CHOP_TO_DRAGON_PRE_MIN_RATE and one_run_rate < 0.58:
        return {
            "B": 0.5,
            "P": 0.5,
            "label": "前段非單跳短路",
            "strength": 0.0,
            "active": False,
            "pre_switch_rate": round(pre_rate, 3),
            "tail_switch_rate": round(tail_rate, 3),
            "one_run_rate": round(one_run_rate, 3),
        }

    # Phase control: 2 hands is early reversal, 3+ hands is confirmed connection.
    over = max(0, current_len - CHOP_TO_DRAGON_MIN_RUN)
    if current_len >= CHOP_TO_DRAGON_CONFIRM_RUN:
        phase = "confirmed"
        label = f"單跳反轉接龍{current_side}{current_len}｜跳路破壞後同邊延伸"
        base_strength = CHOP_TO_DRAGON_STRENGTH + 0.025
    else:
        phase = "early"
        label = f"單跳反轉觀察{current_side}{current_len}｜短跳破壞疑似接龍"
        base_strength = CHOP_TO_DRAGON_STRENGTH * 0.82

    edge = CHOP_TO_DRAGON_EDGE + over * 0.010
    if tail_rate >= 0.82:
        edge += 0.006
    if one_run_rate >= 0.70:
        edge += 0.006
    edge = min(CHOP_TO_DRAGON_MAX_EDGE, edge)

    # If the connected side already got too long, hand over to dragon reversal layer.
    # Do not keep this layer blindly following beyond the protected zone.
    if current_len >= max(REVERSAL_MIN_LEN, CHOP_TO_DRAGON_CONFIRM_RUN + CHOP_TO_DRAGON_PROTECT_STEPS + 2):
        edge *= 0.70
        base_strength *= 0.82
        label += "｜後續交給轉龍判斷"

    b, p = _bp_score(current_side, 0.5 + edge)
    return {
        "B": b,
        "P": p,
        "label": label,
        "strength": _clamp(base_strength, 0.08, 0.245),
        "active": True,
        "phase": phase,
        "target_side": current_side,
        "current_len": current_len,
        "edge": round(edge, 5),
        "road_action": "單跳反轉接龍/續龍",
        "pre_switch_rate": round(pre_rate, 3),
        "tail_switch_rate": round(tail_rate, 3),
        "one_run_rate": round(one_run_rate, 3),
        "protect_zone": current_len <= CHOP_TO_DRAGON_CONFIRM_RUN + CHOP_TO_DRAGON_PROTECT_STEPS,
    }



def _mirror_run_score(non_tie: List[str]) -> Dict[str, Any]:
    """
    Mirror Run Mode / 對稱龍長模式.

    This catches the road type the user described:
        previous run: B3
        current run:  P2 -> expect P to fill one more hand to P3
        current run:  P3 -> mirror length completed, prepare B instead of chasing P4 blindly

    It looks only at run lengths, not card points. Tie is ignored upstream.
    """
    if not MIRROR_RUN_MODE or len(non_tie) < 4:
        return {"B": 0.5, "P": 0.5, "label": "對稱龍資料不足", "strength": 0.0, "active": False}

    runs = _runs(non_tie)
    if len(runs) < 2:
        return {"B": 0.5, "P": 0.5, "label": "對稱龍資料不足", "strength": 0.0, "active": False}

    current_side, current_len = runs[-1]
    prev_side, prev_len = runs[-2]

    if prev_side == current_side:
        return {"B": 0.5, "P": 0.5, "label": "非對稱轉邊", "strength": 0.0, "active": False}

    if prev_len < MIRROR_RUN_MIN_LEN or prev_len > MIRROR_RUN_MAX_TARGET:
        return {
            "B": 0.5,
            "P": 0.5,
            "label": "前段龍長不適合對稱",
            "strength": 0.0,
            "active": False,
            "prev_len": prev_len,
            "current_len": current_len,
        }

    if current_len < MIRROR_RUN_CURRENT_MIN:
        return {
            "B": 0.5,
            "P": 0.5,
            "label": "對稱龍尚未啟動",
            "strength": 0.0,
            "active": False,
            "prev_len": prev_len,
            "current_len": current_len,
        }

    recent_completed = runs[:-1]
    recent_lengths = [n for _s, n in recent_completed[-MIRROR_RUN_LOOKBACK:]] if MIRROR_RUN_LOOKBACK > 0 else [n for _s, n in recent_completed]
    near_prev_hits = sum(1 for n in recent_lengths if abs(n - prev_len) <= max(0, MIRROR_RUN_MATCH_TOLERANCE))
    short_mirror_context = prev_len <= 4 and current_len <= prev_len + max(0, MIRROR_RUN_MATCH_TOLERANCE)

    # Phase 1: fill to the previous run length.
    # Example: B3 -> P2, target P to complete P3.
    fill_threshold = prev_len - max(0, MIRROR_RUN_MATCH_TOLERANCE)
    if current_len < fill_threshold:
        target_side = current_side
        missing = max(1, prev_len - current_len)
        edge = MIRROR_RUN_EDGE + min(0.016, current_len * 0.006) + min(0.010, near_prev_hits * 0.003)
        edge = min(MIRROR_RUN_MAX_EDGE, edge)
        b, p = _bp_score(target_side, 0.5 + edge)
        return {
            "B": b,
            "P": p,
            "label": f"對稱補龍{current_side}{current_len}→{prev_len}｜承接前段{prev_side}{prev_len}",
            "strength": _clamp(MIRROR_RUN_STRENGTH + min(0.025, near_prev_hits * 0.006), 0.08, 0.235),
            "active": True,
            "phase": "fill",
            "target_side": target_side,
            "prev_side": prev_side,
            "prev_len": prev_len,
            "current_side": current_side,
            "current_len": current_len,
            "missing": missing,
            "edge": round(edge, 5),
            "near_prev_hits": near_prev_hits,
            "road_action": "對稱補龍/續到前段長度",
        }

    # Phase 2: mirror length completed; prepare reversal.
    # Example: B3 -> P3, target B instead of blindly following P4.
    if MIRROR_RUN_COMPLETE_REVERSAL and short_mirror_context:
        target_side = _opposite(current_side)
        edge = MIRROR_RUN_REVERSAL_EDGE + min(0.014, near_prev_hits * 0.004)
        # If the current side is also a first-side/breakout dragon, do not over-punish.
        if current_len >= DRAGON_STRONG_LEN + 1:
            edge *= 0.75
        edge = min(MIRROR_RUN_REVERSAL_MAX_EDGE, edge)
        b, p = _bp_score(target_side, 0.5 + edge)
        return {
            "B": b,
            "P": p,
            "label": f"對稱補滿轉邊{current_side}{current_len}≈前段{prev_side}{prev_len}｜補滿觀察{target_side}",
            "strength": _clamp(MIRROR_RUN_STRENGTH + 0.028 + min(0.025, near_prev_hits * 0.006), 0.10, 0.255),
            "active": True,
            "phase": "complete_reversal",
            "target_side": target_side,
            "prev_side": prev_side,
            "prev_len": prev_len,
            "current_side": current_side,
            "current_len": current_len,
            "edge": round(edge, 5),
            "near_prev_hits": near_prev_hits,
            "road_action": "對稱補滿/轉邊",
        }

    # If current run already exceeds the mirror target, hand it back to dragon / breakout logic.
    return {
        "B": 0.5,
        "P": 0.5,
        "label": "對稱已超長交回龍判斷",
        "strength": 0.0,
        "active": False,
        "prev_side": prev_side,
        "prev_len": prev_len,
        "current_side": current_side,
        "current_len": current_len,
    }

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

    # 7.5) If the single-jump break is turning into a connected dragon, do not
    # treat it as pure chaos. This is the exact transition pattern the model
    # previously missed: short chop -> repeat -> dragon continuation.
    chop_to_dragon = _chop_to_dragon_score(non_tie)
    if chop_to_dragon.get("active"):
        metrics["chop_to_dragon"] = {
            "phase": chop_to_dragon.get("phase"),
            "target_side": chop_to_dragon.get("target_side"),
            "current_len": chop_to_dragon.get("current_len"),
            "pre_switch_rate": chop_to_dragon.get("pre_switch_rate"),
            "tail_switch_rate": chop_to_dragon.get("tail_switch_rate"),
        }
        relief = CHOP_TO_DRAGON_CHAOS_RELIEF if chop_to_dragon.get("protect_zone") else CHOP_TO_DRAGON_CHAOS_RELIEF * 0.45
        score -= relief
        reasons.append("單跳反轉接龍")

    # 7.6) Mirror-run behavior is not pure chaos either: it is a short-run length
    # completion pattern. Reduce chaos so the mirror layer can guide fill/reversal.
    mirror_run = _mirror_run_score(non_tie)
    if mirror_run.get("active"):
        metrics["mirror_run"] = {
            "phase": mirror_run.get("phase"),
            "target_side": mirror_run.get("target_side"),
            "prev_len": mirror_run.get("prev_len"),
            "current_len": mirror_run.get("current_len"),
        }
        score -= MIRROR_RUN_CHAOS_RELIEF
        reasons.append("對稱補龍/補滿轉邊")

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
        "chop_to_dragon_active": bool(locals().get("chop_to_dragon", {}).get("active")),
        "chop_to_dragon": locals().get("chop_to_dragon", None),
        "mirror_run_active": bool(locals().get("mirror_run", {}).get("active")),
        "mirror_run": locals().get("mirror_run", None),
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
        # When chaos is caused by a clean single-jump break into a connected run,
        # keep enough road/streak weight to catch the new dragon instead of
        # flattening everything into weak chaos mode.
        if chaos.get("chop_to_dragon_active"):
            road_factor = max(road_factor, 0.72)
            streak_factor = max(streak_factor, 0.66)
            recent_factor = max(recent_factor, 1.38)
            markov_factor = max(markov_factor, 1.12)
        if chaos.get("mirror_run_active"):
            road_factor = max(road_factor, 0.70)
            streak_factor = max(streak_factor, 0.64)
            recent_factor = max(recent_factor, 1.32)
            markov_factor = max(markov_factor, 1.10)
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
        _mirror_run_score(non_tie),
        _chop_score(non_tie),
        _chop_to_dragon_score(non_tie),
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
    if best.get("active") and "單跳反轉" in str(best.get("label", "")):
        best["chop_to_dragon"] = dict(best)
    if best.get("active") and "對稱" in str(best.get("label", "")):
        best["mirror_run"] = dict(best)
    if second and second.get("strength", 0) >= 0.12 and second.get("label") != best.get("label"):
        best["secondary_label"] = second.get("label")
        # Small blend to avoid one pattern totally dominating another valid road mode.
        # Exception: if dragon reversal is the best mode, do not let pattern-memory
        # pull the score back to the old dragon side and make the action inconsistent.
        blend = 0.0 if best.get("road_action") == "斷龍/轉邊" else 0.25
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



def _majority_chase_guard(
    non_tie: List[str],
    b_prob: float,
    p_prob: float,
    tie_prob: float,
    road: Dict[str, Any],
    chaos: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Majority Chase Guard / 多數邊追擊保護.

    Problem this solves:
    If Banker or Player count becomes very uneven, road/streak/recent layers can
    keep chasing the already-dominant side. This guard does not blindly bet the
    minority side. Instead, it checks whether the majority side is still a valid
    strong dragon. If not, it dampens majority chasing; if the minority has begun
    to connect or reversal evidence is strong, it can flip the final calibration.
    """
    base = {
        "active": False,
        "adjusted": False,
        "forced": False,
        "B": b_prob,
        "P": p_prob,
        "T": tie_prob,
        "label": "多數邊保護未啟動",
        "score": 0.0,
        "reasons": [],
    }
    if not MAJORITY_CHASE_GUARD or len(non_tie) < MAJORITY_MIN_HISTORY:
        return base

    side_window = MAJORITY_SIDE_RATE_WINDOW if MAJORITY_SIDE_RATE_WINDOW > 0 else len(non_tie)
    side_seq = non_tie[-min(len(non_tie), side_window):]
    if not side_seq:
        return base

    b_count = side_seq.count("B")
    p_count = side_seq.count("P")
    if b_count == p_count:
        return base

    majority = "B" if b_count > p_count else "P"
    minority = _opposite(majority)
    majority_count = max(b_count, p_count)
    side_rate = majority_count / len(side_seq)
    if side_rate < MAJORITY_IMBALANCE_TRIGGER:
        base.update({
            "majority": majority,
            "minority": minority,
            "majority_rate": round(side_rate, 3),
            "label": "莊閒顆數尚未達多數邊保護",
        })
        return base

    pred_side = "B" if b_prob >= p_prob else "P"
    current_side, current_len = _streak(non_tie)
    recent = non_tie[-min(len(non_tie), max(4, MAJORITY_RECENT_WINDOW)):]
    recent_majority_rate = recent.count(majority) / len(recent) if recent else 0.5
    recent_minority_rate = recent.count(minority) / len(recent) if recent else 0.5

    reasons: List[str] = []
    score = 0.0
    imbalance_score = _safe_div(side_rate - MAJORITY_IMBALANCE_TRIGGER, max(0.001, 1 - MAJORITY_IMBALANCE_TRIGGER), 0.0)
    score += min(0.30, imbalance_score * 0.36)
    reasons.append(f"{majority}顆數佔比{int(side_rate * 100)}%")

    if pred_side == majority:
        score += 0.12
        reasons.append("模型追多數邊")
    else:
        # Already not chasing the majority; only keep debug context.
        base.update({
            "majority": majority,
            "minority": minority,
            "majority_rate": round(side_rate, 3),
            "recent_majority_rate": round(recent_majority_rate, 3),
            "current_streak": (current_side, current_len),
            "label": f"多數邊{majority}已偏高但目前未追多數",
            "reasons": reasons,
        })
        return base

    if recent_majority_rate >= MAJORITY_RECENT_STRENGTH_RATE:
        score += 0.08
        reasons.append("近期多數邊仍偏強")

    majority_overheat = current_side == majority and current_len >= MAJORITY_OVERHEAT_LEN
    if majority_overheat:
        score += 0.16 + min(0.12, max(0, current_len - MAJORITY_OVERHEAT_LEN) * 0.035)
        reasons.append(f"多數邊{majority}{current_len}過熱")

    minority_revive = current_side == minority and current_len >= MINORITY_REVIVE_LEN
    if minority_revive:
        score += 0.22 + min(0.10, max(0, current_len - MINORITY_REVIVE_LEN) * 0.035)
        reasons.append(f"少數邊{minority}{current_len}回補啟動")

    road_label = str(road.get("label", "")) if isinstance(road, dict) else ""
    road_action = str(road.get("road_action", "")) if isinstance(road, dict) else ""

    road_reversal = road.get("reversal") if isinstance(road, dict) else None
    if isinstance(road_reversal, dict) and road_reversal.get("active") and road_reversal.get("target_side") == minority:
        score += 0.16 + (0.08 if road_reversal.get("strong") else 0.0)
        reasons.append("龍長反轉指向少數邊")

    road_mirror = road.get("mirror_run") if isinstance(road, dict) else None
    if isinstance(road_mirror, dict) and road_mirror.get("active") and road_mirror.get("target_side") == minority:
        score += 0.10
        reasons.append("對稱龍長指向少數邊")

    road_chop_to_dragon = road.get("chop_to_dragon") if isinstance(road, dict) else None
    if isinstance(road_chop_to_dragon, dict) and road_chop_to_dragon.get("active") and road_chop_to_dragon.get("target_side") == minority:
        score += 0.10
        reasons.append("單跳轉龍指向少數邊")

    # If majority side is a fresh, valid road state, do not punish it too much.
    valid_majority_dragon = (
        current_side == majority
        and current_len >= MAJORITY_VALID_DRAGON_LEN
        and road_action == "續龍"
        and any(key in road_label for key in ["首見", "突破", "長龍", "單跳反轉接龍"])
        and not minority_revive
        and not (isinstance(road_reversal, dict) and road_reversal.get("strong"))
    )
    if valid_majority_dragon:
        score *= MAJORITY_VALID_DRAGON_PROTECT
        reasons.append("多數邊真龍保護")

    if chaos.get("active"):
        score += 0.06 if not chaos.get("strong") else 0.10
        reasons.append("混亂盤降低追擊信心")

    score = _clamp(score, 0.0, 1.0)
    if score < MAJORITY_REVERSAL_TRIGGER:
        base.update({
            "active": False,
            "adjusted": False,
            "forced": False,
            "majority": majority,
            "minority": minority,
            "majority_rate": round(side_rate, 3),
            "recent_majority_rate": round(recent_majority_rate, 3),
            "recent_minority_rate": round(recent_minority_rate, 3),
            "current_streak": (current_side, current_len),
            "score": round(score, 3),
            "label": f"多數邊{majority}偏高｜追擊風險未達修正",
            "reasons": reasons,
        })
        return base

    bp_total = max(0.001, 1 - tie_prob)
    b_side = b_prob / bp_total
    p_side = p_prob / bp_total

    forced = MAJORITY_FINAL_OVERRIDE and (
        score >= MAJORITY_FORCE_REVERSAL_SCORE
        and (minority_revive or majority_overheat or (isinstance(road_reversal, dict) and road_reversal.get("strong")))
    )

    if forced:
        edge = min(MAJORITY_MAX_EDGE, max(0.026, MINORITY_REVIVE_EDGE + (score - MAJORITY_FORCE_REVERSAL_SCORE) * 0.055))
        if minority == "B":
            b_side, p_side = 0.5 + edge, 0.5 - edge
        else:
            b_side, p_side = 0.5 - edge, 0.5 + edge
        label = f"多數邊追擊保護｜{majority}過熱轉看{minority}"
    else:
        dampen = min(MAJORITY_MAX_EDGE, MAJORITY_CHASE_DAMPEN + max(0.0, score - MAJORITY_REVERSAL_TRIGGER) * 0.050)
        if majority == "B":
            b_side = max(0.5 + 0.002, b_side - dampen)
            p_side = 1 - b_side
        else:
            p_side = max(0.5 + 0.002, p_side - dampen)
            b_side = 1 - p_side
        label = f"多數邊追擊保護｜降低{majority}追擊信心"

    new_b = b_side * bp_total
    new_p = p_side * bp_total
    new_b, new_p, new_t = _normalize_three(new_b, new_p, tie_prob)
    return {
        "active": True,
        "adjusted": True,
        "forced": forced,
        "B": new_b,
        "P": new_p,
        "T": new_t,
        "majority": majority,
        "minority": minority,
        "majority_rate": round(side_rate, 3),
        "recent_majority_rate": round(recent_majority_rate, 3),
        "recent_minority_rate": round(recent_minority_rate, 3),
        "current_streak": (current_side, current_len),
        "score": round(score, 3),
        "label": label,
        "reasons": reasons,
        "bet_mode_hint": "最小注" if not forced else "反轉小注",
    }


def _last_tie_gap(history: List[str]) -> int:
    """Return number of rounds since the last tie. Large value means no recent tie."""
    if not history:
        return 999
    gap = 0
    for x in reversed(history):
        if x == "T":
            return gap
        gap += 1
    return 999


def _global_reversal_guard(
    non_tie: List[str],
    history: List[str],
    b_prob: float,
    p_prob: float,
    tie_prob: float,
    road: Dict[str, Any],
    chaos: Dict[str, Any],
    majority_guard: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Global Reversal Mode / 全局反轉雷達.

    Purpose:
    - Catch ordinary reversals, not only long-dragon reversals.
    - Make complex-road turns more sensitive when recent momentum has shifted.
    - Treat the 1~2 hands after a tie as a caution zone against blindly chasing.
    - Avoid killing true valid dragons too early by protecting strong, fresh dragon states.

    It only changes the final probability if evidence points to the opposite side
    of the current final pick. Otherwise it returns debug context without action.
    """
    base = {
        "active": False,
        "adjusted": False,
        "forced": False,
        "B": b_prob,
        "P": p_prob,
        "T": tie_prob,
        "label": "全局反轉未啟動",
        "score": 0.0,
        "target_side": "",
        "reasons": [],
    }
    if not GLOBAL_REVERSAL_MODE or len(non_tie) < GLOBAL_REVERSAL_MIN_HISTORY:
        return base

    bp_total = max(0.001, 1 - tie_prob)
    b_side = b_prob / bp_total
    p_side = p_prob / bp_total
    pred_side = "B" if b_side >= p_side else "P"
    opp_side = _opposite(pred_side)

    current_side, current_len = _streak(non_tie)
    recent_window = max(4, GLOBAL_REVERSAL_WINDOW)
    recent = non_tie[-min(len(non_tie), recent_window):]
    prev = non_tie[-min(len(non_tie), recent_window * 2):-min(len(non_tie), recent_window)] if len(non_tie) > recent_window else []
    runs = _runs(non_tie)

    target_scores = {"B": 0.0, "P": 0.0}
    target_reasons = {"B": [], "P": []}
    general_score = 0.0
    general_reasons: List[str] = []

    def add_target(side: str, points: float, reason: str) -> None:
        if side in {"B", "P"} and points > 0:
            target_scores[side] += points
            target_reasons[side].append(reason)

    # 1) The newest side is already connecting against the current final pick.
    if current_side in {"B", "P"} and current_side != pred_side and current_len >= 2:
        add_target(current_side, 0.20 + min(0.12, (current_len - 2) * 0.045), f"近期{current_side}{current_len}反向連接")

    # 2) Recent side-rate shift points against the current final pick.
    if recent:
        recent_b = recent.count("B") / len(recent)
        recent_p = recent.count("P") / len(recent)
        recent_side = "B" if recent_b >= recent_p else "P"
        recent_rate = max(recent_b, recent_p)
        if recent_side != pred_side and recent_rate >= GLOBAL_REVERSAL_RECENT_SHIFT:
            add_target(recent_side, 0.16 + min(0.10, (recent_rate - GLOBAL_REVERSAL_RECENT_SHIFT) * 0.45), f"近{len(recent)}手{recent_side}轉強{int(recent_rate * 100)}%")

    # 3) Momentum shifted from old dominance to the opposite side.
    if MOMENTUM_SHIFT_MODE and len(non_tie) >= max(10, MOMENTUM_SHIFT_WINDOW * 2):
        w = max(4, MOMENTUM_SHIFT_WINDOW)
        old = non_tie[-2 * w:-w]
        new = non_tie[-w:]
        if old and new:
            old_b = old.count("B") / len(old)
            old_p = old.count("P") / len(old)
            new_b = new.count("B") / len(new)
            new_p = new.count("P") / len(new)
            old_side = "B" if old_b >= old_p else "P"
            new_side = "B" if new_b >= new_p else "P"
            old_rate = max(old_b, old_p)
            new_rate = max(new_b, new_p)
            if old_side == pred_side and new_side != pred_side and new_rate >= MOMENTUM_SHIFT_TRIGGER:
                add_target(new_side, 0.18 + min(0.10, (new_rate - MOMENTUM_SHIFT_TRIGGER) * 0.45), f"動能由{old_side}轉{new_side}")
            elif new_side != pred_side and new_rate >= MOMENTUM_SHIFT_TRIGGER + 0.08:
                add_target(new_side, 0.12, f"短窗動能{new_side}偏強")

    # 4) Road sub-models already point to an opposite turn.
    road_reversal = road.get("reversal") if isinstance(road, dict) else None
    if isinstance(road_reversal, dict) and road_reversal.get("active") and road_reversal.get("target_side") in {"B", "P"}:
        target = str(road_reversal.get("target_side"))
        if target != pred_side:
            add_target(target, 0.17 + (0.08 if road_reversal.get("strong") else 0.0), "龍長反轉同步指向")

    road_mirror = road.get("mirror_run") if isinstance(road, dict) else None
    if isinstance(road_mirror, dict) and road_mirror.get("active") and road_mirror.get("target_side") in {"B", "P"}:
        target = str(road_mirror.get("target_side"))
        if target != pred_side:
            add_target(target, 0.12, "對稱補滿/補龍指向反邊")

    road_chop_to_dragon = road.get("chop_to_dragon") if isinstance(road, dict) else None
    if isinstance(road_chop_to_dragon, dict) and road_chop_to_dragon.get("active") and road_chop_to_dragon.get("target_side") in {"B", "P"}:
        target = str(road_chop_to_dragon.get("target_side"))
        if target != pred_side:
            add_target(target, 0.12, "單跳轉龍指向反邊")

    # 5) Chaos mode means the current road is not stable; become more sensitive to short-window turns.
    if chaos.get("active"):
        general_score += 0.07 if not chaos.get("strong") else 0.12
        general_reasons.append("複雜/破路盤提高反轉靈敏")
        chaos_side = "B" if float(chaos.get("B", 0.5)) >= float(chaos.get("P", 0.5)) else "P"
        chaos_gap = abs(float(chaos.get("B", 0.5)) - float(chaos.get("P", 0.5)))
        if chaos_side != pred_side and chaos_gap >= 0.012:
            add_target(chaos_side, 0.08 + min(0.06, chaos_gap * 1.5), "短窗混亂校準反向")

    # 6) Majority guard has detected overheating but may not fully flip yet.
    if isinstance(majority_guard, dict) and majority_guard.get("active"):
        maj_target = "B" if float(majority_guard.get("B", 0.5)) >= float(majority_guard.get("P", 0.5)) else "P"
        if maj_target != pred_side:
            add_target(maj_target, 0.11 + (0.06 if majority_guard.get("forced") else 0.0), "多數邊過熱校準同步")
        if majority_guard.get("adjusted"):
            general_score += 0.04
            general_reasons.append("多數邊保護已啟動")

    # 7) After a tie: do not chase old direction too hard; if a side connects after the tie,
    # treat that side as early reversal evidence.
    tie_gap = _last_tie_gap(history)
    after_tie_active = AFTER_TIE_REVERSAL_MODE and tie_gap <= AFTER_TIE_WINDOW
    if after_tie_active:
        general_score += AFTER_TIE_SCORE_BONUS
        general_reasons.append(f"和局後{tie_gap}手內降追擊")
        if current_side in {"B", "P"} and current_side != pred_side:
            add_target(current_side, 0.10 + min(0.06, current_len * 0.025), f"和局後{current_side}轉向")
        elif current_side in {"B", "P"} and current_side == pred_side and current_len >= 2:
            # Same-side after tie can still continue, but reduce overconfidence instead of flipping.
            general_score += min(0.06, AFTER_TIE_CHASE_DAMPEN * 1.7)
            general_reasons.append("和局後同邊追擊降信心")

    # 8) Run rhythm reversal: two completed runs in a row cut shorter than the model's current side.
    if len(runs) >= 4:
        last_side, last_len = runs[-1]
        prev_side, prev_len = runs[-2]
        prev2_side, prev2_len = runs[-3]
        if last_side != pred_side and last_len >= 2 and prev_side == pred_side and prev_len <= 2:
            add_target(last_side, 0.10, "短連段轉向")
        if prev2_side == pred_side and prev_len <= 2 and last_side != pred_side and last_len >= prev_len + 1:
            add_target(last_side, 0.08, "前段短切後反邊延伸")

    # Protect a true valid dragon from getting killed too early.
    road_label = str(road.get("label", "")) if isinstance(road, dict) else ""
    road_action = str(road.get("road_action", "")) if isinstance(road, dict) else ""
    valid_dragon_protect = (
        current_side == pred_side
        and current_len >= max(DRAGON_MIN_LEN, MAJORITY_VALID_DRAGON_LEN)
        and road_action == "續龍"
        and any(key in road_label for key in ["首見", "突破", "長龍", "單跳反轉接龍"])
        and not after_tie_active
        and not chaos.get("strong")
    )

    # Pick target side.
    target_side = "B" if target_scores["B"] >= target_scores["P"] else "P"
    target_score = target_scores[target_side]
    if target_side == pred_side or target_score < GLOBAL_REVERSAL_MIN_TARGET_SCORE:
        base.update({
            "active": False,
            "adjusted": False,
            "forced": False,
            "target_side": target_side if target_side != pred_side else "",
            "score": round(_clamp(general_score + target_score, 0.0, 1.0), 3),
            "label": "全局反轉證據不足",
            "reasons": general_reasons + target_reasons.get(target_side, []),
            "tie_gap": tie_gap,
            "target_scores": {k: round(v, 3) for k, v in target_scores.items()},
        })
        return base

    score = _clamp(general_score + target_score, 0.0, 1.0)
    if valid_dragon_protect:
        score *= GLOBAL_REVERSAL_VALID_DRAGON_PROTECT
        general_reasons.append("有效龍保護降低反轉")

    if score < GLOBAL_REVERSAL_TRIGGER:
        base.update({
            "active": False,
            "adjusted": False,
            "forced": False,
            "target_side": target_side,
            "score": round(score, 3),
            "label": "全局反轉未達門檻",
            "reasons": general_reasons + target_reasons.get(target_side, []),
            "tie_gap": tie_gap,
            "target_scores": {k: round(v, 3) for k, v in target_scores.items()},
        })
        return base

    forced = GLOBAL_REVERSAL_FINAL_OVERRIDE and score >= GLOBAL_REVERSAL_STRONG_TRIGGER
    edge = GLOBAL_REVERSAL_EDGE + max(0.0, score - GLOBAL_REVERSAL_TRIGGER) * 0.060
    if after_tie_active:
        edge += AFTER_TIE_REVERSAL_EDGE * 0.55
    if MOMENTUM_SHIFT_MODE and target_score >= 0.34:
        edge += MOMENTUM_SHIFT_EDGE * 0.35
    edge = min(GLOBAL_REVERSAL_HARD_EDGE, max(0.020, edge))

    if forced:
        if target_side == "B":
            b_side, p_side = 0.5 + edge, 0.5 - edge
        else:
            b_side, p_side = 0.5 - edge, 0.5 + edge
        label = f"全局反轉校準｜轉看{target_side}"
    else:
        # Non-forced mode: lower the old pick and move close to neutral / slight target.
        damp = min(edge, abs((b_side if pred_side == "B" else p_side) - 0.5) + edge * 0.45)
        if pred_side == "B":
            b_side = max(0.5 - edge * 0.20, b_side - damp)
            p_side = 1 - b_side
        else:
            p_side = max(0.5 - edge * 0.20, p_side - damp)
            b_side = 1 - p_side
        # If target evidence is clearly above trigger, allow slight flip even before strong trigger.
        if target_score >= GLOBAL_REVERSAL_MIN_TARGET_SCORE + 0.12:
            if target_side == "B":
                b_side = max(b_side, 0.5 + min(edge, GLOBAL_REVERSAL_EDGE))
                p_side = 1 - b_side
            else:
                p_side = max(p_side, 0.5 + min(edge, GLOBAL_REVERSAL_EDGE))
                b_side = 1 - p_side
        label = f"全局反轉雷達｜降低{pred_side}追擊"

    new_b = b_side * bp_total
    new_p = p_side * bp_total
    new_b, new_p, new_t = _normalize_three(new_b, new_p, tie_prob)
    return {
        "active": True,
        "adjusted": True,
        "forced": forced,
        "B": new_b,
        "P": new_p,
        "T": new_t,
        "target_side": target_side,
        "from_side": pred_side,
        "score": round(score, 3),
        "label": label,
        "reasons": general_reasons + target_reasons.get(target_side, []),
        "tie_gap": tie_gap,
        "after_tie_active": after_tie_active,
        "valid_dragon_protect": valid_dragon_protect,
        "target_scores": {k: round(v, 3) for k, v in target_scores.items()},
        "edge": round(edge, 5),
        "bet_mode_hint": "反轉小注" if forced else "最小注/觀察",
    }

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
        "global_reversal_config": {
            "enabled": GLOBAL_REVERSAL_MODE,
            "window": GLOBAL_REVERSAL_WINDOW,
            "trigger": GLOBAL_REVERSAL_TRIGGER,
            "after_tie": AFTER_TIE_REVERSAL_MODE,
        },
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

    # If the dragon reversal detector is strong, allow it to actually flip the final
    # recommendation. Without this guard, Markov/Streak often keep following the old
    # dragon even when the road layer has already detected a turning zone.
    reversal_final_override = False
    road_reversal = road.get("reversal") if isinstance(road, dict) else None
    if (
        REVERSAL_FINAL_OVERRIDE
        and isinstance(road_reversal, dict)
        and road_reversal.get("active")
        and road_reversal.get("strong")
        and road.get("road_action") == "斷龍/轉邊"
        and road_reversal.get("target_side") in {"B", "P"}
    ):
        target_side = str(road_reversal.get("target_side"))
        edge = min(REVERSAL_OVERRIDE_EDGE, max(0.026, float(road_reversal.get("break_edge", REVERSAL_EDGE)) * 0.90))
        bp_total = max(0.001, 1 - tie_prob)
        if target_side == "B":
            b_prob = (0.5 + edge) * bp_total
            p_prob = (0.5 - edge) * bp_total
        else:
            b_prob = (0.5 - edge) * bp_total
            p_prob = (0.5 + edge) * bp_total
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)
        reversal_final_override = True

    chop_to_dragon_final_override = False
    road_chop_to_dragon = road.get("chop_to_dragon") if isinstance(road, dict) else None
    if (
        CHOP_TO_DRAGON_FINAL_OVERRIDE
        and isinstance(road_chop_to_dragon, dict)
        and road_chop_to_dragon.get("active")
        and road_chop_to_dragon.get("target_side") in {"B", "P"}
        and road_chop_to_dragon.get("phase") in {"confirmed", "early"}
    ):
        target_side = str(road_chop_to_dragon.get("target_side"))
        raw_edge = float(road_chop_to_dragon.get("edge", CHOP_TO_DRAGON_EDGE))
        # Early phase should guide but not overrule too aggressively.
        factor = 0.78 if road_chop_to_dragon.get("phase") == "early" else 1.0
        edge = min(CHOP_TO_DRAGON_OVERRIDE_EDGE, max(0.022, raw_edge * factor))
        bp_total = max(0.001, 1 - tie_prob)
        if target_side == "B":
            b_prob = (0.5 + edge) * bp_total
            p_prob = (0.5 - edge) * bp_total
        else:
            b_prob = (0.5 - edge) * bp_total
            p_prob = (0.5 + edge) * bp_total
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)
        chop_to_dragon_final_override = True

    mirror_run_final_override = False
    road_mirror_run = road.get("mirror_run") if isinstance(road, dict) else None
    if (
        MIRROR_RUN_FINAL_OVERRIDE
        and isinstance(road_mirror_run, dict)
        and road_mirror_run.get("active")
        and road_mirror_run.get("target_side") in {"B", "P"}
        and road_mirror_run.get("phase") in {"fill", "complete_reversal"}
    ):
        target_side = str(road_mirror_run.get("target_side"))
        raw_edge = float(road_mirror_run.get("edge", MIRROR_RUN_EDGE))
        factor = 0.82 if road_mirror_run.get("phase") == "fill" else 1.0
        edge = min(MIRROR_RUN_OVERRIDE_EDGE, max(0.022, raw_edge * factor))
        bp_total = max(0.001, 1 - tie_prob)
        if target_side == "B":
            b_prob = (0.5 + edge) * bp_total
            p_prob = (0.5 - edge) * bp_total
        else:
            b_prob = (0.5 - edge) * bp_total
            p_prob = (0.5 + edge) * bp_total
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)
        mirror_run_final_override = True


    majority_guard = _majority_chase_guard(non_tie, b_prob, p_prob, tie_prob, road, chaos)
    if majority_guard.get("active") and majority_guard.get("adjusted"):
        b_prob = float(majority_guard.get("B", b_prob))
        p_prob = float(majority_guard.get("P", p_prob))
        tie_prob = float(majority_guard.get("T", tie_prob))
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)

    global_reversal = _global_reversal_guard(non_tie, history, b_prob, p_prob, tie_prob, road, chaos, majority_guard)
    if global_reversal.get("active") and global_reversal.get("adjusted"):
        b_prob = float(global_reversal.get("B", b_prob))
        p_prob = float(global_reversal.get("P", p_prob))
        tie_prob = float(global_reversal.get("T", tie_prob))
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
    if majority_guard.get("active") and majority_guard.get("adjusted"):
        votes.append("B" if majority_guard.get("B", 0.5) >= majority_guard.get("P", 0.5) else "P")
    if global_reversal.get("active") and global_reversal.get("adjusted"):
        votes.append("B" if global_reversal.get("B", 0.5) >= global_reversal.get("P", 0.5) else "P")
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
    if majority_guard.get("active") and majority_guard.get("adjusted"):
        cap = MAJORITY_STRONG_CONF_CAP if majority_guard.get("forced") else MAJORITY_CONF_CAP
        conf = min(conf, cap)
        if not chaos.get("active"):
            level = "多數邊過熱警戒" if majority_guard.get("forced") else "多數邊保護弱訊號"
    if global_reversal.get("active") and global_reversal.get("adjusted"):
        cap = GLOBAL_REVERSAL_STRONG_CONF_CAP if global_reversal.get("forced") else GLOBAL_REVERSAL_CONF_CAP
        conf = min(conf, cap)
        if not chaos.get("active") and not (majority_guard.get("active") and majority_guard.get("adjusted")):
            level = "全局反轉警戒" if global_reversal.get("forced") else "反轉觀察弱訊號"
    reason_parts = [road.get("label", "牌路"), f"模型一致{int(agreement * 100)}%"]
    if chaos.get("active"):
        reason_parts.insert(0, f"{chaos.get('label')}({int(float(chaos.get('score', 0))*100)}%)")
        if LOW_CONFIDENCE_MINBET:
            reason_parts.append("建議最小注")
    if majority_guard.get("active") and majority_guard.get("adjusted"):
        reason_parts.insert(0, f"{majority_guard.get('label')}({int(float(majority_guard.get('score', 0))*100)}%)")
        if majority_guard.get("forced"):
            reason_parts.append("多數邊過熱轉向校準")
        else:
            reason_parts.append("降低多數邊追擊")
    if global_reversal.get("active") and global_reversal.get("adjusted"):
        reason_parts.insert(0, f"{global_reversal.get('label')}({int(float(global_reversal.get('score', 0))*100)}%)")
        if global_reversal.get("forced"):
            reason_parts.append("全局反轉校準")
        else:
            reason_parts.append("反轉雷達降追擊")
    if road.get("road_action"):
        reason_parts.append(f"動作:{road.get('road_action')}")
    if 'reversal_final_override' in locals() and reversal_final_override:
        reason_parts.append("強轉龍校準")
    if 'chop_to_dragon_final_override' in locals() and chop_to_dragon_final_override:
        reason_parts.append("單跳轉龍校準")
    if 'mirror_run_final_override' in locals() and mirror_run_final_override:
        reason_parts.append("對稱龍長校準")
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
        "bet_mode": "最小注" if (
            (chaos.get("active") and LOW_CONFIDENCE_MINBET)
            or (majority_guard.get("active") and majority_guard.get("adjusted"))
            or (global_reversal.get("active") and global_reversal.get("adjusted"))
        ) else "信心分級",
        "pattern_label": road.get("label", ""),
        "chaos_label": chaos.get("label", ""),
        "chaos_score": chaos.get("score", 0),
        "reason": " / ".join(reason_parts),
        "dragon": {
            "current_streak": _streak(non_tie),
            "runs_tail": run_data[-10:],
            "road_strength": round(float(road.get("strength", 0)), 3),
            "first_side": road.get("first_side"),
            "breakout": road.get("breakout"),
            "reversal": road.get("reversal"),
            "chop_to_dragon": road.get("chop_to_dragon"),
            "mirror_run": road.get("mirror_run"),
            "road_action": road.get("road_action", ""),
        },
        "chaos": chaos,
        "majority_guard": majority_guard,
        "global_reversal": global_reversal,
        "effective_weights": {k: round(v, 4) for k, v in weights.items()},
        "ai_used": bool(ai_result and not ai_result.get("error")),
        "ai_result": ai_result if os.getenv("DEBUG_AI_RESULT", "0") == "1" else None,
        "debug": feature_payload if os.getenv("DEBUG_PREDICTOR", "0") == "1" else None,
    }
