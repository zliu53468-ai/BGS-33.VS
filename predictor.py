import math
import os
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Dict, List, Tuple

from deepseek_client import DeepSeekClient

# ============================================================
# V19 Classic / Lite + Full Markov + Regime/Dynamic/Bayes
# 目標：回到百家樂核心牌路判斷，並補強莊閒排列順序捕捉。
# 主判斷保留 Road / Room / Foot / Chop / Dragon / After-Tie。
# 新增 Sequence Pattern Layer、Full Markov、Regime Switch、Dynamic Weight、Bayesian Calibration。
# AI、全靴前中後、全局反轉、多數邊、Chaos 預設只做輕量保護或關閉硬覆蓋。
# ============================================================

# Base baccarat long-run priors, used only as soft priors for display calibration.
B_PRIOR = float(os.getenv("B_PRIOR", "0.4586"))
P_PRIOR = float(os.getenv("P_PRIOR", "0.4462"))
T_PRIOR = float(os.getenv("T_PRIOR", "0.0952"))

# Main ensemble weights.
MARKOV_WEIGHT = float(os.getenv("MARKOV_WEIGHT", "0.14"))
ROAD_WEIGHT = float(os.getenv("ROAD_WEIGHT", "0.30"))
STREAK_WEIGHT = float(os.getenv("STREAK_WEIGHT", "0.06"))
BALANCE_WEIGHT = float(os.getenv("BALANCE_WEIGHT", "0.06"))
RECENT_WEIGHT = float(os.getenv("RECENT_WEIGHT", "0.09"))
TIE_WEIGHT = float(os.getenv("TIE_WEIGHT", "0.02"))
AI_BLEND = float(os.getenv("AI_BLEND", "0.06"))

# DeepSeek full-shoe payload controls.
# The local model always uses the full current shoe. These settings make the
# DeepSeek calibration layer receive the full current shoe as raw B/P/T sequence
# plus full run-length diagnostics instead of only a short tail summary.
AI_FULL_HISTORY_MODE = os.getenv("AI_FULL_HISTORY_MODE", "1") == "1"
AI_HISTORY_FULL_LIMIT = int(os.getenv("AI_HISTORY_FULL_LIMIT", "90"))
AI_NON_TIE_FULL_LIMIT = int(os.getenv("AI_NON_TIE_FULL_LIMIT", "90"))
AI_RUNS_FULL_LIMIT = int(os.getenv("AI_RUNS_FULL_LIMIT", "50"))
AI_HISTORY_TAIL_LIMIT = int(os.getenv("AI_HISTORY_TAIL_LIMIT", "50"))
AI_RUNS_TAIL_LIMIT = int(os.getenv("AI_RUNS_TAIL_LIMIT", "28"))
AI_INCLUDE_PATTERN_DIAGNOSTICS = os.getenv("AI_INCLUDE_PATTERN_DIAGNOSTICS", "1") == "1"

# Tie handling. Tie should usually be a warning layer, not a main recommendation.
TIE_SHRINK = float(os.getenv("TIE_SHRINK", "0.35"))
TIE_MAX_PROB = float(os.getenv("TIE_MAX_PROB", "0.18"))
ALLOW_TIE_RECOMMEND = os.getenv("ALLOW_TIE_RECOMMEND", "0") == "1"
TIE_RECOMMEND_MIN = float(os.getenv("TIE_RECOMMEND_MIN", "0.165"))

MIN_HISTORY_FOR_AI = int(os.getenv("MIN_HISTORY_FOR_AI", "12"))
MIN_HISTORY_FOR_SIGNAL = int(os.getenv("MIN_HISTORY_FOR_SIGNAL", "4"))

# Advanced road / dragon controls.
DRAGON_MIN_LEN = int(os.getenv("DRAGON_MIN_LEN", "2"))
DRAGON_STRONG_LEN = int(os.getenv("DRAGON_STRONG_LEN", "4"))
DRAGON_FATIGUE_START = int(os.getenv("DRAGON_FATIGUE_START", "4"))
DRAGON_MAX_EDGE = float(os.getenv("DRAGON_MAX_EDGE", "0.056"))
DRAGON_BREAK_EDGE = float(os.getenv("DRAGON_BREAK_EDGE", "0.092"))
RUN_CYCLE_MIN_HITS = int(os.getenv("RUN_CYCLE_MIN_HITS", "2"))
ROAD_PATTERN_WINDOW = int(os.getenv("ROAD_PATTERN_WINDOW", "12"))
PATTERN_LOOKBACK = int(os.getenv("PATTERN_LOOKBACK", "4"))
MARKOV_ALPHA = float(os.getenv("MARKOV_ALPHA", "2.6"))
MARKOV_FULL_SAMPLE = float(os.getenv("MARKOV_FULL_SAMPLE", "16"))

# V18 Full Markov Layer / 完整馬可夫層:
# Adds multi-order Markov transitions (1~5 order by default) plus optional
# run-length state Markov. This is still a light calibration layer; by default
# it does not hard-override Road / Room / Foot / Dragon.
FULL_MARKOV_MODE = os.getenv("FULL_MARKOV_MODE", "1") == "1"
FULL_MARKOV_WEIGHT = float(os.getenv("FULL_MARKOV_WEIGHT", "0.25"))
FULL_MARKOV_ORDER_MIN = int(os.getenv("FULL_MARKOV_ORDER_MIN", "1"))
FULL_MARKOV_ORDER_MAX = int(os.getenv("FULL_MARKOV_ORDER_MAX", "5"))
FULL_MARKOV_MIN_HISTORY = int(os.getenv("FULL_MARKOV_MIN_HISTORY", "10"))
FULL_MARKOV_MIN_SAMPLE = int(os.getenv("FULL_MARKOV_MIN_SAMPLE", "2"))
FULL_MARKOV_ALPHA = float(os.getenv("FULL_MARKOV_ALPHA", "1.4"))
FULL_MARKOV_EDGE = float(os.getenv("FULL_MARKOV_EDGE", "0.042"))
FULL_MARKOV_MAX_EDGE = float(os.getenv("FULL_MARKOV_MAX_EDGE", "0.074"))
FULL_MARKOV_DECAY = float(os.getenv("FULL_MARKOV_DECAY", "0.965"))
FULL_MARKOV_RUN_STATE_MODE = os.getenv("FULL_MARKOV_RUN_STATE_MODE", "1") == "1"
FULL_MARKOV_RUN_WEIGHT = float(os.getenv("FULL_MARKOV_RUN_WEIGHT", "0.42"))
FULL_MARKOV_FINAL_OVERRIDE = os.getenv("FULL_MARKOV_FINAL_OVERRIDE", "0") == "1"
FULL_MARKOV_CONF_CAP = float(os.getenv("FULL_MARKOV_CONF_CAP", "0.50"))
FULL_MARKOV_STRONG_LOCAL_GAP = float(os.getenv("FULL_MARKOV_STRONG_LOCAL_GAP", "0.055"))
FULL_MARKOV_CHAOS_FACTOR = float(os.getenv("FULL_MARKOV_CHAOS_FACTOR", "0.70"))

# Breakout Dragon Mode:
# Handles shoes where a Banker/Player dragon suddenly exceeds previous run lengths.
# It protects 1~2 hands after a true breakout so the model does not force-break too early.
DRAGON_MEMORY_LOOKBACK = int(os.getenv("DRAGON_MEMORY_LOOKBACK", "28"))
DRAGON_BREAK_REPEAT_MIN = int(os.getenv("DRAGON_BREAK_REPEAT_MIN", "3"))
BREAKOUT_DRAGON_MODE = os.getenv("BREAKOUT_DRAGON_MODE", "1") == "1"
BREAKOUT_MIN_LEN = int(os.getenv("BREAKOUT_MIN_LEN", str(DRAGON_STRONG_LEN)))
BREAKOUT_PROTECT_STEPS = int(os.getenv("BREAKOUT_PROTECT_STEPS", "0"))
BREAKOUT_EXTEND_STEPS = int(os.getenv("BREAKOUT_EXTEND_STEPS", "1"))
BREAKOUT_CONT_EDGE = float(os.getenv("BREAKOUT_CONT_EDGE", "0.014"))
BREAKOUT_NEW_HIGH_BONUS = float(os.getenv("BREAKOUT_NEW_HIGH_BONUS", "0.004"))
BREAKOUT_OVERHEAT_PENALTY = float(os.getenv("BREAKOUT_OVERHEAT_PENALTY", "0.030"))

# Chaos / broken-road regime controls.
# These protect the model from forcing dragon/chop/double-chop logic on unstable shoes.
CHAOS_MODE = os.getenv("CHAOS_MODE", "1") == "1"
PATTERN_FAILURE_COUNTER = os.getenv("PATTERN_FAILURE_COUNTER", "1") == "1"
FAKE_DRAGON_DETECTOR = os.getenv("FAKE_DRAGON_DETECTOR", "1") == "1"
CHOP_BREAK_DETECTOR = os.getenv("CHOP_BREAK_DETECTOR", "1") == "1"
LOW_CONFIDENCE_MINBET = os.getenv("LOW_CONFIDENCE_MINBET", "1") == "1"

CHAOS_WINDOW = int(os.getenv("CHAOS_WINDOW", "18"))
CHAOS_TRIGGER = float(os.getenv("CHAOS_TRIGGER", "0.70"))
CHAOS_STRONG_TRIGGER = float(os.getenv("CHAOS_STRONG_TRIGGER", "0.84"))
CHAOS_MAX_EDGE = float(os.getenv("CHAOS_MAX_EDGE", "0.025"))
CHAOS_RECENT_BLEND = float(os.getenv("CHAOS_RECENT_BLEND", "0.12"))
CHAOS_CONF_CAP = float(os.getenv("CHAOS_CONF_CAP", "0.58"))
CHAOS_STRONG_CONF_CAP = float(os.getenv("CHAOS_STRONG_CONF_CAP", "0.48"))
CHAOS_AI_BLEND_FACTOR = float(os.getenv("CHAOS_AI_BLEND_FACTOR", "0.45"))

# Dynamic weight factors used only when chaos mode is active.
CHAOS_MARKOV_WEIGHT_FACTOR = float(os.getenv("CHAOS_MARKOV_WEIGHT_FACTOR", "1.22"))
CHAOS_ROAD_WEIGHT_FACTOR = float(os.getenv("CHAOS_ROAD_WEIGHT_FACTOR", "0.85"))
CHAOS_STREAK_WEIGHT_FACTOR = float(os.getenv("CHAOS_STREAK_WEIGHT_FACTOR", "0.75"))
CHAOS_BALANCE_WEIGHT_FACTOR = float(os.getenv("CHAOS_BALANCE_WEIGHT_FACTOR", "0.70"))
CHAOS_RECENT_WEIGHT_FACTOR = float(os.getenv("CHAOS_RECENT_WEIGHT_FACTOR", "1.10"))

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
FIRST_SIDE_DRAGON_PROTECT_STEPS = int(os.getenv("FIRST_SIDE_DRAGON_PROTECT_STEPS", "0"))
FIRST_SIDE_DRAGON_EDGE = float(os.getenv("FIRST_SIDE_DRAGON_EDGE", "0.014"))
FIRST_SIDE_DRAGON_MAX_EDGE = float(os.getenv("FIRST_SIDE_DRAGON_MAX_EDGE", "0.030"))
SIDE_AWARE_BREAKOUT = os.getenv("SIDE_AWARE_BREAKOUT", "1") == "1"

# Dragon reversal / turning-point controls.
# This layer asks whether an active dragon is likely entering a turning zone.
# It uses length fatigue, repeated same-side cut lengths, imbalance, and prior
# max length instead of blindly following every long dragon.
DRAGON_REVERSAL_MODE = os.getenv("DRAGON_REVERSAL_MODE", "1") == "1"
REVERSAL_MIN_LEN = int(os.getenv("REVERSAL_MIN_LEN", "3"))
REVERSAL_FATIGUE_LEN = int(os.getenv("REVERSAL_FATIGUE_LEN", "4"))
REVERSAL_TRIGGER = float(os.getenv("REVERSAL_TRIGGER", "0.38"))
REVERSAL_STRONG_TRIGGER = float(os.getenv("REVERSAL_STRONG_TRIGGER", "0.56"))
REVERSAL_REPEAT_NEAR_MIN = int(os.getenv("REVERSAL_REPEAT_NEAR_MIN", "1"))
REVERSAL_OVER_MAX_LEN = int(os.getenv("REVERSAL_OVER_MAX_LEN", "1"))
REVERSAL_IMBALANCE_TRIGGER = float(os.getenv("REVERSAL_IMBALANCE_TRIGGER", "0.62"))
REVERSAL_EDGE = float(os.getenv("REVERSAL_EDGE", "0.052"))
REVERSAL_HARD_EDGE = float(os.getenv("REVERSAL_HARD_EDGE", "0.088"))
REVERSAL_PROTECT_FIRST_STEPS = int(os.getenv("REVERSAL_PROTECT_FIRST_STEPS", "1"))
REVERSAL_FINAL_OVERRIDE = os.getenv("REVERSAL_FINAL_OVERRIDE", "0") == "1"
REVERSAL_OVERRIDE_EDGE = float(os.getenv("REVERSAL_OVERRIDE_EDGE", "0.040"))

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
CHOP_TO_DRAGON_FINAL_OVERRIDE = os.getenv("CHOP_TO_DRAGON_FINAL_OVERRIDE", "0") == "1"
CHOP_TO_DRAGON_OVERRIDE_EDGE = float(os.getenv("CHOP_TO_DRAGON_OVERRIDE_EDGE", "0.040"))

# Room Pattern Mode:
# Early detector for double chop, one-room-two-halls, and two-room-one-hall rhythms.
# It reads run lengths directly so patterns like BB PP B, B PP B, or BB P BB
# can be detected before the road is already fully obvious.
ROOM_PATTERN_MODE = os.getenv("ROOM_PATTERN_MODE", "1") == "1"
ROOM_PATTERN_LOOKBACK = int(os.getenv("ROOM_PATTERN_LOOKBACK", "8"))
ROOM_PATTERN_MIN_RUNS = int(os.getenv("ROOM_PATTERN_MIN_RUNS", "3"))
ROOM_PATTERN_EARLY_DETECT = os.getenv("ROOM_PATTERN_EARLY_DETECT", "1") == "1"
ROOM_PATTERN_MIN_CONSISTENCY = float(os.getenv("ROOM_PATTERN_MIN_CONSISTENCY", "0.64"))
ROOM_PATTERN_EDGE = float(os.getenv("ROOM_PATTERN_EDGE", "0.066"))
ROOM_PATTERN_MAX_EDGE = float(os.getenv("ROOM_PATTERN_MAX_EDGE", "0.092"))
ROOM_PATTERN_STRENGTH = float(os.getenv("ROOM_PATTERN_STRENGTH", "0.300"))
ONE_TWO_PATTERN_MODE = os.getenv("ONE_TWO_PATTERN_MODE", "1") == "1"
ONE_TWO_PATTERN_EDGE = float(os.getenv("ONE_TWO_PATTERN_EDGE", "0.068"))
TWO_ONE_PATTERN_MODE = os.getenv("TWO_ONE_PATTERN_MODE", "1") == "1"
TWO_ONE_PATTERN_EDGE = float(os.getenv("TWO_ONE_PATTERN_EDGE", "0.068"))
DOUBLE_CHOP_EARLY_MODE = os.getenv("DOUBLE_CHOP_EARLY_MODE", "1") == "1"
DOUBLE_CHOP_EARLY_HITS = int(os.getenv("DOUBLE_CHOP_EARLY_HITS", "2"))
DOUBLE_CHOP_EDGE = float(os.getenv("DOUBLE_CHOP_EDGE", "0.066"))
DOUBLE_CHOP_BREAK_GUARD = os.getenv("DOUBLE_CHOP_BREAK_GUARD", "1") == "1"
ROOM_PATTERN_FINAL_OVERRIDE = os.getenv("ROOM_PATTERN_FINAL_OVERRIDE", "1") == "1"
ROOM_PATTERN_OVERRIDE_EDGE = float(os.getenv("ROOM_PATTERN_OVERRIDE_EDGE", "0.074"))
ROOM_PATTERN_CHAOS_RELIEF = float(os.getenv("ROOM_PATTERN_CHAOS_RELIEF", "0.080"))
ROOM_PATTERN_PROTECT_RUNS = int(os.getenv("ROOM_PATTERN_PROTECT_RUNS", "1"))

# Foot Alignment Mode / 對應齊腳模式:
# Handles "aligned feet" situations: one side fills to the previous side's run length,
# then decides whether the road usually breaks at the alignment point or continues past it.
# Examples:
#   B B -> P       : P may need to fill to P2.
#   B B -> P P     : feet are aligned; decide P breaks to B or over-steps to P3.
#   B B B -> P P   : P may need to fill to P3.
FOOT_ALIGNMENT_MODE = os.getenv("FOOT_ALIGNMENT_MODE", "1") == "1"
FOOT_ALIGN_LOOKBACK = int(os.getenv("FOOT_ALIGN_LOOKBACK", "10"))
FOOT_ALIGN_MIN_LEN = int(os.getenv("FOOT_ALIGN_MIN_LEN", "2"))
FOOT_ALIGN_MAX_LEN = int(os.getenv("FOOT_ALIGN_MAX_LEN", "5"))
FOOT_ALIGN_MATCH_TOLERANCE = int(os.getenv("FOOT_ALIGN_MATCH_TOLERANCE", "0"))
FOOT_ALIGN_MIN_CONTEXT = int(os.getenv("FOOT_ALIGN_MIN_CONTEXT", "2"))
FOOT_ALIGN_EDGE = float(os.getenv("FOOT_ALIGN_EDGE", "0.052"))
FOOT_ALIGN_BREAK_EDGE = float(os.getenv("FOOT_ALIGN_BREAK_EDGE", "0.064"))
FOOT_ALIGN_OVER_EDGE = float(os.getenv("FOOT_ALIGN_OVER_EDGE", "0.040"))
FOOT_ALIGN_MAX_EDGE = float(os.getenv("FOOT_ALIGN_MAX_EDGE", "0.074"))
FOOT_ALIGN_STRENGTH = float(os.getenv("FOOT_ALIGN_STRENGTH", "0.250"))
FOOT_ALIGN_BREAK_RATE = float(os.getenv("FOOT_ALIGN_BREAK_RATE", "0.58"))
FOOT_ALIGN_OVER_RATE = float(os.getenv("FOOT_ALIGN_OVER_RATE", "0.58"))
FOOT_ALIGN_DEFAULT_BREAK = os.getenv("FOOT_ALIGN_DEFAULT_BREAK", "1") == "1"
FOOT_ALIGN_BREAK_GUARD = os.getenv("FOOT_ALIGN_BREAK_GUARD", "1") == "1"
FOOT_ALIGN_OVERFLOW_HANDOFF = int(os.getenv("FOOT_ALIGN_OVERFLOW_HANDOFF", "2"))
FOOT_ALIGN_FINAL_OVERRIDE = os.getenv("FOOT_ALIGN_FINAL_OVERRIDE", "1") == "1"
FOOT_ALIGN_OVERRIDE_EDGE = float(os.getenv("FOOT_ALIGN_OVERRIDE_EDGE", "0.064"))
FOOT_ALIGN_CHAOS_RELIEF = float(os.getenv("FOOT_ALIGN_CHAOS_RELIEF", "0.075"))
FOOT_ALIGN_ROOM_BOOST = float(os.getenv("FOOT_ALIGN_ROOM_BOOST", "0.014"))

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
MIRROR_RUN_FINAL_OVERRIDE = os.getenv("MIRROR_RUN_FINAL_OVERRIDE", "0") == "1"
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
MAJORITY_FINAL_OVERRIDE = os.getenv("MAJORITY_FINAL_OVERRIDE", "0") == "1"

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
GLOBAL_REVERSAL_FINAL_OVERRIDE = os.getenv("GLOBAL_REVERSAL_FINAL_OVERRIDE", "0") == "1"
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


# V14 Global Shoe Context Mode:
# Full-shoe controller for 60~70-round baccarat tables. It compares early/mid/recent
# shoe phases and prevents local short-window signals from overpowering the shoe's
# dominant road structure too easily.
GLOBAL_SHOE_CONTEXT_MODE = os.getenv("GLOBAL_SHOE_CONTEXT_MODE", "1") == "1"
GLOBAL_SHOE_MIN_HISTORY = int(os.getenv("GLOBAL_SHOE_MIN_HISTORY", "18"))
GLOBAL_SHOE_WINDOW = int(os.getenv("GLOBAL_SHOE_WINDOW", "24"))
GLOBAL_SHOE_CONTEXT_TRIGGER = float(os.getenv("GLOBAL_SHOE_CONTEXT_TRIGGER", "0.58"))
GLOBAL_SHOE_STRONG_TRIGGER = float(os.getenv("GLOBAL_SHOE_STRONG_TRIGGER", "0.74"))
GLOBAL_SHOE_CONTEXT_EDGE = float(os.getenv("GLOBAL_SHOE_CONTEXT_EDGE", "0.022"))
GLOBAL_SHOE_OVERRIDE_EDGE = float(os.getenv("GLOBAL_SHOE_OVERRIDE_EDGE", "0.036"))
GLOBAL_SHOE_CONTEXT_WEIGHT = float(os.getenv("GLOBAL_SHOE_CONTEXT_WEIGHT", "0.10"))
GLOBAL_SHOE_CONF_CAP = float(os.getenv("GLOBAL_SHOE_CONF_CAP", "0.50"))
GLOBAL_SHOE_STRONG_CONF_CAP = float(os.getenv("GLOBAL_SHOE_STRONG_CONF_CAP", "0.42"))
GLOBAL_SHOE_PHASE_SPLIT = int(os.getenv("GLOBAL_SHOE_PHASE_SPLIT", "3"))
GLOBAL_SHOE_ALLOW_RECENT_SHIFT = os.getenv("GLOBAL_SHOE_ALLOW_RECENT_SHIFT", "0") == "1"

# V14 Early Dragon Guard:
# Prevents blindly following very early 2~4 hand dragons unless there is room/foot/mirror
# support or strong full-shoe evidence. This is separate from long-dragon reversal.
EARLY_DRAGON_GUARD = os.getenv("EARLY_DRAGON_GUARD", "0") == "1"
EARLY_DRAGON_WARN_LEN = int(os.getenv("EARLY_DRAGON_WARN_LEN", "2"))
EARLY_DRAGON_ALERT_LEN = int(os.getenv("EARLY_DRAGON_ALERT_LEN", "3"))
EARLY_DRAGON_MAX_LEN = int(os.getenv("EARLY_DRAGON_MAX_LEN", "4"))
EARLY_DRAGON_TRIGGER = float(os.getenv("EARLY_DRAGON_TRIGGER", "0.42"))
EARLY_DRAGON_STRONG_TRIGGER = float(os.getenv("EARLY_DRAGON_STRONG_TRIGGER", "0.62"))
EARLY_DRAGON_FOLLOW_CAP = float(os.getenv("EARLY_DRAGON_FOLLOW_CAP", "0.535"))
EARLY_DRAGON_BREAK_EDGE = float(os.getenv("EARLY_DRAGON_BREAK_EDGE", "0.052"))
EARLY_DRAGON_REQUIRE_ROOM_SUPPORT = os.getenv("EARLY_DRAGON_REQUIRE_ROOM_SUPPORT", "1") == "1"
EARLY_DRAGON_CONF_CAP = float(os.getenv("EARLY_DRAGON_CONF_CAP", "0.44"))
EARLY_DRAGON_STRONG_CONF_CAP = float(os.getenv("EARLY_DRAGON_STRONG_CONF_CAP", "0.36"))

# V14 Room Break To Chop Mode:
# Detects when one-room-two-halls / two-room-one-hall / double-chop rhythm has repeated
# 3~4 cycles and then starts breaking into single chop or a new short rhythm.
ROOM_BREAK_TO_CHOP_MODE = os.getenv("ROOM_BREAK_TO_CHOP_MODE", "0") == "1"
ROOM_BREAK_REPEAT_MIN = int(os.getenv("ROOM_BREAK_REPEAT_MIN", "3"))
ROOM_BREAK_LOOKBACK = int(os.getenv("ROOM_BREAK_LOOKBACK", "8"))
ROOM_BREAK_CONSISTENCY = float(os.getenv("ROOM_BREAK_CONSISTENCY", "0.68"))
ROOM_BREAK_TO_CHOP_EDGE = float(os.getenv("ROOM_BREAK_TO_CHOP_EDGE", "0.056"))
ROOM_BREAK_DAMPEN_ROOM = float(os.getenv("ROOM_BREAK_DAMPEN_ROOM", "0.055"))
ROOM_BREAK_FINAL_OVERRIDE = os.getenv("ROOM_BREAK_FINAL_OVERRIDE", "0") == "1"
ROOM_BREAK_CONF_CAP = float(os.getenv("ROOM_BREAK_CONF_CAP", "0.46"))

# V14 After Tie Safe Mode:
# When a tie appears, the next N hands are treated as low-confidence / min-bet zone.
# Direction can still be shown, but strong-entry confidence is capped.
AFTER_TIE_SAFE_MODE = os.getenv("AFTER_TIE_SAFE_MODE", "1") == "1"
AFTER_TIE_NO_ENTRY_WINDOW = int(os.getenv("AFTER_TIE_NO_ENTRY_WINDOW", "3"))
AFTER_TIE_FORCE_MINBET = os.getenv("AFTER_TIE_FORCE_MINBET", "1") == "1"
AFTER_TIE_CONF_CAP = float(os.getenv("AFTER_TIE_CONF_CAP", "0.36"))


# V17 Sequence Pattern Layer / 莊閒排列順序層:
# Captures direct B/P ordering rhythms such as BPBP, BBPP, BPPBPP, BBPBBP,
# and short N-gram tails. This layer is intentionally light-weight: it can
# nudge the B/P probability, but by default it never hard-overrides the pick.
SEQUENCE_PATTERN_MODE = os.getenv("SEQUENCE_PATTERN_MODE", "1") == "1"
SEQUENCE_LOOKBACK = int(os.getenv("SEQUENCE_LOOKBACK", "14"))
SEQUENCE_MIN_HISTORY = int(os.getenv("SEQUENCE_MIN_HISTORY", "8"))
SEQUENCE_NGRAM_MIN = int(os.getenv("SEQUENCE_NGRAM_MIN", "3"))
SEQUENCE_NGRAM_MAX = int(os.getenv("SEQUENCE_NGRAM_MAX", "5"))
SEQUENCE_MIN_SAMPLE = int(os.getenv("SEQUENCE_MIN_SAMPLE", "2"))
SEQUENCE_EDGE = float(os.getenv("SEQUENCE_EDGE", "0.038"))
SEQUENCE_MAX_EDGE = float(os.getenv("SEQUENCE_MAX_EDGE", "0.060"))
SEQUENCE_WEIGHT = float(os.getenv("SEQUENCE_WEIGHT", "0.16"))
SEQUENCE_FINAL_OVERRIDE = os.getenv("SEQUENCE_FINAL_OVERRIDE", "0") == "1"
SEQUENCE_CONF_CAP = float(os.getenv("SEQUENCE_CONF_CAP", "0.50"))
SEQUENCE_CHAOS_FACTOR = float(os.getenv("SEQUENCE_CHAOS_FACTOR", "0.70"))
SEQUENCE_STRONG_LOCAL_GAP = float(os.getenv("SEQUENCE_STRONG_LOCAL_GAP", "0.050"))


# V19 Regime Switch Model / 路型狀態切換模型:
# Detects the current shoe regime first, then lightly shifts which model should speak louder.
# It keeps Final Override off; it only changes weights/blends so the model keeps baccarat road feel.
REGIME_SWITCH_MODE = os.getenv("REGIME_SWITCH_MODE", "1") == "1"
REGIME_WINDOW = int(os.getenv("REGIME_WINDOW", "18"))
REGIME_MIN_CONFIDENCE = float(os.getenv("REGIME_MIN_CONFIDENCE", "0.58"))
REGIME_WEIGHT = float(os.getenv("REGIME_WEIGHT", "0.18"))
REGIME_MAX_SHIFT = float(os.getenv("REGIME_MAX_SHIFT", "0.12"))
REGIME_FINAL_OVERRIDE = os.getenv("REGIME_FINAL_OVERRIDE", "0") == "1"
REGIME_CHAOS_RELIEF = float(os.getenv("REGIME_CHAOS_RELIEF", "0.70"))

# V19 Bayesian Calibration / 貝葉斯校準模型:
# Shrinks low-sample Markov/sequence signals back toward 50/50 so short coincidences do not dominate.
BAYES_CALIBRATION_MODE = os.getenv("BAYES_CALIBRATION_MODE", "1") == "1"
BAYES_ALPHA = float(os.getenv("BAYES_ALPHA", "2.0"))
BAYES_MIN_SAMPLE = int(os.getenv("BAYES_MIN_SAMPLE", "3"))
BAYES_SHRINK = float(os.getenv("BAYES_SHRINK", "0.62"))
BAYES_MAX_EDGE = float(os.getenv("BAYES_MAX_EDGE", "0.042"))
BAYES_APPLY_FULL_MARKOV = os.getenv("BAYES_APPLY_FULL_MARKOV", "1") == "1"
BAYES_APPLY_SEQUENCE = os.getenv("BAYES_APPLY_SEQUENCE", "1") == "1"

# V19 Dynamic Ensemble Weight / 動態權重模型:
# Backtests recent in-shoe component picks and nudges weights toward the components that are currently fitting.
DYNAMIC_WEIGHT_MODE = os.getenv("DYNAMIC_WEIGHT_MODE", "1") == "1"
DYNAMIC_WEIGHT_WINDOW = int(os.getenv("DYNAMIC_WEIGHT_WINDOW", "14"))
DYNAMIC_WEIGHT_MIN_SAMPLE = int(os.getenv("DYNAMIC_WEIGHT_MIN_SAMPLE", "6"))
DYNAMIC_WEIGHT_MAX_SHIFT = float(os.getenv("DYNAMIC_WEIGHT_MAX_SHIFT", "0.08"))
DYNAMIC_WEIGHT_DECAY = float(os.getenv("DYNAMIC_WEIGHT_DECAY", "0.92"))
DYNAMIC_WEIGHT_STEP = float(os.getenv("DYNAMIC_WEIGHT_STEP", "0.025"))
DYNAMIC_WEIGHT_APPLY_FULL_MARKOV = os.getenv("DYNAMIC_WEIGHT_APPLY_FULL_MARKOV", "1") == "1"
DYNAMIC_WEIGHT_APPLY_SEQUENCE = os.getenv("DYNAMIC_WEIGHT_APPLY_SEQUENCE", "1") == "1"


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



def _room_pattern_score(non_tie: List[str]) -> Dict[str, Any]:
    """
    Room Pattern Mode / 房型規律模型.

    Detects these short-run rhythms early:
    - Double chop / 雙跳:       2-2-2...  e.g. BB PP B -> expect B to fill BB; BB PP BB -> expect P
    - One-room-two-halls / 一房兩廳: 1-2-1-2... e.g. B PP B -> expect P; B PP B P -> expect P to fill PP
    - Two-room-one-hall / 兩房一廳: 2-1-2-1... e.g. BB P BB -> expect P; BB P BB P -> expect B

    The key improvement is early detection: the current run is allowed to be
    unfinished when it is shorter than the expected room length. This prevents
    the model from lagging several hands behind the rhythm.
    """
    if not ROOM_PATTERN_MODE or len(non_tie) < 4:
        return {"B": 0.5, "P": 0.5, "label": "房型資料不足", "strength": 0.0, "active": False}

    runs = _runs(non_tie)
    if len(runs) < max(3, ROOM_PATTERN_MIN_RUNS):
        return {"B": 0.5, "P": 0.5, "label": "房型資料不足", "strength": 0.0, "active": False}

    current_side, current_len = runs[-1]
    lengths = [n for _s, n in runs]
    tail_runs = runs[-ROOM_PATTERN_LOOKBACK:] if ROOM_PATTERN_LOOKBACK > 0 else runs
    tail_lengths = [n for _s, n in tail_runs]

    pattern_defs: List[Dict[str, Any]] = []
    if DOUBLE_CHOP_EARLY_MODE:
        pattern_defs.append({
            "key": "double_chop",
            "name": "雙跳",
            "pattern": [2, 2],
            "edge": DOUBLE_CHOP_EDGE,
            "min_runs": max(3, DOUBLE_CHOP_EARLY_HITS + 1),
        })
    if ONE_TWO_PATTERN_MODE:
        pattern_defs.append({
            "key": "one_two",
            "name": "一房兩廳",
            "pattern": [1, 2],
            "edge": ONE_TWO_PATTERN_EDGE,
            "min_runs": max(3, ROOM_PATTERN_MIN_RUNS),
        })
    if TWO_ONE_PATTERN_MODE:
        pattern_defs.append({
            "key": "two_one",
            "name": "兩房一廳",
            "pattern": [2, 1],
            "edge": TWO_ONE_PATTERN_EDGE,
            "min_runs": max(3, ROOM_PATTERN_MIN_RUNS),
        })

    best_eval: Dict[str, Any] | None = None

    for pdef in pattern_defs:
        pattern = pdef["pattern"]
        min_runs = int(pdef["min_runs"])
        if len(tail_lengths) < min_runs:
            continue

        # Prefer the latest 3~8 runs. For each possible phase offset, score how
        # well the observed run lengths fit the expected room rhythm.
        max_eval_runs = min(len(tail_lengths), max(min_runs, ROOM_PATTERN_LOOKBACK))
        eval_lengths = tail_lengths[-max_eval_runs:]
        for offset in range(len(pattern)):
            score = 0.0
            exact_hits = 0
            soft_hits = 0
            completed_hits = 0
            completed_total = 0
            mismatch = 0
            current_expected = pattern[(len(eval_lengths) - 1 + offset) % len(pattern)]

            for i, observed in enumerate(eval_lengths):
                expected = pattern[(i + offset) % len(pattern)]
                is_current = i == len(eval_lengths) - 1
                if is_current:
                    if observed == expected:
                        score += 1.0
                        exact_hits += 1
                    elif ROOM_PATTERN_EARLY_DETECT and observed < expected:
                        # Current room has started but is not filled yet; this is
                        # exactly the early signal we want to catch.
                        score += 0.82
                        soft_hits += 1
                    else:
                        mismatch += 1
                else:
                    completed_total += 1
                    if observed == expected:
                        score += 1.0
                        exact_hits += 1
                        completed_hits += 1
                    else:
                        # Old completed rooms should match more strictly. A small
                        # mismatch in a longer tail is tolerated but penalized.
                        mismatch += 1
                        if len(eval_lengths) >= 6 and abs(observed - expected) == 1 and observed <= 3:
                            score += 0.35

            consistency = score / max(1, len(eval_lengths))
            # Require enough completed confirmation; current soft-fill alone is not enough.
            if completed_total and completed_hits < max(1, min_runs - 2):
                consistency *= 0.72
            # If the current run already exceeds expected, the room rhythm is broken.
            current_overflow = current_len > current_expected
            if current_overflow:
                consistency *= 0.55

            eval_data = {
                "key": pdef["key"],
                "name": pdef["name"],
                "pattern": pattern,
                "offset": offset,
                "consistency": consistency,
                "score": score,
                "exact_hits": exact_hits,
                "soft_hits": soft_hits,
                "completed_hits": completed_hits,
                "completed_total": completed_total,
                "mismatch": mismatch,
                "current_expected": current_expected,
                "current_overflow": current_overflow,
                "base_edge": float(pdef["edge"]),
                "eval_lengths": eval_lengths,
            }
            if best_eval is None or eval_data["consistency"] > best_eval["consistency"]:
                best_eval = eval_data

    if not best_eval:
        return {"B": 0.5, "P": 0.5, "label": "未見房型規律", "strength": 0.0, "active": False}

    consistency = float(best_eval["consistency"])
    current_expected = int(best_eval["current_expected"])
    current_overflow = bool(best_eval["current_overflow"])

    # Break guard: do not keep forcing a room pattern after it has clearly overflowed.
    if current_overflow:
        if DOUBLE_CHOP_BREAK_GUARD and consistency >= ROOM_PATTERN_MIN_CONSISTENCY * 0.70:
            return {
                "B": 0.5,
                "P": 0.5,
                "label": f"房型破壞觀察｜{best_eval['name']}超長",
                "strength": 0.055,
                "active": False,
                "room_pattern": True,
                "phase": "break_guard",
                "pattern_name": best_eval["name"],
                "current_side": current_side,
                "current_len": current_len,
                "expected_len": current_expected,
                "consistency": round(consistency, 3),
                "road_action": "房型破壞/交回全局反轉",
            }
        return {"B": 0.5, "P": 0.5, "label": "房型已破壞", "strength": 0.0, "active": False}

    if consistency < ROOM_PATTERN_MIN_CONSISTENCY:
        return {
            "B": 0.5,
            "P": 0.5,
            "label": "房型規律未達門檻",
            "strength": 0.0,
            "active": False,
            "best_pattern": best_eval["name"],
            "consistency": round(consistency, 3),
        }

    # If current run is shorter than expected, predict same side to fill it.
    # If it already reached expected length, predict the opposite side to start the next room.
    if current_len < current_expected:
        phase = "fill"
        target_side = current_side
        action = f"補足{current_side}{current_expected}"
        label = f"{best_eval['name']}早期補房｜{current_side}{current_len}→{current_expected}"
    else:
        phase = "turn"
        target_side = _opposite(current_side)
        next_expected = best_eval["pattern"][(len(best_eval["eval_lengths"]) + best_eval["offset"]) % len(best_eval["pattern"])]
        action = f"轉邊開{target_side}{next_expected}"
        label = f"{best_eval['name']}節奏成型｜{current_side}{current_len}後{action}"

    # Edge/strength grow with consistency and exact hits, but stay capped so a room
    # pattern does not overpower a genuinely strong dragon forever.
    edge = ROOM_PATTERN_EDGE + (consistency - ROOM_PATTERN_MIN_CONSISTENCY) * 0.055
    edge += min(0.010, int(best_eval.get("exact_hits", 0)) * 0.0025)
    if best_eval["key"] == "double_chop":
        edge = max(edge, DOUBLE_CHOP_EDGE)
    elif best_eval["key"] == "one_two":
        edge = max(edge, ONE_TWO_PATTERN_EDGE)
    elif best_eval["key"] == "two_one":
        edge = max(edge, TWO_ONE_PATTERN_EDGE)
    edge = min(ROOM_PATTERN_MAX_EDGE, edge)

    strength = ROOM_PATTERN_STRENGTH + min(0.045, (consistency - ROOM_PATTERN_MIN_CONSISTENCY) * 0.18)
    if phase == "fill":
        strength += 0.012
    if int(best_eval.get("exact_hits", 0)) >= 4:
        strength += 0.012
    strength = _clamp(strength, 0.10, 0.285)

    b, p = _bp_score(target_side, 0.5 + edge)
    return {
        "B": b,
        "P": p,
        "label": label,
        "strength": strength,
        "active": True,
        "room_pattern": True,
        "phase": phase,
        "pattern_key": best_eval["key"],
        "pattern_name": best_eval["name"],
        "pattern": best_eval["pattern"],
        "target_side": target_side,
        "current_side": current_side,
        "current_len": current_len,
        "expected_len": current_expected,
        "edge": round(edge, 5),
        "consistency": round(consistency, 3),
        "exact_hits": int(best_eval.get("exact_hits", 0)),
        "soft_hits": int(best_eval.get("soft_hits", 0)),
        "mismatch": int(best_eval.get("mismatch", 0)),
        "eval_lengths": best_eval.get("eval_lengths", []),
        "road_action": "房型補足/轉邊",
        "bet_mode_hint": "房型小注" if consistency < 0.82 else "房型順勢",
    }



def _foot_alignment_score(non_tie: List[str]) -> Dict[str, Any]:
    """
    Foot Alignment Mode / 對應齊腳模式.

    This layer focuses on the exact situation the user described:
    - One side is filling up to the previous side's run length (對應齊腳前補腳).
    - Once both sides have the same run length, decide whether that alignment usually
      breaks to the other side or continues past the foot.
    - If it over-steps by only 1~2 hands, treat it as "not broken yet" but hand it
      back to dragon/reversal if it keeps extending.

    It is deliberately more specific than Mirror Run and Room Pattern:
    Mirror Run asks whether the current run should mirror the prior length;
    Foot Alignment asks what to do exactly at the aligned foot and immediately after it.
    """
    if not FOOT_ALIGNMENT_MODE or len(non_tie) < 4:
        return {"B": 0.5, "P": 0.5, "label": "齊腳資料不足", "strength": 0.0, "active": False}

    runs = _runs(non_tie)
    if len(runs) < 2:
        return {"B": 0.5, "P": 0.5, "label": "齊腳資料不足", "strength": 0.0, "active": False}

    prev_side, prev_len = runs[-2]
    current_side, current_len = runs[-1]
    if prev_side == current_side:
        return {"B": 0.5, "P": 0.5, "label": "非對應齊腳", "strength": 0.0, "active": False}

    if prev_len < FOOT_ALIGN_MIN_LEN or prev_len > FOOT_ALIGN_MAX_LEN:
        return {
            "B": 0.5,
            "P": 0.5,
            "label": "齊腳長度不適用",
            "strength": 0.0,
            "active": False,
            "prev_len": prev_len,
            "current_len": current_len,
        }

    tol = max(0, FOOT_ALIGN_MATCH_TOLERANCE)
    completed = runs[:-1]
    recent_completed = completed[-FOOT_ALIGN_LOOKBACK:] if FOOT_ALIGN_LOOKBACK > 0 else completed

    break_count = 0      # aligned then broke at that foot length
    over_count = 0       # aligned then continued past the foot length
    under_count = 0      # did not reach the previous foot length
    pair_samples = 0
    same_target_samples = 0

    # Study prior opposite-side run pairs: prev run length -> next run length.
    # If next run length == prev run length, the road broke right after alignment.
    # If next run length > prev run length, it did not break and over-stepped.
    for i in range(1, len(recent_completed)):
        ps, pl = recent_completed[i - 1]
        cs, cl = recent_completed[i]
        if ps == cs:
            continue
        if pl < FOOT_ALIGN_MIN_LEN or pl > FOOT_ALIGN_MAX_LEN:
            continue
        pair_samples += 1
        if abs(pl - prev_len) <= tol:
            same_target_samples += 1
            if abs(cl - pl) <= tol:
                break_count += 1
            elif cl > pl + tol:
                over_count += 1
            else:
                under_count += 1

    useful_samples = break_count + over_count
    break_rate = _safe_div(break_count, useful_samples, 0.0)
    over_rate = _safe_div(over_count, useful_samples, 0.0)

    # Phase 1: current side has not aligned yet, so fill to the previous foot.
    if current_len < prev_len - tol:
        missing = prev_len - current_len
        edge = FOOT_ALIGN_EDGE + min(0.014, current_len * 0.005) + min(0.010, same_target_samples * 0.003)
        edge = min(FOOT_ALIGN_MAX_EDGE, edge)
        strength = FOOT_ALIGN_STRENGTH + 0.012 + min(0.020, same_target_samples * 0.005)
        b, p = _bp_score(current_side, 0.5 + edge)
        return {
            "B": b,
            "P": p,
            "label": f"對應補齊腳{current_side}{current_len}→{prev_len}｜承接{prev_side}{prev_len}",
            "strength": _clamp(strength, 0.10, 0.255),
            "active": True,
            "foot_alignment": True,
            "phase": "fill_to_foot",
            "target_side": current_side,
            "prev_side": prev_side,
            "prev_len": prev_len,
            "current_side": current_side,
            "current_len": current_len,
            "missing": missing,
            "edge": round(edge, 5),
            "break_count": break_count,
            "over_count": over_count,
            "break_rate": round(break_rate, 3),
            "over_rate": round(over_rate, 3),
            "same_target_samples": same_target_samples,
            "road_action": "對應補齊腳/續到同長度",
        }

    # Phase 2: exactly at the foot. Decide break versus no-break.
    aligned_now = abs(current_len - prev_len) <= tol
    if aligned_now:
        has_context = useful_samples >= FOOT_ALIGN_MIN_CONTEXT
        room_pattern = _room_pattern_score(non_tie) if ROOM_PATTERN_MODE else {"active": False}
        room_target = room_pattern.get("target_side") if isinstance(room_pattern, dict) else None
        room_active = bool(isinstance(room_pattern, dict) and room_pattern.get("active"))

        # Historical evidence: if prior equal-foot situations usually broke, turn.
        if has_context and break_rate >= FOOT_ALIGN_BREAK_RATE:
            target_side = _opposite(current_side)
            phase = "aligned_break"
            edge = FOOT_ALIGN_BREAK_EDGE + min(0.016, (break_rate - FOOT_ALIGN_BREAK_RATE) * 0.08)
            label = f"齊腳破路觀察{current_side}{current_len}={prev_side}{prev_len}｜轉看{target_side}"
            action = "齊腳後破路/轉邊"
            strength_bonus = 0.030
        # Historical evidence: if prior equal-foot situations usually over-stepped, continue.
        elif has_context and over_rate >= FOOT_ALIGN_OVER_RATE:
            target_side = current_side
            phase = "aligned_over"
            edge = FOOT_ALIGN_OVER_EDGE + min(0.014, (over_rate - FOOT_ALIGN_OVER_RATE) * 0.07)
            label = f"齊腳未破續腳{current_side}{current_len}={prev_side}{prev_len}｜觀察過腳"
            action = "齊腳未破/續腳"
            strength_bonus = 0.018
        # Room rhythm can resolve tie-breaks at the foot.
        elif room_active and room_target in {"B", "P"}:
            target_side = str(room_target)
            phase = "aligned_room"
            edge = FOOT_ALIGN_EDGE + FOOT_ALIGN_ROOM_BOOST
            label = f"齊腳對應房型{current_side}{current_len}={prev_side}{prev_len}｜房型指向{target_side}"
            action = "齊腳後依房型補足/轉邊"
            strength_bonus = 0.022
        elif FOOT_ALIGN_DEFAULT_BREAK:
            target_side = _opposite(current_side)
            phase = "aligned_default_break"
            # Default break is intentionally weaker than evidence-backed break.
            edge = max(0.026, FOOT_ALIGN_BREAK_EDGE * 0.72)
            label = f"齊腳轉邊觀察{current_side}{current_len}={prev_side}{prev_len}｜樣本不足先看{target_side}"
            action = "齊腳樣本不足/小注轉邊觀察"
            strength_bonus = 0.000
        else:
            return {
                "B": 0.5,
                "P": 0.5,
                "label": "齊腳樣本不足",
                "strength": 0.0,
                "active": False,
                "phase": "aligned_unclear",
                "prev_len": prev_len,
                "current_len": current_len,
                "break_rate": round(break_rate, 3),
                "over_rate": round(over_rate, 3),
            }

        edge = min(FOOT_ALIGN_MAX_EDGE, edge)
        strength = FOOT_ALIGN_STRENGTH + strength_bonus + min(0.018, useful_samples * 0.004)
        b, p = _bp_score(target_side, 0.5 + edge)
        return {
            "B": b,
            "P": p,
            "label": label,
            "strength": _clamp(strength, 0.10, 0.270),
            "active": True,
            "foot_alignment": True,
            "phase": phase,
            "target_side": target_side,
            "prev_side": prev_side,
            "prev_len": prev_len,
            "current_side": current_side,
            "current_len": current_len,
            "edge": round(edge, 5),
            "break_count": break_count,
            "over_count": over_count,
            "under_count": under_count,
            "break_rate": round(break_rate, 3),
            "over_rate": round(over_rate, 3),
            "useful_samples": useful_samples,
            "same_target_samples": same_target_samples,
            "room_pattern": room_pattern if room_active else None,
            "road_action": action,
            "bet_mode_hint": "齊腳小注" if phase == "aligned_default_break" else "齊腳順勢",
        }

    # Phase 3: already over-stepped the foot. If the over-step is small, treat it
    # as "not broken yet"; if it keeps going, hand back to dragon/global reversal.
    if current_len > prev_len + tol:
        overflow = current_len - prev_len
        if FOOT_ALIGN_BREAK_GUARD and overflow <= FOOT_ALIGN_OVERFLOW_HANDOFF:
            # Continue only mildly: this confirms no-break, but should not fight a strong reversal forever.
            edge = FOOT_ALIGN_OVER_EDGE + min(0.012, overflow * 0.005) + min(0.010, over_rate * 0.012)
            edge = min(FOOT_ALIGN_MAX_EDGE, edge)
            strength = FOOT_ALIGN_STRENGTH * 0.78 + min(0.018, over_count * 0.004)
            b, p = _bp_score(current_side, 0.5 + edge)
            return {
                "B": b,
                "P": p,
                "label": f"齊腳未破過腳{current_side}{current_len}>{prev_side}{prev_len}｜短延伸續路",
                "strength": _clamp(strength, 0.08, 0.220),
                "active": True,
                "foot_alignment": True,
                "phase": "overfoot_continue",
                "target_side": current_side,
                "prev_side": prev_side,
                "prev_len": prev_len,
                "current_side": current_side,
                "current_len": current_len,
                "overflow": overflow,
                "edge": round(edge, 5),
                "break_rate": round(break_rate, 3),
                "over_rate": round(over_rate, 3),
                "over_count": over_count,
                "road_action": "齊腳未破/短過腳續路",
            }
        return {
            "B": 0.5,
            "P": 0.5,
            "label": "齊腳已過長交回龍判斷",
            "strength": 0.0,
            "active": False,
            "phase": "overfoot_handoff",
            "prev_len": prev_len,
            "current_len": current_len,
        }

    return {"B": 0.5, "P": 0.5, "label": "未見齊腳對應", "strength": 0.0, "active": False}


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

    # 7.7) Room patterns are structured rhythms, not pure chaos. When a clear
    # one-two / two-one / double-chop rhythm is active, reduce chaos so the room
    # model can guide the next fill/turn earlier.
    room_pattern = _room_pattern_score(non_tie)
    if room_pattern.get("active"):
        metrics["room_pattern"] = {
            "phase": room_pattern.get("phase"),
            "pattern_name": room_pattern.get("pattern_name"),
            "target_side": room_pattern.get("target_side"),
            "current_len": room_pattern.get("current_len"),
            "expected_len": room_pattern.get("expected_len"),
            "consistency": room_pattern.get("consistency"),
        }
        score -= ROOM_PATTERN_CHAOS_RELIEF
        reasons.append("房型規律補足/轉邊")

    # 7.8) Foot alignment is also a structured road state. It should reduce chaos
    # when it is guiding a fill / aligned-foot break / no-break continuation.
    foot_alignment = _foot_alignment_score(non_tie)
    if foot_alignment.get("active"):
        metrics["foot_alignment"] = {
            "phase": foot_alignment.get("phase"),
            "target_side": foot_alignment.get("target_side"),
            "prev_len": foot_alignment.get("prev_len"),
            "current_len": foot_alignment.get("current_len"),
            "break_rate": foot_alignment.get("break_rate"),
            "over_rate": foot_alignment.get("over_rate"),
        }
        score -= FOOT_ALIGN_CHAOS_RELIEF
        reasons.append("對應齊腳補腳/破路")

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
        "room_pattern_active": bool(locals().get("room_pattern", {}).get("active")),
        "room_pattern": locals().get("room_pattern", None),
        "foot_alignment_active": bool(locals().get("foot_alignment", {}).get("active")),
        "foot_alignment": locals().get("foot_alignment", None),
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
        if chaos.get("room_pattern_active"):
            road_factor = max(road_factor, 0.78)
            streak_factor = max(streak_factor, 0.60)
            recent_factor = max(recent_factor, 1.28)
            markov_factor = max(markov_factor, 1.08)
        if chaos.get("foot_alignment_active"):
            road_factor = max(road_factor, 0.80)
            streak_factor = max(streak_factor, 0.58)
            recent_factor = max(recent_factor, 1.26)
            markov_factor = max(markov_factor, 1.08)
        weights = {
            "markov": MARKOV_WEIGHT * markov_factor,
            "road": ROAD_WEIGHT * road_factor,
            "streak": STREAK_WEIGHT * streak_factor,
            "balance": BALANCE_WEIGHT * CHAOS_BALANCE_WEIGHT_FACTOR,
            "recent": RECENT_WEIGHT * recent_factor,
        }
    return weights

def _markov_run_bucket(n: int) -> int:
    """Bucket run length for run-state Markov, keeping the state space small."""
    if n <= 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 3
    if n == 4:
        return 4
    if n == 5:
        return 5
    return 6


def _full_markov_score(non_tie: List[str]) -> Dict[str, Any]:
    """
    V18 Full Markov Layer / 完整馬可夫層.

    What it adds beyond the old first-order Markov:
    1) Multi-order Markov: checks suffix states of length 1~5 by default.
       Example: BPB -> next, BBPP -> next, BPPB -> next.
    2) Recency decay: newer repeated states in the same shoe count slightly more.
    3) Run-state Markov: current side + current run length bucket, e.g. B2/P3.

    Safety rule:
    - It only nudges B/P probability unless FULL_MARKOV_FINAL_OVERRIDE=1.
    - Default is no hard override, so Road / Room / Foot / Dragon keep the road feel.
    """
    if not FULL_MARKOV_MODE:
        return {"active": False, "B": 0.5, "P": 0.5, "label": "完整馬可夫關閉", "strength": 0.0}

    n_total = len(non_tie)
    if n_total < FULL_MARKOV_MIN_HISTORY:
        return {"active": False, "B": 0.5, "P": 0.5, "label": "完整馬可夫資料不足", "strength": 0.0}

    candidates: List[Dict[str, Any]] = []
    order_min = max(1, FULL_MARKOV_ORDER_MIN)
    order_max = max(order_min, FULL_MARKOV_ORDER_MAX)
    alpha = max(0.01, FULL_MARKOV_ALPHA)
    decay = _clamp(FULL_MARKOV_DECAY, 0.80, 1.00)

    # 1) Multi-order suffix Markov.
    for k in range(min(order_max, n_total - 1), order_min - 1, -1):
        key = tuple(non_tie[-k:])
        b_score = 0.0
        p_score = 0.0
        sample = 0
        last_match_pos = -1

        # Current tail has no known next hand, so stop before it.
        for i in range(0, n_total - k):
            if tuple(non_tie[i:i + k]) != key:
                continue
            nxt = non_tie[i + k]
            # More recent observations in the same shoe get more influence.
            age = max(0, n_total - (i + k))
            w = decay ** age
            if nxt == "B":
                b_score += w
            elif nxt == "P":
                p_score += w
            sample += 1
            last_match_pos = i

        if sample < FULL_MARKOV_MIN_SAMPLE:
            continue

        weighted_total = b_score + p_score
        if weighted_total <= 0:
            continue

        b_rate = (b_score + alpha) / (weighted_total + 2.0 * alpha)
        target = "B" if b_rate >= 0.5 else "P"
        dominance = abs(b_rate - 0.5) * 2.0
        sample_factor = min(1.0, sample / max(2.0, FULL_MARKOV_MIN_SAMPLE + 3.0))
        order_factor = 0.84 + min(0.22, (k - order_min) * 0.045)
        edge = FULL_MARKOV_EDGE * (0.68 + dominance * 0.42) * (0.70 + sample_factor * 0.30) * order_factor
        edge = min(FULL_MARKOV_MAX_EDGE, max(0.016, edge))
        strength = _clamp(0.105 + dominance * 0.105 + sample_factor * 0.070 + (k - order_min) * 0.012, 0.07, 0.32)
        b, p = _bp_score(target, 0.5 + edge)
        candidates.append({
            "active": True,
            "mode": "multi_order",
            "order": k,
            "B": b,
            "P": p,
            "label": f"完整馬可夫{k}階｜{''.join(key)}→{target}({sample})",
            "strength": strength,
            "target_side": target,
            "edge": round(edge, 5),
            "sample": sample,
            "weighted_sample": round(weighted_total, 4),
            "b_rate": round(b_rate, 4),
            "key": "".join(key),
            "last_match_pos": last_match_pos,
        })

    # 2) Run-state Markov: current side + current run length bucket.
    if FULL_MARKOV_RUN_STATE_MODE:
        current_side, current_len = _streak(non_tie)
        if current_side and current_len >= 1:
            current_bucket = _markov_run_bucket(current_len)
            b_score = 0.0
            p_score = 0.0
            sample = 0
            # j is the prefix length; non_tie[j] is the known next hand.
            for j in range(1, n_total):
                prefix = non_tie[:j]
                s, ln = _streak(prefix)
                if not s:
                    continue
                if s != current_side or _markov_run_bucket(ln) != current_bucket:
                    continue
                nxt = non_tie[j]
                age = max(0, n_total - j)
                w = decay ** age
                if nxt == "B":
                    b_score += w
                elif nxt == "P":
                    p_score += w
                sample += 1

            if sample >= FULL_MARKOV_MIN_SAMPLE:
                weighted_total = b_score + p_score
                if weighted_total > 0:
                    b_rate = (b_score + alpha) / (weighted_total + 2.0 * alpha)
                    target = "B" if b_rate >= 0.5 else "P"
                    dominance = abs(b_rate - 0.5) * 2.0
                    sample_factor = min(1.0, sample / max(2.0, FULL_MARKOV_MIN_SAMPLE + 3.0))
                    edge = FULL_MARKOV_EDGE * (0.66 + dominance * 0.40) * (0.70 + sample_factor * 0.30) * FULL_MARKOV_RUN_WEIGHT
                    edge = min(FULL_MARKOV_MAX_EDGE, max(0.014, edge))
                    strength = _clamp(0.090 + dominance * 0.090 + sample_factor * 0.060, 0.06, 0.26)
                    b, p = _bp_score(target, 0.5 + edge)
                    candidates.append({
                        "active": True,
                        "mode": "run_state",
                        "B": b,
                        "P": p,
                        "label": f"跑長馬可夫｜{current_side}{current_bucket}→{target}({sample})",
                        "strength": strength,
                        "target_side": target,
                        "edge": round(edge, 5),
                        "sample": sample,
                        "weighted_sample": round(weighted_total, 4),
                        "b_rate": round(b_rate, 4),
                        "current_side": current_side,
                        "current_len": current_len,
                        "bucket": current_bucket,
                    })

    if not candidates:
        return {"active": False, "B": 0.5, "P": 0.5, "label": "完整馬可夫未找到相同狀態", "strength": 0.0}

    candidates.sort(key=lambda x: (float(x.get("strength", 0)), float(x.get("edge", 0))), reverse=True)
    best = dict(candidates[0])

    # If the top candidates agree, slightly strengthen the signal.
    agree = [c for c in candidates[:4] if c.get("target_side") == best.get("target_side")]
    if len(agree) >= 2:
        best["secondary_label"] = agree[1].get("label")
        best["strength"] = _clamp(float(best.get("strength", 0)) + 0.015 * min(2, len(agree) - 1), 0.0, 0.34)
        best["edge"] = round(min(FULL_MARKOV_MAX_EDGE, float(best.get("edge", 0)) + 0.004 * min(2, len(agree) - 1)), 5)
        b, p = _bp_score(str(best.get("target_side", "B")), 0.5 + float(best.get("edge", FULL_MARKOV_EDGE)))
        best["B"], best["P"] = b, p

    best["candidates"] = candidates[:4]
    return best


def _sequence_pattern_score(non_tie: List[str]) -> Dict[str, Any]:
    """
    V17 Sequence Pattern Layer / 莊閒排列順序層.

    Purpose:
    - Read the B/P ordering itself, not cross-shoe global history.
    - Catch repeated sequence rhythms like BPBP, BBPP, BPPBPP, BBPBBP.
    - Use current-shoe N-gram tails such as BPB -> P, BPP -> B.

    Design rule:
    - This layer only provides a light B/P probability nudge.
    - It does not hard-overwrite the final recommendation unless
      SEQUENCE_FINAL_OVERRIDE=1, which is not recommended by default.
    """
    if not SEQUENCE_PATTERN_MODE:
        return {"active": False, "B": 0.5, "P": 0.5, "label": "排列層關閉", "strength": 0.0}

    n_total = len(non_tie)
    if n_total < SEQUENCE_MIN_HISTORY:
        return {"active": False, "B": 0.5, "P": 0.5, "label": "排列資料不足", "strength": 0.0}

    candidates: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # 1) Current-shoe N-gram tail lookup.
    #    Example: if the tail is BPB, search earlier BPB occurrences
    #    and learn whether the next hand tended to be B or P.
    # ------------------------------------------------------------
    n_min = max(2, SEQUENCE_NGRAM_MIN)
    n_max = max(n_min, SEQUENCE_NGRAM_MAX)
    for k in range(min(n_max, n_total - 1), n_min - 1, -1):
        key = tuple(non_tie[-k:])
        b_score = 0.0
        p_score = 0.0
        sample = 0
        last_match_pos = -1

        # range stops at n_total-k because current tail has no known next hand.
        for i in range(0, n_total - k):
            if tuple(non_tie[i:i + k]) != key:
                continue
            nxt = non_tie[i + k]
            # Recent matches in the same shoe get slightly more weight, but not too much.
            recency = 0.70 + 0.30 * _safe_div(i + k, max(1, n_total), 0.0)
            if nxt == "B":
                b_score += recency
            elif nxt == "P":
                p_score += recency
            sample += 1
            last_match_pos = i

        if sample < SEQUENCE_MIN_SAMPLE:
            continue

        total = b_score + p_score
        if total <= 0:
            continue
        b_rate = b_score / total
        target = "B" if b_rate >= 0.5 else "P"
        dominance = abs(b_rate - 0.5) * 2.0
        sample_factor = min(1.0, sample / max(2.0, SEQUENCE_MIN_SAMPLE + 2.0))
        k_factor = 0.88 + min(0.18, (k - n_min) * 0.045)
        edge = SEQUENCE_EDGE * (0.72 + dominance * 0.38) * (0.72 + sample_factor * 0.28) * k_factor
        edge = min(SEQUENCE_MAX_EDGE, max(0.018, edge))
        strength = _clamp(0.110 + dominance * 0.090 + sample_factor * 0.060 + (k - n_min) * 0.012, 0.08, 0.30)
        b, p = _bp_score(target, 0.5 + edge)
        candidates.append({
            "active": True,
            "mode": "ngram",
            "B": b,
            "P": p,
            "label": f"排列尾段{k}碼｜{''.join(key)}→{target}({sample})",
            "strength": strength,
            "target_side": target,
            "edge": round(edge, 5),
            "sample": sample,
            "b_rate": round(b_rate, 4),
            "key": "".join(key),
            "last_match_pos": last_match_pos,
        })

    # ------------------------------------------------------------
    # 2) Fixed rhythm detector using the recent 6~14 hands.
    #    Examples:
    #    - BPBPBP       period 2 -> next B/P alternating
    #    - BBPPBBPP     period 4 -> next B
    #    - BPPBPP       period 3 -> next B
    #    - BBPBBP       period 3 -> next B
    # ------------------------------------------------------------
    lookback = max(6, SEQUENCE_LOOKBACK)
    tail = non_tie[-min(n_total, lookback):]
    tail_len = len(tail)
    for period_len in (2, 3, 4, 5):
        if tail_len < period_len * 2:
            continue
        period = tail[-period_len:]
        start = tail_len - period_len
        matches = 0
        total = 0
        for idx, val in enumerate(tail):
            expected = period[(idx - start) % period_len]
            if val == expected:
                matches += 1
            total += 1
        consistency = matches / max(1, total)
        if consistency < 0.72:
            continue

        # Avoid treating a pure same-side long dragon as a sequence rhythm.
        if len(set(period)) == 1:
            continue

        target = period[0]
        edge = SEQUENCE_EDGE + min(0.024, (consistency - 0.72) * 0.070) + min(0.010, (tail_len / max(1, lookback)) * 0.010)
        edge = min(SEQUENCE_MAX_EDGE, max(0.026, edge))
        strength = _clamp(0.180 + (consistency - 0.72) * 0.35 + min(0.050, period_len * 0.008), 0.13, 0.34)
        b, p = _bp_score(target, 0.5 + edge)
        candidates.append({
            "active": True,
            "mode": "rhythm",
            "B": b,
            "P": p,
            "label": f"固定排列{''.join(period)}｜一致{int(consistency * 100)}%→{target}",
            "strength": strength,
            "target_side": target,
            "edge": round(edge, 5),
            "period": "".join(period),
            "period_len": period_len,
            "consistency": round(consistency, 4),
            "tail": "".join(tail),
        })

    if not candidates:
        return {"active": False, "B": 0.5, "P": 0.5, "label": "未見莊閒排列節奏", "strength": 0.0}

    candidates.sort(key=lambda x: (float(x.get("strength", 0)), float(x.get("edge", 0))), reverse=True)
    best = dict(candidates[0])
    if len(candidates) > 1 and candidates[1].get("target_side") == best.get("target_side"):
        best["secondary_label"] = candidates[1].get("label")
        best["strength"] = _clamp(float(best.get("strength", 0)) + 0.015, 0.0, 0.34)
    best["candidates"] = candidates[:3]
    return best

def _road_pattern_score(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 3:
        return {"B": 0.5, "P": 0.5, "label": "資料不足", "strength": 0.0}

    candidates = [
        _dragon_score(non_tie),
        _run_cycle_score(non_tie),
        _room_pattern_score(non_tie),
        _foot_alignment_score(non_tie),
        _sequence_pattern_score(non_tie),
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
    if best.get("active") and best.get("room_pattern"):
        best["room_pattern"] = dict(best)
    if best.get("active") and best.get("foot_alignment"):
        best["foot_alignment"] = dict(best)
    if best.get("active") and str(best.get("mode", "")) in {"ngram", "rhythm"}:
        best["sequence_pattern"] = dict(best)
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


def _side_probs_from_total(target_side: str, edge: float, tie_prob: float) -> Tuple[float, float, float]:
    """Build normalized B/P/T probabilities from a target side and B/P edge."""
    edge = _clamp(edge, 0.0, 0.145)
    bp_total = max(0.001, 1 - tie_prob)
    if target_side == "B":
        b_prob = (0.5 + edge) * bp_total
        p_prob = (0.5 - edge) * bp_total
    else:
        b_prob = (0.5 - edge) * bp_total
        p_prob = (0.5 + edge) * bp_total
    return _normalize_three(b_prob, p_prob, tie_prob)


def _road_type_profile(seq: List[str]) -> Dict[str, Any]:
    """Compact road-state profile used by V14 global shoe context."""
    if len(seq) < 4:
        return {"type": "unknown", "strength": 0.0, "switch_rate": 0.5, "runs": []}
    runs = _runs(seq)
    lengths = [n for _s, n in runs]
    sides = [s for s, _n in runs]
    switch_rate = _window_switch_rate(seq)
    current_side, current_len = _streak(seq)

    # Single chop / alternating road.
    if switch_rate >= 0.74:
        return {
            "type": "single_chop",
            "strength": _clamp(0.50 + (switch_rate - 0.74) * 1.15, 0.0, 0.92),
            "switch_rate": round(switch_rate, 3),
            "runs": lengths[-10:],
            "current": (current_side, current_len),
        }

    # Run-length rhythm detection.
    tail = lengths[-min(len(lengths), 8):]
    best_name = "mixed"
    best_score = 0.0
    best_pattern: List[int] = []
    patterns = [
        ("double_chop", [2, 2]),
        ("one_two", [1, 2]),
        ("two_one", [2, 1]),
    ]
    if len(tail) >= 3:
        for name, pat in patterns:
            for offset in range(len(pat)):
                score = 0.0
                for i, obs in enumerate(tail):
                    exp = pat[(i + offset) % len(pat)]
                    if obs == exp:
                        score += 1.0
                    elif abs(obs - exp) == 1 and obs <= 3:
                        score += 0.35
                consistency = score / max(1, len(tail))
                if consistency > best_score:
                    best_score = consistency
                    best_name = name
                    best_pattern = pat

    if best_score >= 0.66:
        return {
            "type": best_name,
            "strength": round(best_score, 3),
            "pattern": best_pattern,
            "switch_rate": round(switch_rate, 3),
            "runs": lengths[-10:],
            "current": (current_side, current_len),
        }

    # Dragon / connected road.
    if current_len >= 3 or switch_rate <= 0.42:
        long_rate = _safe_div(sum(1 for n in lengths if n >= 3), len(lengths), 0.0)
        return {
            "type": "dragon",
            "strength": _clamp(0.42 + current_len * 0.055 + long_rate * 0.18, 0.0, 0.82),
            "switch_rate": round(switch_rate, 3),
            "runs": lengths[-10:],
            "current": (current_side, current_len),
        }

    return {
        "type": "mixed",
        "strength": _clamp(0.25 + abs(switch_rate - 0.50) * 0.40, 0.0, 0.55),
        "switch_rate": round(switch_rate, 3),
        "runs": lengths[-10:],
        "current": (current_side, current_len),
    }


def _room_target_from_pattern(non_tie: List[str], pattern: List[int]) -> str:
    """Infer the next side for a [1,2], [2,1], or [2,2] room rhythm."""
    runs = _runs(non_tie)
    if not runs or not pattern:
        return ""
    current_side, current_len = runs[-1]
    lengths = [n for _s, n in runs]
    best_offset = 0
    best_score = -1.0
    tail = lengths[-min(len(lengths), 8):]
    for offset in range(len(pattern)):
        score = 0.0
        for i, obs in enumerate(tail):
            exp = pattern[(i + offset) % len(pattern)]
            if obs == exp:
                score += 1.0
            elif i == len(tail) - 1 and obs < exp:
                score += 0.75
            elif abs(obs - exp) == 1 and obs <= 3:
                score += 0.25
        if score > best_score:
            best_score = score
            best_offset = offset
    current_expected = pattern[(len(tail) - 1 + best_offset) % len(pattern)] if tail else pattern[0]
    return current_side if current_len < current_expected else _opposite(current_side)


def _target_side_for_road_type(non_tie: List[str], road_type: str, road: Dict[str, Any]) -> str:
    if not non_tie:
        return ""
    current_side, current_len = _streak(non_tie)
    if current_side not in {"B", "P"}:
        return ""

    # If specialized road models are already active, prefer their target.
    for key in ("room_pattern", "foot_alignment", "mirror_run", "chop_to_dragon"):
        obj = road.get(key) if isinstance(road, dict) else None
        if isinstance(obj, dict) and obj.get("active") and obj.get("target_side") in {"B", "P"}:
            return str(obj.get("target_side"))

    if road_type == "single_chop":
        return _opposite(current_side)
    if road_type == "double_chop":
        return _room_target_from_pattern(non_tie, [2, 2])
    if road_type == "one_two":
        return _room_target_from_pattern(non_tie, [1, 2])
    if road_type == "two_one":
        return _room_target_from_pattern(non_tie, [2, 1])
    if road_type == "dragon":
        # Do not over-follow early dragons; use road/reversal if available.
        reversal = road.get("reversal") if isinstance(road, dict) else None
        if isinstance(reversal, dict) and reversal.get("active") and reversal.get("target_side") in {"B", "P"}:
            return str(reversal.get("target_side"))
        return current_side
    return ""


def _global_shoe_context_guard(
    non_tie: List[str],
    history: List[str],
    b_prob: float,
    p_prob: float,
    tie_prob: float,
    road: Dict[str, Any],
    chaos: Dict[str, Any],
) -> Dict[str, Any]:
    """
    V14 full-shoe context controller.

    It is designed for 60~70-round tables. It splits the current shoe into phases
    and compares early/mid/recent structure so the model can recognize whether the
    latest signal continues the shoe's main road or is only a local fake move.
    """
    base = {
        "active": False,
        "adjusted": False,
        "forced": False,
        "B": b_prob,
        "P": p_prob,
        "T": tie_prob,
        "label": "全靴總控未啟動",
        "score": 0.0,
        "target_side": "",
        "reasons": [],
        "profiles": {},
    }
    if not GLOBAL_SHOE_CONTEXT_MODE or len(non_tie) < GLOBAL_SHOE_MIN_HISTORY:
        return base

    bp_total = max(0.001, 1 - tie_prob)
    pred_side = "B" if b_prob >= p_prob else "P"
    n = len(non_tie)
    split = max(2, GLOBAL_SHOE_PHASE_SPLIT)
    chunk_size = max(1, math.ceil(n / split))
    phases: List[List[str]] = []
    for i in range(split):
        start = i * chunk_size
        end = min(n, (i + 1) * chunk_size)
        if start < n:
            phases.append(non_tie[start:end])
    recent = non_tie[-min(n, max(8, GLOBAL_SHOE_WINDOW)):]

    phase_profiles = [_road_type_profile(x) for x in phases if len(x) >= 4]
    recent_profile = _road_type_profile(recent)
    type_counts = Counter(p["type"] for p in phase_profiles if p.get("type") not in {"unknown", "mixed"})
    main_type = type_counts.most_common(1)[0][0] if type_counts else recent_profile.get("type", "mixed")
    main_count = type_counts[main_type] if main_type in type_counts else 0
    recent_type = str(recent_profile.get("type", "mixed"))

    target_type = recent_type if GLOBAL_SHOE_ALLOW_RECENT_SHIFT and recent_profile.get("strength", 0) >= 0.72 else main_type
    target_side = _target_side_for_road_type(non_tie, target_type, road)
    if target_side not in {"B", "P"}:
        base.update({
            "label": "全靴總控無明確方向",
            "profiles": {"phases": phase_profiles, "recent": recent_profile, "main_type": main_type, "recent_type": recent_type},
        })
        return base

    score = 0.0
    reasons: List[str] = []
    if main_count >= 2:
        score += 0.18 + min(0.12, (main_count - 2) * 0.05)
        reasons.append(f"前中段主路:{main_type}")
    if recent_type == main_type:
        score += 0.20
        reasons.append("近端延續全靴主路")
    elif GLOBAL_SHOE_ALLOW_RECENT_SHIFT and recent_profile.get("strength", 0) >= 0.72:
        score += 0.16
        reasons.append(f"近端換路:{recent_type}")
    score += min(0.20, float(recent_profile.get("strength", 0.0)) * 0.20)

    if chaos.get("strong"):
        score *= 0.76
        reasons.append("強混亂降低全靴強制")
    elif chaos.get("active"):
        score *= 0.88
        reasons.append("破路中保守套用全靴")

    score = _clamp(score, 0.0, 1.0)
    base_profiles = {
        "phases": phase_profiles,
        "recent": recent_profile,
        "main_type": main_type,
        "recent_type": recent_type,
        "target_type": target_type,
    }

    if target_side == pred_side:
        base.update({
            "active": score >= GLOBAL_SHOE_CONTEXT_TRIGGER,
            "adjusted": False,
            "forced": False,
            "score": round(score, 3),
            "target_side": target_side,
            "label": f"全靴總控同向｜{target_type}",
            "reasons": reasons,
            "profiles": base_profiles,
        })
        return base

    if score < GLOBAL_SHOE_CONTEXT_TRIGGER:
        base.update({
            "active": False,
            "adjusted": False,
            "score": round(score, 3),
            "target_side": target_side,
            "label": "全靴總控未達門檻",
            "reasons": reasons,
            "profiles": base_profiles,
        })
        return base

    forced = score >= GLOBAL_SHOE_STRONG_TRIGGER
    edge = GLOBAL_SHOE_CONTEXT_EDGE + max(0.0, score - GLOBAL_SHOE_CONTEXT_TRIGGER) * GLOBAL_SHOE_CONTEXT_WEIGHT
    edge = min(GLOBAL_SHOE_OVERRIDE_EDGE, max(0.020, edge))
    if not forced:
        # Soft mode: pull close to neutral / slight target rather than hard flip.
        old_side_prob = b_prob / bp_total if pred_side == "B" else p_prob / bp_total
        soft_edge = min(edge * 0.70, max(0.018, old_side_prob - 0.50 + edge * 0.35))
        if target_side == "B":
            b_side = max(0.5 + edge * 0.25, 0.5 + soft_edge * 0.35)
            p_side = 1 - b_side
        else:
            p_side = max(0.5 + edge * 0.25, 0.5 + soft_edge * 0.35)
            b_side = 1 - p_side
        new_b, new_p, new_t = _normalize_three(b_side * bp_total, p_side * bp_total, tie_prob)
    else:
        new_b, new_p, new_t = _side_probs_from_total(target_side, edge, tie_prob)

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
        "label": f"全靴總控校準｜{target_type}→{target_side}",
        "reasons": reasons,
        "profiles": base_profiles,
        "edge": round(edge, 5),
        "bet_mode_hint": "全靴小注" if not forced else "全靴校準注",
    }


def _early_dragon_guard(
    non_tie: List[str],
    b_prob: float,
    p_prob: float,
    tie_prob: float,
    road: Dict[str, Any],
) -> Dict[str, Any]:
    """V14 guard for 2~4 hand dragons so the model does not blindly chase every early dragon."""
    base = {
        "active": False,
        "adjusted": False,
        "forced": False,
        "B": b_prob,
        "P": p_prob,
        "T": tie_prob,
        "label": "早期龍保護未啟動",
        "score": 0.0,
        "target_side": "",
        "reasons": [],
    }
    if not EARLY_DRAGON_GUARD or len(non_tie) < 4:
        return base
    current_side, current_len = _streak(non_tie)
    if current_side not in {"B", "P"} or current_len < EARLY_DRAGON_WARN_LEN or current_len > EARLY_DRAGON_MAX_LEN:
        return base

    bp_total = max(0.001, 1 - tie_prob)
    pred_side = "B" if b_prob >= p_prob else "P"
    side_prob = b_prob / bp_total if current_side == "B" else p_prob / bp_total
    opp_side = _opposite(current_side)

    support = False
    support_labels: List[str] = []
    for key, label in (("room_pattern", "房型"), ("foot_alignment", "齊腳"), ("mirror_run", "對稱"), ("chop_to_dragon", "單跳轉龍")):
        obj = road.get(key) if isinstance(road, dict) else None
        if isinstance(obj, dict) and obj.get("active") and obj.get("target_side") == current_side:
            support = True
            support_labels.append(label)

    completed = _runs(non_tie)[:-1]
    recent_lengths = [n for _s, n in completed[-10:]]
    cut_near = sum(1 for n in recent_lengths if n <= current_len)
    exact_cut = sum(1 for n in recent_lengths if n == current_len)
    short_cut_rate = _safe_div(cut_near, len(recent_lengths), 0.0)

    score = 0.18
    reasons = [f"{current_side}{current_len}早期龍"]
    if current_len >= EARLY_DRAGON_ALERT_LEN:
        score += 0.16
        reasons.append("達警戒長度")
    if EARLY_DRAGON_REQUIRE_ROOM_SUPPORT and not support:
        score += 0.20
        reasons.append("無房型/齊腳支撐")
    elif support:
        score -= 0.12
        reasons.append("有" + "+".join(support_labels) + "支撐")
    if exact_cut >= 1:
        score += 0.14
        reasons.append("同長度曾斷")
    if short_cut_rate >= 0.62 and recent_lengths:
        score += 0.10
        reasons.append("近期短切率高")
    if pred_side == current_side and side_prob > EARLY_DRAGON_FOLLOW_CAP:
        score += min(0.12, (side_prob - EARLY_DRAGON_FOLLOW_CAP) * 0.8)
        reasons.append("續龍機率過高")

    score = _clamp(score, 0.0, 1.0)
    if score < EARLY_DRAGON_TRIGGER or pred_side != current_side:
        base.update({
            "active": score >= EARLY_DRAGON_TRIGGER,
            "adjusted": False,
            "score": round(score, 3),
            "target_side": opp_side,
            "label": "早期龍觀察",
            "reasons": reasons,
            "support": support,
        })
        return base

    forced = score >= EARLY_DRAGON_STRONG_TRIGGER
    if forced:
        edge = min(EARLY_DRAGON_BREAK_EDGE, max(0.020, EARLY_DRAGON_BREAK_EDGE * (0.75 + score * 0.35)))
        new_b, new_p, new_t = _side_probs_from_total(opp_side, edge, tie_prob)
        label = f"早期龍防傻跟｜{current_side}{current_len}轉看{opp_side}"
    else:
        # Cap the follow side rather than immediately flipping.
        capped_side = min(side_prob, EARLY_DRAGON_FOLLOW_CAP)
        if current_side == "B":
            new_b = capped_side * bp_total
            new_p = (1 - capped_side) * bp_total
        else:
            new_b = (1 - capped_side) * bp_total
            new_p = capped_side * bp_total
        new_b, new_p, new_t = _normalize_three(new_b, new_p, tie_prob)
        label = f"早期龍降追擊｜{current_side}{current_len}"

    return {
        "active": True,
        "adjusted": True,
        "forced": forced,
        "B": new_b,
        "P": new_p,
        "T": new_t,
        "target_side": opp_side,
        "from_side": current_side,
        "score": round(score, 3),
        "label": label,
        "reasons": reasons,
        "support": support,
        "side_prob_before": round(side_prob, 4),
        "bet_mode_hint": "早期龍小注" if not forced else "早期龍反轉小注",
    }


def _best_completed_room_pattern(lengths: List[int]) -> Dict[str, Any]:
    patterns = [
        ("double_chop", "雙跳", [2, 2]),
        ("one_two", "一房兩廳", [1, 2]),
        ("two_one", "兩房一廳", [2, 1]),
    ]
    best = {"active": False, "consistency": 0.0, "name": "", "pattern": [], "offset": 0}
    if len(lengths) < ROOM_BREAK_REPEAT_MIN:
        return best
    tail = lengths[-min(len(lengths), ROOM_BREAK_LOOKBACK):]
    for key, name, pattern in patterns:
        for offset in range(len(pattern)):
            score = 0.0
            for i, obs in enumerate(tail):
                exp = pattern[(i + offset) % len(pattern)]
                if obs == exp:
                    score += 1.0
                elif abs(obs - exp) == 1 and obs <= 3:
                    score += 0.30
            consistency = score / max(1, len(tail))
            if consistency > best["consistency"]:
                best = {
                    "active": consistency >= ROOM_BREAK_CONSISTENCY,
                    "consistency": consistency,
                    "key": key,
                    "name": name,
                    "pattern": pattern,
                    "offset": offset,
                    "tail": tail,
                }
    return best


def _room_break_to_chop_guard(
    non_tie: List[str],
    b_prob: float,
    p_prob: float,
    tie_prob: float,
    road: Dict[str, Any],
) -> Dict[str, Any]:
    """V14 detector for room rhythm break into single chop / short reversal."""
    base = {
        "active": False,
        "adjusted": False,
        "forced": False,
        "B": b_prob,
        "P": p_prob,
        "T": tie_prob,
        "label": "房型斷點未啟動",
        "score": 0.0,
        "target_side": "",
        "reasons": [],
    }
    if not ROOM_BREAK_TO_CHOP_MODE or len(non_tie) < 8:
        return base
    runs = _runs(non_tie)
    if len(runs) < 5:
        return base
    current_side, current_len = runs[-1]
    lengths = [n for _s, n in runs]
    completed_lengths = lengths[:-1]
    best = _best_completed_room_pattern(completed_lengths)
    if not best.get("active"):
        return base

    pattern = best.get("pattern", [])
    offset = int(best.get("offset", 0))
    expected_current = pattern[(len(best.get("tail", [])) + offset) % len(pattern)] if pattern else 1
    recent = non_tie[-min(len(non_tie), 8):]
    recent_switch = _window_switch_rate(recent)
    pred_side = "B" if b_prob >= p_prob else "P"
    target_side = _opposite(current_side)

    score = 0.0
    reasons: List[str] = [f"{best.get('name')}重複後檢查斷點"]
    if current_len > expected_current:
        score += 0.30
        reasons.append(f"當前{current_side}{current_len}超過預期{expected_current}")
    if recent_switch >= 0.68:
        score += 0.18
        reasons.append("尾段轉單跳")
        target_side = _opposite(current_side)
    if current_len == 1 and expected_current == 2:
        score += 0.14
        reasons.append("補房未完成先觀察反切")
    room_obj = road.get("room_pattern") if isinstance(road, dict) else None
    if isinstance(room_obj, dict) and room_obj.get("active") and room_obj.get("phase") == "turn":
        score += 0.10
        target_side = str(room_obj.get("target_side", target_side)) if room_obj.get("target_side") in {"B", "P"} else target_side
        reasons.append("房型轉邊同步")
    if best.get("consistency", 0.0) >= 0.82 and current_len <= expected_current:
        score *= 0.72
        reasons.append("房型仍穩定不強制破")

    score = _clamp(score, 0.0, 1.0)
    if score < ROOM_BREAK_CONSISTENCY - 0.12:
        base.update({
            "active": False,
            "score": round(score, 3),
            "label": "房型斷點證據不足",
            "target_side": target_side,
            "reasons": reasons,
            "pattern": best,
            "expected_current": expected_current,
            "recent_switch": round(recent_switch, 3),
        })
        return base

    # If target equals current prediction, treat as confirmation rather than adjustment.
    if target_side == pred_side:
        base.update({
            "active": True,
            "adjusted": False,
            "score": round(score, 3),
            "label": f"房型斷點同向｜{best.get('name')}",
            "target_side": target_side,
            "reasons": reasons,
            "pattern": best,
            "expected_current": expected_current,
        })
        return base

    forced = ROOM_BREAK_FINAL_OVERRIDE and score >= ROOM_BREAK_CONSISTENCY
    edge = min(ROOM_BREAK_TO_CHOP_EDGE, max(0.022, ROOM_BREAK_TO_CHOP_EDGE * (0.75 + score * 0.35)))
    if forced:
        new_b, new_p, new_t = _side_probs_from_total(target_side, edge, tie_prob)
    else:
        bp_total = max(0.001, 1 - tie_prob)
        if pred_side == "B":
            b_side = max(0.5 - edge * 0.15, (b_prob / bp_total) - ROOM_BREAK_DAMPEN_ROOM)
            p_side = 1 - b_side
        else:
            p_side = max(0.5 - edge * 0.15, (p_prob / bp_total) - ROOM_BREAK_DAMPEN_ROOM)
            b_side = 1 - p_side
        new_b, new_p, new_t = _normalize_three(b_side * bp_total, p_side * bp_total, tie_prob)

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
        "label": f"房型斷點轉單跳｜{best.get('name')}",
        "reasons": reasons,
        "pattern": best,
        "expected_current": expected_current,
        "recent_switch": round(recent_switch, 3),
        "edge": round(edge, 5),
        "bet_mode_hint": "房型斷點小注",
    }


def _after_tie_safe_guard(history: List[str], b_prob: float, p_prob: float, tie_prob: float) -> Dict[str, Any]:
    base = {
        "active": False,
        "adjusted": False,
        "forced": False,
        "B": b_prob,
        "P": p_prob,
        "T": tie_prob,
        "label": "和局後安全模式未啟動",
        "score": 0.0,
        "tie_gap": _last_tie_gap(history),
    }
    if not AFTER_TIE_SAFE_MODE:
        return base
    tie_gap = _last_tie_gap(history)
    if tie_gap > AFTER_TIE_NO_ENTRY_WINDOW:
        return base
    score = _clamp(1.0 - tie_gap / max(1, AFTER_TIE_NO_ENTRY_WINDOW + 1), 0.20, 1.0)
    base.update({
        "active": True,
        "adjusted": False,
        "forced": AFTER_TIE_FORCE_MINBET,
        "label": f"和局後{tie_gap}手安全模式",
        "score": round(score, 3),
        "tie_gap": tie_gap,
        "conf_cap": AFTER_TIE_CONF_CAP,
        "bet_mode_hint": "最小注/觀望" if AFTER_TIE_FORCE_MINBET else "低信心",
    })
    return base


def _regime_pattern_consistency(lengths: List[int], pattern: List[int]) -> float:
    if not lengths or not pattern:
        return 0.0
    best = 0.0
    max_eval = min(len(lengths), max(4, REGIME_WINDOW))
    tail = lengths[-max_eval:]
    for offset in range(len(pattern)):
        score = 0.0
        for i, val in enumerate(tail):
            exp = pattern[(i + offset) % len(pattern)]
            if val == exp:
                score += 1.0
            elif abs(val - exp) == 1 and val <= 3:
                score += 0.38
        best = max(best, score / max(1, len(tail)))
    return best


def _regime_switch_context(non_tie: List[str], road: Dict[str, Any], sequence: Dict[str, Any], full_markov: Dict[str, Any], chaos: Dict[str, Any]) -> Dict[str, Any]:
    """V19 Regime Switch: identify the current road state before choosing which model to trust."""
    if not REGIME_SWITCH_MODE or len(non_tie) < max(8, REGIME_WINDOW // 2):
        return {"active": False, "regime": "insufficient", "confidence": 0.0, "weight_factors": {}, "module_factors": {}, "label": "路型狀態資料不足"}

    window = max(8, REGIME_WINDOW)
    recent = non_tie[-min(len(non_tie), window):]
    switch_rate = _window_switch_rate(recent)
    runs = _runs(recent)
    run_lengths = [n for _s, n in runs]
    current_side, current_len = _streak(non_tie)
    b_rate = recent.count("B") / len(recent) if recent else 0.5
    side_rate = max(b_rate, 1.0 - b_rate)
    room_12 = _regime_pattern_consistency(run_lengths, [1, 2])
    room_21 = _regime_pattern_consistency(run_lengths, [2, 1])
    double_chop_rate = _safe_div(sum(1 for n in run_lengths[-6:] if n == 2), min(6, len(run_lengths)), 0.0)

    regime = "neutral"
    conf = 0.50
    weight_factors = {"markov": 1.00, "road": 1.00, "streak": 1.00, "balance": 1.00, "recent": 1.00}
    module_factors = {"full_markov": 1.00, "sequence": 1.00, "global_shoe": 1.00, "room": 1.00, "foot": 1.00, "dragon": 1.00}
    reasons: List[str] = []

    if chaos.get("strong"):
        regime = "chaos"
        conf = max(conf, 0.66)
        reasons.append("強亂路")
        weight_factors.update({"markov": 0.92, "road": 0.82, "streak": 0.82, "balance": 1.08, "recent": 1.12})
        module_factors.update({"full_markov": 0.82, "sequence": 0.86, "global_shoe": 0.80})
    elif switch_rate >= 0.74:
        regime = "single_chop"
        conf = max(conf, min(0.84, 0.55 + (switch_rate - 0.70) * 1.35))
        reasons.append("單跳路")
        weight_factors.update({"markov": 1.06, "road": 1.06, "streak": 0.86, "balance": 0.96, "recent": 1.00})
        module_factors.update({"full_markov": 1.08, "sequence": 1.22, "global_shoe": 0.90})
    elif double_chop_rate >= 0.50:
        regime = "double_chop"
        conf = max(conf, 0.60 + min(0.18, (double_chop_rate - 0.50) * 0.45))
        reasons.append("雙跳路")
        weight_factors.update({"markov": 1.02, "road": 1.12, "streak": 0.88, "balance": 0.96, "recent": 0.96})
        module_factors.update({"full_markov": 1.00, "sequence": 1.16, "room": 1.24, "foot": 1.08})
    elif max(room_12, room_21) >= 0.62:
        regime = "room_12" if room_12 >= room_21 else "room_21"
        conf = max(conf, max(room_12, room_21))
        reasons.append("房型節奏")
        weight_factors.update({"markov": 0.96, "road": 1.14, "streak": 0.90, "balance": 0.96, "recent": 0.94})
        module_factors.update({"full_markov": 0.98, "sequence": 1.12, "room": 1.28, "foot": 1.15, "global_shoe": 0.96})
    elif current_len >= DRAGON_STRONG_LEN:
        regime = "dragon"
        conf = max(conf, min(0.82, 0.58 + (current_len - DRAGON_STRONG_LEN) * 0.055))
        reasons.append(f"{current_side}{current_len}龍路")
        weight_factors.update({"markov": 1.00, "road": 1.15, "streak": 1.06, "balance": 0.92, "recent": 0.94})
        module_factors.update({"full_markov": 1.06, "sequence": 0.88, "dragon": 1.22, "global_shoe": 0.98})
    elif side_rate >= 0.64:
        regime = "side_dominance"
        conf = max(conf, min(0.78, 0.55 + (side_rate - 0.60) * 0.90))
        reasons.append("單邊偏重")
        weight_factors.update({"markov": 0.94, "road": 0.96, "streak": 0.88, "balance": 1.20, "recent": 0.90})
        module_factors.update({"full_markov": 0.86, "sequence": 0.92, "global_shoe": 0.82})
    else:
        reasons.append("中性路型")
        conf = max(conf, 0.52)

    # Keep regime switching light. It changes weight, not the final pick.
    if conf < REGIME_MIN_CONFIDENCE:
        weight_factors = {k: 1.0 for k in weight_factors}
        module_factors = {k: 1.0 for k in module_factors}

    max_shift = max(0.0, REGIME_MAX_SHIFT)
    scale = max(0.0, REGIME_WEIGHT) / 0.18 if REGIME_WEIGHT > 0 else 0.0
    if chaos.get("active"):
        scale *= REGIME_CHAOS_RELIEF

    def _limit_factor(v: float) -> float:
        return _clamp(1.0 + (v - 1.0) * scale, 1.0 - max_shift, 1.0 + max_shift)

    weight_factors = {k: _limit_factor(v) for k, v in weight_factors.items()}
    module_factors = {k: _limit_factor(v) for k, v in module_factors.items()}

    return {
        "active": conf >= REGIME_MIN_CONFIDENCE,
        "regime": regime,
        "confidence": round(conf, 3),
        "score": round(conf, 3),
        "label": "路型切換｜" + "+".join(reasons[:2]),
        "switch_rate": round(switch_rate, 3),
        "side_rate": round(side_rate, 3),
        "current_streak": (current_side, current_len),
        "room_12": round(room_12, 3),
        "room_21": round(room_21, 3),
        "double_chop_rate": round(double_chop_rate, 3),
        "weight_factors": weight_factors,
        "module_factors": module_factors,
    }


def _bayes_signal_sample(signal: Dict[str, Any]) -> float:
    vals: List[float] = []
    for key in ("sample", "weighted_sample"):
        try:
            vals.append(float(signal.get(key, 0)))
        except Exception:
            pass
    for c in signal.get("candidates", []) or []:
        if isinstance(c, dict):
            for key in ("sample", "weighted_sample"):
                try:
                    vals.append(float(c.get(key, 0)))
                except Exception:
                    pass
    return max(vals) if vals else 0.0


def _bayes_calibrate_signal(signal: Dict[str, Any], name: str) -> Dict[str, Any]:
    """V19 Bayesian Calibration: shrink low-sample directional edges toward neutral."""
    if not BAYES_CALIBRATION_MODE or not isinstance(signal, dict) or not signal.get("active"):
        return signal
    if name == "full_markov" and not BAYES_APPLY_FULL_MARKOV:
        return signal
    if name == "sequence" and not BAYES_APPLY_SEQUENCE:
        return signal

    sample = _bayes_signal_sample(signal)
    if sample <= 0:
        return signal

    b_side = _clamp(float(signal.get("B", 0.5)), 0.001, 0.999)
    sign = 1.0 if b_side >= 0.5 else -1.0
    raw_edge = abs(b_side - 0.5)
    alpha = max(0.01, BAYES_ALPHA)
    evidence = sample / (sample + 2.0 * alpha)
    if sample < BAYES_MIN_SAMPLE:
        evidence *= max(0.20, sample / max(1.0, float(BAYES_MIN_SAMPLE)))

    shrink = _clamp(BAYES_SHRINK, 0.0, 0.95) * (1.0 - evidence)
    new_edge = min(BAYES_MAX_EDGE, raw_edge * (1.0 - shrink))
    new_b = 0.5 + sign * new_edge
    new_b = _clamp(new_b, 0.001, 0.999)
    out = dict(signal)
    out["B"] = new_b
    out["P"] = 1.0 - new_b
    out["bayes_calibrated"] = True
    out["bayes_sample"] = round(sample, 3)
    out["bayes_evidence"] = round(evidence, 3)
    out["bayes_edge_before"] = round(raw_edge, 5)
    out["bayes_edge_after"] = round(new_edge, 5)
    out["strength"] = _clamp(float(out.get("strength", 0.0)) * (0.86 + 0.14 * evidence), 0.0, 0.34)
    if out.get("label"):
        out["label"] = str(out.get("label")) + "｜貝葉斯校準"
    return out


def _dynamic_pick_accuracy(non_tie: List[str], model_name: str, start_i: int, decay: float) -> Dict[str, Any]:
    total = 0.0
    hit = 0.0
    sample = 0
    details: List[str] = []
    n = len(non_tie)
    for i in range(max(2, start_i), n):
        prefix = non_tie[:i]
        actual = non_tie[i]
        pred = None
        active = True
        try:
            if model_name == "markov":
                s = _transition_prob(prefix)
                pred = "B" if s.get("B", 0.5) >= s.get("P", 0.5) else "P"
            elif model_name == "road":
                s = _road_pattern_score(prefix)
                pred = "B" if s.get("B", 0.5) >= s.get("P", 0.5) else "P"
            elif model_name == "streak":
                s = _streak_score(prefix)
                pred = "B" if s.get("B", 0.5) >= s.get("P", 0.5) else "P"
            elif model_name == "recent":
                s = _recent_score(prefix)
                pred = "B" if s.get("B", 0.5) >= s.get("P", 0.5) else "P"
            elif model_name == "full_markov":
                s = _full_markov_score(prefix)
                active = bool(s.get("active"))
                pred = "B" if s.get("B", 0.5) >= s.get("P", 0.5) else "P"
            elif model_name == "sequence":
                s = _sequence_pattern_score(prefix)
                active = bool(s.get("active"))
                pred = "B" if s.get("B", 0.5) >= s.get("P", 0.5) else "P"
        except Exception:
            active = False
        if pred not in {"B", "P"} or not active:
            continue
        age = max(0, n - 1 - i)
        w = decay ** age
        total += w
        hit += w if pred == actual else 0.0
        sample += 1
        if len(details) < 3:
            details.append(f"{pred}{'✓' if pred == actual else '×'}{actual}")
    acc = hit / total if total > 0 else 0.5
    return {"sample": sample, "weighted_sample": round(total, 3), "accuracy": round(acc, 3), "details": details}


def _dynamic_ensemble_weight_context(non_tie: List[str]) -> Dict[str, Any]:
    """V19 Dynamic Ensemble Weight: recent in-shoe backtest based model weighting."""
    if not DYNAMIC_WEIGHT_MODE or len(non_tie) < max(10, DYNAMIC_WEIGHT_MIN_SAMPLE + 3):
        return {"active": False, "label": "動態權重資料不足", "weight_factors": {}, "module_factors": {}, "scores": {}}

    window = max(DYNAMIC_WEIGHT_MIN_SAMPLE, DYNAMIC_WEIGHT_WINDOW)
    start_i = max(2, len(non_tie) - window)
    decay = _clamp(DYNAMIC_WEIGHT_DECAY, 0.75, 1.0)
    model_names = ["markov", "road", "streak", "recent", "full_markov", "sequence"]
    scores = {m: _dynamic_pick_accuracy(non_tie, m, start_i, decay) for m in model_names}
    max_shift = max(0.0, DYNAMIC_WEIGHT_MAX_SHIFT)

    def factor_for(m: str) -> float:
        sc = scores.get(m, {})
        if int(sc.get("sample", 0)) < DYNAMIC_WEIGHT_MIN_SAMPLE:
            return 1.0
        acc = float(sc.get("accuracy", 0.5))
        # Convert recent fit into a small multiplier. Step is a sensitivity knob.
        raw_shift = (acc - 0.5) * (DYNAMIC_WEIGHT_STEP / 0.025) * 0.36
        return _clamp(1.0 + raw_shift, 1.0 - max_shift, 1.0 + max_shift)

    weight_factors = {
        "markov": factor_for("markov"),
        "road": factor_for("road"),
        "streak": factor_for("streak"),
        "recent": factor_for("recent"),
        "balance": 1.0,
    }
    module_factors = {
        "full_markov": factor_for("full_markov") if DYNAMIC_WEIGHT_APPLY_FULL_MARKOV else 1.0,
        "sequence": factor_for("sequence") if DYNAMIC_WEIGHT_APPLY_SEQUENCE else 1.0,
    }
    active = any(abs(v - 1.0) >= 0.012 for v in list(weight_factors.values()) + list(module_factors.values()))
    best_name = max(scores.keys(), key=lambda k: (float(scores[k].get("accuracy", 0.5)), int(scores[k].get("sample", 0))))
    return {
        "active": active,
        "label": f"動態權重｜近期較適合:{best_name}",
        "score": round(float(scores[best_name].get("accuracy", 0.5)), 3),
        "best_model": best_name,
        "weight_factors": weight_factors,
        "module_factors": module_factors,
        "scores": scores,
    }


def _apply_v19_weight_contexts(weights: Dict[str, float], regime: Dict[str, Any], dynamic: Dict[str, Any]) -> Dict[str, float]:
    out = dict(weights)
    for ctx in (regime, dynamic):
        factors = ctx.get("weight_factors", {}) if isinstance(ctx, dict) else {}
        for k, f in factors.items():
            if k in out:
                out[k] *= _clamp(float(f), 0.70, 1.30)
    # Avoid zero / degenerate sums.
    for k in list(out.keys()):
        out[k] = max(0.001, float(out[k]))
    return out


def _module_factor(regime: Dict[str, Any], dynamic: Dict[str, Any], key: str) -> float:
    f = 1.0
    for ctx in (regime, dynamic):
        factors = ctx.get("module_factors", {}) if isinstance(ctx, dict) else {}
        try:
            f *= float(factors.get(key, 1.0))
        except Exception:
            pass
    return _clamp(f, 0.70, 1.30)

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



def _limit_sequence(seq: List[Any], limit: int) -> Tuple[List[Any], bool]:
    """
    Return the whole sequence when limit <= 0, otherwise return the latest `limit`
    items and a truncation flag. This keeps DeepSeek token usage controlled while
    still allowing full-shoe mode for normal baccarat shoe lengths.
    """
    if limit <= 0 or len(seq) <= limit:
        return list(seq), False
    return list(seq[-limit:]), True


def _window_profile(seq: List[str], windows: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for w in windows:
        tail = seq[-w:] if len(seq) > w else list(seq)
        if not tail:
            out[str(w)] = {
                "len": 0,
                "B": 0,
                "P": 0,
                "B_rate": 0.0,
                "P_rate": 0.0,
                "switch_rate": 0.5,
            }
            continue
        b = tail.count("B")
        p = tail.count("P")
        out[str(w)] = {
            "len": len(tail),
            "B": b,
            "P": p,
            "B_rate": round(b / len(tail), 4),
            "P_rate": round(p / len(tail), 4),
            "switch_rate": round(_window_switch_rate(tail), 4),
            "tail": "".join(tail),
        }
    return out


def _run_profile(run_data: List[Tuple[str, int]]) -> Dict[str, Any]:
    if not run_data:
        return {
            "run_count": 0,
            "lengths": [],
            "sides": [],
            "max_run": 0,
            "median_run": 0,
            "length_counts": {},
            "room_rhythm_tail": [],
        }
    lengths = [n for _s, n in run_data]
    sides = [s for s, _n in run_data]
    length_counts = Counter(min(8, n) for n in lengths)
    return {
        "run_count": len(run_data),
        "lengths": lengths,
        "sides": sides,
        "max_run": max(lengths),
        "median_run": float(median(lengths)) if lengths else 0,
        "length_counts": {str(k): int(v) for k, v in sorted(length_counts.items())},
        "room_rhythm_tail": lengths[-12:],
        "side_rhythm_tail": sides[-12:],
        "current_run": run_data[-1],
        "previous_run": run_data[-2] if len(run_data) >= 2 else None,
    }


def _ai_full_history_payload(history: List[str], non_tie: List[str], run_data: List[Tuple[str, int]]) -> Dict[str, Any]:
    """
    DeepSeek full-shoe payload.

    Earlier versions only sent short tails such as history[-48] and runs[-12].
    This payload lets DeepSeek see the full current shoe when the shoe length is
    within the configured limits, while still adding compact diagnostics so it
    can reason about room patterns, foot alignment, reversals, and broken-road
    states without needing to infer everything from raw text.
    """
    limited_history, history_truncated = _limit_sequence(history, AI_HISTORY_FULL_LIMIT)
    limited_non_tie, non_tie_truncated = _limit_sequence(non_tie, AI_NON_TIE_FULL_LIMIT)
    limited_runs, runs_truncated = _limit_sequence(run_data, AI_RUNS_FULL_LIMIT)
    tail_history, _ = _limit_sequence(history, AI_HISTORY_TAIL_LIMIT)
    tail_runs, _ = _limit_sequence(run_data, AI_RUNS_TAIL_LIMIT)

    payload: Dict[str, Any] = {
        "mode": "full_current_shoe" if AI_FULL_HISTORY_MODE else "tail_summary",
        "history_full": "".join(limited_history),
        "non_tie_full": "".join(limited_non_tie),
        "runs_full": limited_runs,
        "history_full_len": len(history),
        "non_tie_full_len": len(non_tie),
        "runs_full_len": len(run_data),
        "history_full_truncated": history_truncated,
        "non_tie_full_truncated": non_tie_truncated,
        "runs_full_truncated": runs_truncated,
        "history_tail_extended": "".join(tail_history),
        "runs_tail_extended": tail_runs,
    }

    if AI_INCLUDE_PATTERN_DIAGNOSTICS:
        b = non_tie.count("B")
        p = non_tie.count("P")
        t = history.count("T")
        payload["full_shoe_diagnostics"] = {
            "counts": {
                "B": b,
                "P": p,
                "T": t,
                "B_minus_P": b - p,
                "total": len(history),
                "non_tie_total": len(non_tie),
            },
            "rates": {
                "B_rate_non_tie": round(_safe_div(b, len(non_tie), 0.0), 4),
                "P_rate_non_tie": round(_safe_div(p, len(non_tie), 0.0), 4),
                "T_rate_total": round(_safe_div(t, len(history), 0.0), 4),
                "overall_switch_rate": round(_window_switch_rate(non_tie), 4),
            },
            "windows": _window_profile(non_tie, [8, 12, 18, 24, 36, 48, 72]),
            "runs": _run_profile(run_data),
            "last_non_tie_streak": _streak(non_tie),
        }
    return payload


def predict(history: List[str], venue: str = "", room: str = "", shoe_id: str = "") -> Dict[str, Any]:
    history = [x.upper() for x in history if x.upper() in {"B", "P", "T"}]
    non_tie = _last_non_tie(history)

    markov = _transition_prob(non_tie)
    full_markov = _full_markov_score(non_tie)
    road = _road_pattern_score(non_tie)
    sequence = _sequence_pattern_score(non_tie)
    recent = _recent_score(non_tie)
    balance = _balance_score(non_tie)
    streak = _streak_score(non_tie)
    run_data = _runs(non_tie)
    chaos = _chaos_regime_score(non_tie, history)

    # ----- V19 Bayesian / Regime / Dynamic contexts -----
    full_markov = _bayes_calibrate_signal(full_markov, "full_markov")
    sequence = _bayes_calibrate_signal(sequence, "sequence")
    regime_context = _regime_switch_context(non_tie, road, sequence, full_markov, chaos)
    dynamic_weight_context = _dynamic_ensemble_weight_context(non_tie)

    weights = _effective_weights(chaos)
    weights = _apply_v19_weight_contexts(weights, regime_context, dynamic_weight_context)

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
        "history_tail": "".join(history[-min(len(history), AI_HISTORY_TAIL_LIMIT):]) if AI_HISTORY_TAIL_LIMIT > 0 else "".join(history),
        "non_tie_tail": "".join(non_tie[-min(len(non_tie), AI_HISTORY_TAIL_LIMIT):]) if AI_HISTORY_TAIL_LIMIT > 0 else "".join(non_tie),
        "runs_tail": run_data[-min(len(run_data), AI_RUNS_TAIL_LIMIT):] if AI_RUNS_TAIL_LIMIT > 0 else run_data,
        "current_streak": _streak(non_tie),
        "markov": markov,
        "full_markov": full_markov,
        "road": road,
        "sequence": sequence,
        "recent": recent,
        "balance": balance,
        "streak": streak,
        "chaos": chaos,
        "regime_switch": regime_context,
        "dynamic_weight": dynamic_weight_context,
        "effective_weights": {k: round(v, 5) for k, v in weights.items()},
        "local_probs": {"B": round(b_prob, 5), "P": round(p_prob, 5), "T": round(tie_prob, 5)},
        "ai_full_history_payload": _ai_full_history_payload(history, non_tie, run_data) if AI_FULL_HISTORY_MODE else {},
        "ai_payload_config": {
            "full_history_mode": AI_FULL_HISTORY_MODE,
            "history_full_limit": AI_HISTORY_FULL_LIMIT,
            "non_tie_full_limit": AI_NON_TIE_FULL_LIMIT,
            "runs_full_limit": AI_RUNS_FULL_LIMIT,
            "history_tail_limit": AI_HISTORY_TAIL_LIMIT,
            "runs_tail_limit": AI_RUNS_TAIL_LIMIT,
            "include_pattern_diagnostics": AI_INCLUDE_PATTERN_DIAGNOSTICS,
        },
        "global_reversal_config": {
            "enabled": GLOBAL_REVERSAL_MODE,
            "window": GLOBAL_REVERSAL_WINDOW,
            "trigger": GLOBAL_REVERSAL_TRIGGER,
            "after_tie": AFTER_TIE_REVERSAL_MODE,
        },
        "room_pattern_config": {
            "enabled": ROOM_PATTERN_MODE,
            "lookback": ROOM_PATTERN_LOOKBACK,
            "min_consistency": ROOM_PATTERN_MIN_CONSISTENCY,
            "final_override": ROOM_PATTERN_FINAL_OVERRIDE,
        },
        "foot_alignment_config": {
            "enabled": FOOT_ALIGNMENT_MODE,
            "lookback": FOOT_ALIGN_LOOKBACK,
            "break_rate": FOOT_ALIGN_BREAK_RATE,
            "over_rate": FOOT_ALIGN_OVER_RATE,
            "final_override": FOOT_ALIGN_FINAL_OVERRIDE,
        },
        "full_markov_config": {
            "enabled": FULL_MARKOV_MODE,
            "weight": FULL_MARKOV_WEIGHT,
            "order_min": FULL_MARKOV_ORDER_MIN,
            "order_max": FULL_MARKOV_ORDER_MAX,
            "min_sample": FULL_MARKOV_MIN_SAMPLE,
            "run_state_mode": FULL_MARKOV_RUN_STATE_MODE,
            "final_override": FULL_MARKOV_FINAL_OVERRIDE,
        },
        "sequence_pattern_config": {
            "enabled": SEQUENCE_PATTERN_MODE,
            "lookback": SEQUENCE_LOOKBACK,
            "ngram_min": SEQUENCE_NGRAM_MIN,
            "ngram_max": SEQUENCE_NGRAM_MAX,
            "min_sample": SEQUENCE_MIN_SAMPLE,
            "weight": SEQUENCE_WEIGHT,
            "final_override": SEQUENCE_FINAL_OVERRIDE,
        },
        "v19_regime_dynamic_bayes_config": {
            "regime_switch_mode": REGIME_SWITCH_MODE,
            "regime_window": REGIME_WINDOW,
            "regime_weight": REGIME_WEIGHT,
            "bayes_calibration_mode": BAYES_CALIBRATION_MODE,
            "bayes_alpha": BAYES_ALPHA,
            "bayes_shrink": BAYES_SHRINK,
            "dynamic_weight_mode": DYNAMIC_WEIGHT_MODE,
            "dynamic_weight_window": DYNAMIC_WEIGHT_WINDOW,
            "dynamic_weight_max_shift": DYNAMIC_WEIGHT_MAX_SHIFT,
        },
        "v14_context_config": {
            "global_shoe_context_mode": GLOBAL_SHOE_CONTEXT_MODE,
            "global_shoe_window": GLOBAL_SHOE_WINDOW,
            "early_dragon_guard": EARLY_DRAGON_GUARD,
            "room_break_to_chop_mode": ROOM_BREAK_TO_CHOP_MODE,
            "after_tie_safe_mode": AFTER_TIE_SAFE_MODE,
            "after_tie_no_entry_window": AFTER_TIE_NO_ENTRY_WINDOW,
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

    # ----- V18 Full Markov Layer -----
    # Multi-order Markov and run-state Markov only nudge B/P by default.
    full_markov_adjusted = False
    if isinstance(full_markov, dict) and full_markov.get("active"):
        fm_b_side = float(full_markov.get("B", 0.5))
        bp_total = max(0.001, 1 - tie_prob)
        cur_b_side = _clamp(b_prob / bp_total, 0.001, 0.999)
        local_gap = abs(cur_b_side - 0.5) * 2.0
        fm_strength = _clamp(float(full_markov.get("strength", 0.0)), 0.0, 0.34)
        blend = FULL_MARKOV_WEIGHT * (0.68 + min(0.32, fm_strength))
        fm_module_factor = _module_factor(regime_context, dynamic_weight_context, "full_markov")
        blend *= fm_module_factor
        if chaos.get("active"):
            blend *= FULL_MARKOV_CHAOS_FACTOR

        fm_pick = "B" if fm_b_side >= 0.5 else "P"
        cur_pick = "B" if cur_b_side >= 0.5 else "P"
        if fm_pick != cur_pick and local_gap >= FULL_MARKOV_STRONG_LOCAL_GAP and not FULL_MARKOV_FINAL_OVERRIDE:
            blend *= 0.50
        blend = _clamp(blend, 0.0, 0.24)

        new_b_side = cur_b_side * (1 - blend) + fm_b_side * blend
        if FULL_MARKOV_FINAL_OVERRIDE and fm_pick in {"B", "P"}:
            raw_edge = min(FULL_MARKOV_MAX_EDGE, max(FULL_MARKOV_EDGE, float(full_markov.get("edge", FULL_MARKOV_EDGE))))
            if fm_pick == "B":
                new_b_side = 0.5 + raw_edge
            else:
                new_b_side = 0.5 - raw_edge

        b_prob = new_b_side * bp_total
        p_prob = (1 - new_b_side) * bp_total
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)
        full_markov["adjusted"] = True
        full_markov["blend"] = round(blend, 5)
        full_markov["module_factor"] = round(fm_module_factor, 5)
        full_markov["local_gap_before"] = round(local_gap, 5)
        full_markov_adjusted = True
    else:
        full_markov_adjusted = False

    # ----- 強龍保護：當明顯有效長龍時，避免規律層 override 干擾 -----
    current_side, current_len = _streak(non_tie)
    is_strong_dragon = (
        current_len >= DRAGON_STRONG_LEN
        and road.get("road_action") == "續龍"
        and not road.get("reversal", {}).get("active")
    )

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
        and not is_strong_dragon   # 強龍保護
    ):
        target_side = str(road_chop_to_dragon.get("target_side"))
        raw_edge = float(road_chop_to_dragon.get("edge", CHOP_TO_DRAGON_EDGE))
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
        and not is_strong_dragon   # 強龍保護
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

    room_pattern_final_override = False
    road_room_pattern = road.get("room_pattern") if isinstance(road, dict) else None
    if (
        ROOM_PATTERN_FINAL_OVERRIDE
        and isinstance(road_room_pattern, dict)
        and road_room_pattern.get("active")
        and road_room_pattern.get("target_side") in {"B", "P"}
        and road_room_pattern.get("phase") in {"fill", "turn"}
        and not is_strong_dragon   # 強龍保護
    ):
        target_side = str(road_room_pattern.get("target_side"))
        raw_edge = float(road_room_pattern.get("edge", ROOM_PATTERN_EDGE))
        factor = 1.0 if road_room_pattern.get("phase") == "fill" else 0.92
        edge = min(ROOM_PATTERN_OVERRIDE_EDGE, max(0.024, raw_edge * factor))
        bp_total = max(0.001, 1 - tie_prob)
        if target_side == "B":
            b_prob = (0.5 + edge) * bp_total
            p_prob = (0.5 - edge) * bp_total
        else:
            b_prob = (0.5 - edge) * bp_total
            p_prob = (0.5 + edge) * bp_total
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)
        room_pattern_final_override = True

    foot_alignment_final_override = False
    road_foot_alignment = road.get("foot_alignment") if isinstance(road, dict) else None
    if (
        FOOT_ALIGN_FINAL_OVERRIDE
        and isinstance(road_foot_alignment, dict)
        and road_foot_alignment.get("active")
        and road_foot_alignment.get("target_side") in {"B", "P"}
        and road_foot_alignment.get("phase") in {"fill_to_foot", "aligned_break", "aligned_over", "aligned_room", "aligned_default_break", "overfoot_continue"}
        and not is_strong_dragon   # 強龍保護
    ):
        target_side = str(road_foot_alignment.get("target_side"))
        raw_edge = float(road_foot_alignment.get("edge", FOOT_ALIGN_EDGE))
        phase = str(road_foot_alignment.get("phase", ""))
        if phase == "aligned_break":
            factor = 1.00
        elif phase == "aligned_room":
            factor = 0.94
        elif phase == "fill_to_foot":
            factor = 0.88
        elif phase == "aligned_over":
            factor = 0.82
        elif phase == "overfoot_continue":
            factor = 0.72
        else:
            factor = 0.76
        edge = min(FOOT_ALIGN_OVERRIDE_EDGE, max(0.022, raw_edge * factor))
        bp_total = max(0.001, 1 - tie_prob)
        if target_side == "B":
            b_prob = (0.5 + edge) * bp_total
            p_prob = (0.5 - edge) * bp_total
        else:
            b_prob = (0.5 - edge) * bp_total
            p_prob = (0.5 + edge) * bp_total
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)
        foot_alignment_final_override = True

    # ----- V17 Sequence Pattern Layer -----
    # Lightly nudge the B/P side based on direct Banker/Player ordering.
    # It does not hard flip the recommendation unless SEQUENCE_FINAL_OVERRIDE=1.
    sequence_adjusted = False
    if isinstance(sequence, dict) and sequence.get("active"):
        seq_b_side = float(sequence.get("B", 0.5))
        bp_total = max(0.001, 1 - tie_prob)
        cur_b_side = _clamp(b_prob / bp_total, 0.001, 0.999)
        local_gap = abs(cur_b_side - 0.5) * 2.0
        seq_strength = _clamp(float(sequence.get("strength", 0.0)), 0.0, 0.30)
        blend = SEQUENCE_WEIGHT * (0.70 + min(0.30, seq_strength))
        seq_module_factor = _module_factor(regime_context, dynamic_weight_context, "sequence")
        blend *= seq_module_factor
        if chaos.get("active"):
            blend *= SEQUENCE_CHAOS_FACTOR
        # If the current local road is already very clear and sequence points the other way,
        # keep the sequence layer as a small correction only.
        seq_pick = "B" if seq_b_side >= 0.5 else "P"
        cur_pick = "B" if cur_b_side >= 0.5 else "P"
        if seq_pick != cur_pick and local_gap >= SEQUENCE_STRONG_LOCAL_GAP and not SEQUENCE_FINAL_OVERRIDE:
            blend *= 0.45
        blend = _clamp(blend, 0.0, 0.22)

        new_b_side = cur_b_side * (1 - blend) + seq_b_side * blend

        if SEQUENCE_FINAL_OVERRIDE and seq_pick in {"B", "P"}:
            raw_edge = min(SEQUENCE_MAX_EDGE, max(SEQUENCE_EDGE, float(sequence.get("edge", SEQUENCE_EDGE))))
            if seq_pick == "B":
                new_b_side = 0.5 + raw_edge
            else:
                new_b_side = 0.5 - raw_edge

        b_prob = new_b_side * bp_total
        p_prob = (1 - new_b_side) * bp_total
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)
        sequence["adjusted"] = True
        sequence["blend"] = round(blend, 5)
        sequence["module_factor"] = round(seq_module_factor, 5)
        sequence["local_gap_before"] = round(local_gap, 5)
        sequence_adjusted = True
    else:
        sequence_adjusted = False

    # ----- V14 specialized guards -----
    # These layers are applied after the original road overrides and before the
    # majority/global reversal layers so they can correct known weak spots:
    # full-shoe context, early 2~4 hand dragon over-following, room-rhythm breaks,
    # and after-tie safe mode.
    room_break_to_chop = _room_break_to_chop_guard(non_tie, b_prob, p_prob, tie_prob, road)
    if room_break_to_chop.get("active") and room_break_to_chop.get("adjusted"):
        b_prob = float(room_break_to_chop.get("B", b_prob))
        p_prob = float(room_break_to_chop.get("P", p_prob))
        tie_prob = float(room_break_to_chop.get("T", tie_prob))
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)

    early_dragon_guard = _early_dragon_guard(non_tie, b_prob, p_prob, tie_prob, road)
    if early_dragon_guard.get("active") and early_dragon_guard.get("adjusted"):
        b_prob = float(early_dragon_guard.get("B", b_prob))
        p_prob = float(early_dragon_guard.get("P", p_prob))
        tie_prob = float(early_dragon_guard.get("T", tie_prob))
        b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)

    after_tie_safe = _after_tie_safe_guard(history, b_prob, p_prob, tie_prob)

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

    global_shoe_context = _global_shoe_context_guard(non_tie, history, b_prob, p_prob, tie_prob, road, chaos)
    if global_shoe_context.get("active") and global_shoe_context.get("adjusted"):
        b_prob = float(global_shoe_context.get("B", b_prob))
        p_prob = float(global_shoe_context.get("P", p_prob))
        tie_prob = float(global_shoe_context.get("T", tie_prob))
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
    if 'global_shoe_context' in locals() and isinstance(global_shoe_context, dict) and global_shoe_context.get("active"):
        votes.append("B" if global_shoe_context.get("B", 0.5) >= global_shoe_context.get("P", 0.5) else "P")
    if 'early_dragon_guard' in locals() and isinstance(early_dragon_guard, dict) and early_dragon_guard.get("active"):
        votes.append("B" if early_dragon_guard.get("B", 0.5) >= early_dragon_guard.get("P", 0.5) else "P")
    if 'room_break_to_chop' in locals() and isinstance(room_break_to_chop, dict) and room_break_to_chop.get("active"):
        votes.append("B" if room_break_to_chop.get("B", 0.5) >= room_break_to_chop.get("P", 0.5) else "P")
    if isinstance(full_markov, dict) and full_markov.get("active"):
        votes.append("B" if full_markov.get("B", 0.5) >= full_markov.get("P", 0.5) else "P")
    if isinstance(sequence, dict) and sequence.get("active"):
        votes.append("B" if sequence.get("B", 0.5) >= sequence.get("P", 0.5) else "P")
    if 'road_room_pattern' in locals() and isinstance(road_room_pattern, dict) and road_room_pattern.get("active"):
        votes.append("B" if road_room_pattern.get("B", 0.5) >= road_room_pattern.get("P", 0.5) else "P")
    if 'road_foot_alignment' in locals() and isinstance(road_foot_alignment, dict) and road_foot_alignment.get("active"):
        votes.append("B" if road_foot_alignment.get("B", 0.5) >= road_foot_alignment.get("P", 0.5) else "P")
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
    if 'global_shoe_context' in locals() and global_shoe_context.get("active") and global_shoe_context.get("adjusted"):
        cap = GLOBAL_SHOE_STRONG_CONF_CAP if global_shoe_context.get("forced") else GLOBAL_SHOE_CONF_CAP
        conf = min(conf, cap)
        if not chaos.get("active"):
            level = "全靴總控校準" if global_shoe_context.get("forced") else "全靴總控弱訊號"
    if 'early_dragon_guard' in locals() and early_dragon_guard.get("active") and early_dragon_guard.get("adjusted"):
        cap = EARLY_DRAGON_STRONG_CONF_CAP if early_dragon_guard.get("forced") else EARLY_DRAGON_CONF_CAP
        conf = min(conf, cap)
        if not chaos.get("active"):
            level = "早期龍反轉警戒" if early_dragon_guard.get("forced") else "早期龍低追擊"
    if 'room_break_to_chop' in locals() and room_break_to_chop.get("active") and room_break_to_chop.get("adjusted"):
        conf = min(conf, ROOM_BREAK_CONF_CAP)
        if not chaos.get("active"):
            level = "房型斷點警戒"
    if isinstance(full_markov, dict) and full_markov.get("active") and full_markov.get("adjusted"):
        conf = min(conf, FULL_MARKOV_CONF_CAP)
        if not chaos.get("active"):
            level = "完整馬可夫校準"
    if 'after_tie_safe' in locals() and after_tie_safe.get("active"):
        conf = min(conf, AFTER_TIE_CONF_CAP)
        level = "和局後安全觀察"
    reason_parts = [road.get("label", "牌路"), f"模型一致{int(agreement * 100)}%"]
    if chaos.get("active"):
        reason_parts.insert(0, f"{chaos.get('label')}({int(float(chaos.get('score', 0))*100)}%)")
        if LOW_CONFIDENCE_MINBET:
            reason_parts.append("建議最小注")
    if isinstance(regime_context, dict) and regime_context.get("active"):
        reason_parts.insert(0, f"{regime_context.get('label')}({int(float(regime_context.get('score', 0))*100)}%)")
        reason_parts.append(f"路型:{regime_context.get('regime')}")
    if isinstance(dynamic_weight_context, dict) and dynamic_weight_context.get("active"):
        reason_parts.append(f"動態權重:{dynamic_weight_context.get('best_model')}")
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
    if 'global_shoe_context' in locals() and global_shoe_context.get("active"):
        reason_parts.insert(0, f"{global_shoe_context.get('label')}({int(float(global_shoe_context.get('score', 0))*100)}%)")
        if global_shoe_context.get("adjusted"):
            reason_parts.append("全靴前中後段校準")
    if 'early_dragon_guard' in locals() and early_dragon_guard.get("active"):
        reason_parts.insert(0, f"{early_dragon_guard.get('label')}({int(float(early_dragon_guard.get('score', 0))*100)}%)")
        if early_dragon_guard.get("adjusted"):
            reason_parts.append("早期龍防傻跟")
    if 'room_break_to_chop' in locals() and room_break_to_chop.get("active"):
        reason_parts.insert(0, f"{room_break_to_chop.get('label')}({int(float(room_break_to_chop.get('score', 0))*100)}%)")
        if room_break_to_chop.get("adjusted"):
            reason_parts.append("房型斷點轉單跳")
    if isinstance(full_markov, dict) and full_markov.get("active"):
        reason_parts.insert(0, f"{full_markov.get('label')}({int(float(full_markov.get('strength', 0))*100)}%)")
        if full_markov.get("adjusted"):
            reason_parts.append("完整馬可夫校準")
        if full_markov.get("bayes_calibrated"):
            reason_parts.append("馬可夫貝葉斯收斂")
        if full_markov.get("secondary_label"):
            reason_parts.append(f"馬可夫副訊號:{full_markov.get('secondary_label')}")
    if isinstance(sequence, dict) and sequence.get("active"):
        reason_parts.insert(0, f"{sequence.get('label')}({int(float(sequence.get('strength', 0))*100)}%)")
        if sequence.get("adjusted"):
            reason_parts.append("莊閒排列順序校準")
        if sequence.get("bayes_calibrated"):
            reason_parts.append("排列貝葉斯收斂")
    if 'after_tie_safe' in locals() and after_tie_safe.get("active"):
        reason_parts.insert(0, f"{after_tie_safe.get('label')}({int(float(after_tie_safe.get('score', 0))*100)}%)")
        if AFTER_TIE_FORCE_MINBET:
            reason_parts.append("和局後三手最小注/觀望")
    if road.get("road_action"):
        reason_parts.append(f"動作:{road.get('road_action')}")
    if 'reversal_final_override' in locals() and reversal_final_override:
        reason_parts.append("強轉龍校準")
    if 'chop_to_dragon_final_override' in locals() and chop_to_dragon_final_override:
        reason_parts.append("單跳轉龍校準")
    if 'mirror_run_final_override' in locals() and mirror_run_final_override:
        reason_parts.append("對稱龍長校準")
    if 'room_pattern_final_override' in locals() and room_pattern_final_override:
        reason_parts.append("房型規律校準")
    if 'foot_alignment_final_override' in locals() and foot_alignment_final_override:
        reason_parts.append("對應齊腳校準")
    if isinstance(full_markov, dict) and full_markov.get("active") and FULL_MARKOV_FINAL_OVERRIDE:
        reason_parts.append("完整馬可夫硬覆蓋")
    if isinstance(sequence, dict) and sequence.get("active") and SEQUENCE_FINAL_OVERRIDE:
        reason_parts.append("排列順序硬覆蓋")
    if road.get("secondary_label"):
        reason_parts.append(f"副路:{road.get('secondary_label')}")
    if ai_result and ai_result.get("pattern_label"):
        reason_parts.append(f"AI:{ai_result.get('pattern_label')}")
    elif ai_result and ai_result.get("error"):
        reason_parts.append("AI離線改本地判斷")

    return {
        "ok": True,
        "model_version": "V19 Full Markov + Global Shoe + Regime Switch + Dynamic Weight + Bayes",
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
        "regime": regime_context.get("regime") if isinstance(regime_context, dict) else "",
        "dynamic_best_model": dynamic_weight_context.get("best_model") if isinstance(dynamic_weight_context, dict) else "",
        "bet_mode": "最小注" if (
            (chaos.get("active") and LOW_CONFIDENCE_MINBET)
            or (majority_guard.get("active") and majority_guard.get("adjusted"))
            or (global_reversal.get("active") and global_reversal.get("adjusted"))
            or ('global_shoe_context' in locals() and global_shoe_context.get("active") and global_shoe_context.get("adjusted"))
            or ('early_dragon_guard' in locals() and early_dragon_guard.get("active") and early_dragon_guard.get("adjusted"))
            or ('room_break_to_chop' in locals() and room_break_to_chop.get("active") and room_break_to_chop.get("adjusted"))
            or ('after_tie_safe' in locals() and after_tie_safe.get("active") and AFTER_TIE_FORCE_MINBET)
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
            "room_pattern": road.get("room_pattern"),
            "foot_alignment": road.get("foot_alignment"),
            "sequence_pattern": road.get("sequence_pattern"),
            "road_action": road.get("road_action", ""),
        },
        "chaos": chaos,
        "full_markov": full_markov,
        "sequence_pattern": sequence,
        "majority_guard": majority_guard,
        "global_reversal": global_reversal,
        "global_shoe_context": global_shoe_context if 'global_shoe_context' in locals() else None,
        "early_dragon_guard": early_dragon_guard if 'early_dragon_guard' in locals() else None,
        "room_break_to_chop": room_break_to_chop if 'room_break_to_chop' in locals() else None,
        "after_tie_safe": after_tie_safe if 'after_tie_safe' in locals() else None,
        "room_pattern": road.get("room_pattern") if isinstance(road, dict) else None,
        "foot_alignment": road.get("foot_alignment") if isinstance(road, dict) else None,
        "effective_weights": {k: round(v, 4) for k, v in weights.items()},
        "ai_used": bool(ai_result and not ai_result.get("error")),
        "ai_result": ai_result if os.getenv("DEBUG_AI_RESULT", "0") == "1" else None,
        "debug": feature_payload if os.getenv("DEBUG_PREDICTOR", "0") == "1" else None,
    }
