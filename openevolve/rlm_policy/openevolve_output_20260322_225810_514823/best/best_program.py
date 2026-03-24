"""
Evolvable RLM policy program for OpenEvolve.

OpenEvolve mutates only the EVOLVE block. The evaluator imports this file and
calls `build_policy()` to get a candidate policy.
"""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
def build_policy() -> dict[str, Any]:
    return {
        "name": "baseline_policy_v1_deliverable_search",
        # Limited genome: evaluator keeps budgets/scoring/template fixed.
        "root_prompt_suffix": (
            "Solve the puzzle directly (no meta about instructions/protocol). "
            "First identify the required deliverable (number / proof / strategy rule / explicit move list) and provide it explicitly. "
            "If conventions are ambiguous, pick the one most consistent with the wording and commit to a single answer. "
            "Verify by substitution or a small simulation/search when feasible. "
            "Store the result in `final_answer` and end with FINAL_VAR(final_answer) as the last line."
        ),
        # Evolvable delta; evaluator prepends a fixed non-evolvable scaffold.
        "custom_system_prompt_delta": (
            "Internal (do not echo):\n"
            "1) Decide deliverable type; if a move sequence is requested, output an actual move list (use search like IDA*/A* if needed).\n"
            "2) Resolve ambiguity in conventions from wording; do not present multiple interpretations unless asked.\n"
            "3) For games with unusual win conditions (e.g., parity/last-move variants), solve via minimax/DP over the right state, then extract a correct human rule and verify by simulation.\n"
            "4) For 'is there another way?' prompts, look for a constructive method that cancels unknown bias via reversal/repetition/aux material.\n"
            "5) Keep output tight (no long traces); end with FINAL_VAR(final_answer) as the final line."
        ),
    }


# EVOLVE-BLOCK-END
