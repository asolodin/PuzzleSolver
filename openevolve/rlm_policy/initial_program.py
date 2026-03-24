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
        "name": "baseline_policy_v1",
        # Limited genome: evaluator keeps budgets/scoring/template fixed.
        "root_prompt_suffix": (
            "Solve the puzzle directly. Do not output meta acknowledgements about instructions, protocol, or context. "
            "When you have a final result, store it in `final_answer` and finalize with FINAL_VAR(final_answer)."
        ),
        # Evolvable delta; evaluator prepends a fixed non-evolvable scaffold.
        "custom_system_prompt_delta": (
            "Internal protocol (do not echo in the answer):\n"
            "1) Inspect context first and solve the puzzle.\n"
            "2) Compute final content in REPL and assign `final_answer`.\n"
            "3) If uncertain about variable names, call SHOW_VARS().\n"
            "4) Finalize only with FINAL_VAR(final_answer).\n"
            "5) Never output literal FINAL_VAR(...) as plain text."
        ),
    }


# EVOLVE-BLOCK-END
