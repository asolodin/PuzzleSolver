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
        "root_prompt_template": (
            "Solve the following puzzle based on the following question. "
            "The question may contain text and photos. For photos, a diagram and a text description may be provided. "
            "For each iteration explain your reasoning step by step. "
            "Question: {question}"
        ),
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
        "recovery_var_candidates": [
            "final_answer",
            "final_response",
            "answer",
            "result",
            "final_decision",
        ],
        "max_depth": 1,
        "max_iterations": 10,
        "stage_budgets": {
            "A": {"max_depth": 1, "max_iterations": 6},
            "B": {"max_depth": 1, "max_iterations": 10},
            "C": {"max_depth": 1, "max_iterations": 14},
        },
        "judge_pass_threshold": 0.95,
        "score_weights": {
            "correctness": 1.0,
            "tokens_penalty": 0.10,
            "latency_penalty": 0.08,
            "finalization_penalty": 0.30,
            "protocol_penalty": 0.24,
            "iteration_penalty": 0.04,
            "recovery_used_penalty": 0.02,
            "recovery_bonus": 0.10,
        },
        "failure_tag_weights": {
            "final_var_missing": 0.35,
            "final_var_name_mismatch": 0.45,
            "literal_final_var_output": 0.60,
            "max_iter_no_finalization": 0.40,
            "loop_repetition": 0.30,
            "protocol_parroting": 0.45,
            "runtime_error": 0.70,
        },
        "norm_scales": {
            "tokens": 12000.0,
            "latency_s": 90.0,
            "iterations": 20.0,
        },
    }


# EVOLVE-BLOCK-END
