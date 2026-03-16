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
            "Remember: your first task is to examine the context and understand the question. "
            "Do not respond with ERROR until you have seen the full question."
        ),
        # Empty string means "use default RLM system prompt", matching quickstart baseline.
        "custom_system_prompt": "",
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
            "iteration_penalty": 0.04,
        },
        "norm_scales": {
            "tokens": 12000.0,
            "latency_s": 90.0,
            "iterations": 20.0,
        },
    }


# EVOLVE-BLOCK-END
