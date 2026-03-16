"""
Outer-loop evaluator for evolving RLM policies with OpenEvolve.

Each candidate policy is evaluated on staged puzzle batches:
- Stage A: tiny subset, cheap budget, early reject
- Stage B: medium subset for survivors
- Stage C: full dev subset for top survivors
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from rlm import RLM
from rlm.core.types import RLMChatCompletion
from rlm.logger import RLMLogger

EVALUATOR_DIR = Path(__file__).resolve().parent
SRC_DIR = EVALUATOR_DIR.parents[1]
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puzzlesolver.shared import (  # noqa: E402
    LLMJudge,
    answer_key_text,
    build_policy_system_prompt,
    detect_finalization_mode,
    format_question,
    question_text_only,
)

PUZZLES_DIR = PROJECT_ROOT / "rlm" / "experiments" / "moscow_puzzles" / "puzzles"
DEFAULT_RLM_LOG_DIR = PROJECT_ROOT / "rlm" / "logs"
RLM_LOG_DIR = Path(os.getenv("RLM_POLICY_LOG_DIR", str(DEFAULT_RLM_LOG_DIR)))
RLM_LOG_DIR.mkdir(parents=True, exist_ok=True)

STAGE_SPECS: list[tuple[str, int]] = [
    ("A", int(os.getenv("RLM_POLICY_STAGE_A_SIZE", "3"))),
    ("B", int(os.getenv("RLM_POLICY_STAGE_B_SIZE", "7"))),
    ("C", int(os.getenv("RLM_POLICY_STAGE_C_SIZE", "12"))),
]

# Early-stop bars to avoid spending full budget on weak candidates.
STAGE_MIN_COMBINED = {
    "A": float(os.getenv("RLM_POLICY_STAGE_A_MIN", "0.30")),
    "B": float(os.getenv("RLM_POLICY_STAGE_B_MIN", "0.40")),
}

# Freeze model control outside evolution. These are intentionally not in policy genome.
ROOT_MODEL_NAME = os.getenv("RLM_POLICY_ROOT_MODEL", os.getenv("RLM_MODEL_NAME", "gpt-5-mini"))
JUDGE_MODEL_NAME = os.getenv(
    "RLM_POLICY_JUDGE_MODEL", os.getenv("RLM_JUDGE_MODEL", "gpt-5.4")
)
SYSTEM_PROMPT_DELTA_MAX_CHARS = int(os.getenv("RLM_POLICY_SYSTEM_PROMPT_DELTA_MAX_CHARS", "1200"))


def _normalize(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return min(1.0, max(0.0, value / scale))


def _avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _load_policy(program_path: str) -> dict[str, Any]:
    module_name = "candidate_policy_module"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load candidate program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "build_policy"):
        raise RuntimeError("Candidate program must define build_policy()")
    policy = module.build_policy()
    if not isinstance(policy, dict):
        raise RuntimeError("build_policy() must return dict")
    return policy


def _collect_run_metrics(result: RLMChatCompletion) -> dict[str, Any]:
    trajectory = result.metadata or {}
    iterations = trajectory.get("iterations", []) if isinstance(trajectory, dict) else []
    finalization = detect_finalization_mode(trajectory if isinstance(trajectory, dict) else {})

    usage = result.usage_summary
    total_calls = sum(s.total_calls for s in usage.model_usage_summaries.values())
    return {
        "iterations": len(iterations),
        "total_llm_calls": total_calls,
        "total_input_tokens": usage.total_input_tokens,
        "total_output_tokens": usage.total_output_tokens,
        "execution_time_s": float(result.execution_time),
        "finalization_mode": finalization["finalization_mode"],
        "finalization_fail": 0.0 if finalization["finalization_mode"] == "FINAL_VAR" else 1.0,
    }


def _run_single_puzzle(
    puzzle: dict[str, Any],
    policy: dict[str, Any],
    stage_name: str,
    judge: LLMJudge,
) -> dict[str, Any]:
    stage_budget = policy.get("stage_budgets", {}).get(stage_name, {})
    max_depth = int(stage_budget.get("max_depth", policy.get("max_depth", 1)))
    max_iterations = int(stage_budget.get("max_iterations", policy.get("max_iterations", 10)))

    prompt_template = str(policy.get("root_prompt_template", "Question:\n{question}"))
    question_payload = json.dumps(format_question(puzzle), indent=2)
    prompt = prompt_template.format(question=question_payload)
    root_prompt_suffix = str(policy.get("root_prompt_suffix", "")).strip() or None
    system_delta = policy.get("custom_system_prompt_delta", policy.get("custom_system_prompt", ""))
    custom_system_prompt = build_policy_system_prompt(
        str(system_delta),
        max_delta_chars=SYSTEM_PROMPT_DELTA_MAX_CHARS,
    )

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": ROOT_MODEL_NAME,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        environment="docker",
        environment_kwargs={
            "execution_timeout_seconds": 20,
            "memory_limit": "4g",
            "cpu_limit": 4,
        },
        max_depth=max_depth,
        max_iterations=max_iterations,
        logger=RLMLogger(log_dir=str(RLM_LOG_DIR), file_name="rlm_policy_eval"),
        verbose=False,
        custom_system_prompt=custom_system_prompt,
    )
    try:
        result: RLMChatCompletion = rlm.completion(prompt, root_prompt_suffix)
    finally:
        rlm.close()

    run_metrics = _collect_run_metrics(result)
    judge_result = judge.judge(
        problem_statement=question_text_only(puzzle),
        gold_answer=answer_key_text(puzzle),
        candidate_answer=result.response,
    )
    return {
        "puzzle_id": puzzle.get("number"),
        "answer": result.response,
        "judge": judge_result,
        "run_metrics": run_metrics,
    }


def _score_records(records: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, float]:
    correctness_values = [float(r["judge"]["correctness"]) for r in records]
    answer_close_values = [float(r["judge"]["answer_closeness"]) for r in records]
    input_tokens = [float(r["run_metrics"]["total_input_tokens"]) for r in records]
    output_tokens = [float(r["run_metrics"]["total_output_tokens"]) for r in records]
    times = [float(r["run_metrics"]["execution_time_s"]) for r in records]
    iterations = [float(r["run_metrics"]["iterations"]) for r in records]
    finalization_fails = [float(r["run_metrics"]["finalization_fail"]) for r in records]

    avg_correctness = _avg(correctness_values)
    avg_answer_closeness = _avg(answer_close_values)
    avg_tokens = _avg([a + b for a, b in zip(input_tokens, output_tokens, strict=True)])
    avg_latency_s = _avg(times)
    avg_iterations = _avg(iterations)
    finalization_fail_rate = _avg(finalization_fails)
    accuracy = _avg([1.0 if r["judge"]["is_correct"] else 0.0 for r in records])

    weights = policy.get("score_weights", {})
    scales = policy.get("norm_scales", {})
    tokens_penalty = float(weights.get("tokens_penalty", 0.10)) * _normalize(
        avg_tokens, float(scales.get("tokens", 12000.0))
    )
    latency_penalty = float(weights.get("latency_penalty", 0.08)) * _normalize(
        avg_latency_s, float(scales.get("latency_s", 90.0))
    )
    iter_penalty = float(weights.get("iteration_penalty", 0.04)) * _normalize(
        avg_iterations, float(scales.get("iterations", 20.0))
    )
    finalization_penalty = float(weights.get("finalization_penalty", 0.30)) * finalization_fail_rate

    combined_score = float(weights.get("correctness", 1.0)) * avg_correctness
    combined_score -= tokens_penalty + latency_penalty + iter_penalty + finalization_penalty
    combined_score = max(0.0, min(1.0, combined_score))

    # Guardrail: if accuracy is very low, cap score to discourage cheap wrong answers.
    if accuracy < 0.5:
        combined_score *= 0.35

    return {
        "combined_score": combined_score,
        "accuracy": accuracy,
        "avg_correctness": avg_correctness,
        "avg_answer_closeness": avg_answer_closeness,
        "avg_tokens": avg_tokens,
        "avg_latency_s": avg_latency_s,
        "avg_iterations": avg_iterations,
        "finalization_fail_rate": finalization_fail_rate,
    }


def _load_puzzles() -> list[dict[str, Any]]:
    puzzle_paths = sorted(PUZZLES_DIR.glob("*.json"))
    puzzles: list[dict[str, Any]] = []
    for path in puzzle_paths:
        with path.open("r", encoding="utf-8") as f:
            puzzles.append(json.load(f))
    return puzzles


def _choose_stage_subset(puzzles: list[dict[str, Any]], stage_size: int) -> list[dict[str, Any]]:
    return puzzles[: min(stage_size, len(puzzles))]


def evaluate(program_path: str) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "avg_correctness": 0.0,
            "avg_answer_closeness": 0.0,
            "avg_tokens": 0.0,
            "avg_latency_s": 9999.0,
            "avg_iterations": 9999.0,
            "finalization_fail_rate": 1.0,
            "stage_count_reached": 0.0,
            "evaluation_time_s": 0.0,
            "error": "OPENAI_API_KEY not set",
        }

    eval_started = time.time()
    policy = _load_policy(program_path)
    judge = LLMJudge(JUDGE_MODEL_NAME)

    puzzles = _load_puzzles()
    if not puzzles:
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "avg_correctness": 0.0,
            "avg_answer_closeness": 0.0,
            "avg_tokens": 0.0,
            "avg_latency_s": 9999.0,
            "avg_iterations": 9999.0,
            "finalization_fail_rate": 1.0,
            "stage_count_reached": 0.0,
            "evaluation_time_s": 0.0,
            "error": f"No puzzles found in {PUZZLES_DIR}",
        }

    all_records: list[dict[str, Any]] = []
    stage_metrics: dict[str, dict[str, float]] = {}
    stopped_stage: str | None = None

    for stage_name, stage_size in STAGE_SPECS:
        subset = _choose_stage_subset(puzzles, stage_size)
        records: list[dict[str, Any]] = []
        for puzzle in subset:
            try:
                records.append(_run_single_puzzle(puzzle, policy, stage_name, judge))
            except Exception as exc:
                records.append(
                    {
                        "puzzle_id": puzzle.get("number"),
                        "answer": "",
                        "judge": {
                            "is_correct": False,
                            "correctness": 0.0,
                            "answer_closeness": 0.0,
                            "failure_type": "other",
                            "reasoning": f"runtime_error: {exc}",
                        },
                        "run_metrics": {
                            "iterations": 0,
                            "total_llm_calls": 0,
                            "total_input_tokens": 0,
                            "total_output_tokens": 0,
                            "execution_time_s": 0.0,
                            "finalization_mode": "error",
                            "finalization_fail": 1.0,
                        },
                    }
                )

        all_records.extend(records)
        stage_score = _score_records(records, policy)
        stage_metrics[stage_name] = stage_score

        if stage_name in STAGE_MIN_COMBINED and stage_score["combined_score"] < STAGE_MIN_COMBINED[stage_name]:
            stopped_stage = stage_name
            break

    final_stage = list(stage_metrics.keys())[-1]
    final_score = stage_metrics[final_stage]
    elapsed = time.time() - eval_started

    metrics = {
        "combined_score": final_score["combined_score"],
        "accuracy": final_score["accuracy"],
        "avg_correctness": final_score["avg_correctness"],
        "avg_answer_closeness": final_score["avg_answer_closeness"],
        "avg_tokens": final_score["avg_tokens"],
        "avg_latency_s": final_score["avg_latency_s"],
        "avg_iterations": final_score["avg_iterations"],
        "finalization_fail_rate": final_score["finalization_fail_rate"],
        # Extra tracking metrics:
        "stage_count_reached": float(len(stage_metrics)),
        "evaluation_time_s": float(elapsed),
    }

    metrics["policy_name"] = str(policy.get("name", ""))
    metrics["root_model"] = ROOT_MODEL_NAME
    metrics["judge_model"] = JUDGE_MODEL_NAME
    metrics["stopped_stage"] = stopped_stage or ""
    metrics["stage_metrics_json"] = json.dumps(stage_metrics)
    metrics["sample_records_json"] = json.dumps(all_records[: min(5, len(all_records))])
    return metrics
