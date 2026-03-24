"""
Minimal evaluator for evolving RLM policies with OpenEvolve.

Design goals:
1) Run RLM on all puzzles.
2) Judge each result.
3) Return aggregate metrics + per-puzzle feedback for mutation.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
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
    format_question,
    question_text_only,
)

LOGGER = logging.getLogger(__name__)

PUZZLES_DIR = PROJECT_ROOT / "rlm" / "experiments" / "moscow_puzzles" / "puzzles"
DEFAULT_RLM_LOG_DIR = PROJECT_ROOT / "rlm" / "logs"
RLM_LOG_DIR = Path(os.getenv("RLM_POLICY_LOG_DIR", str(DEFAULT_RLM_LOG_DIR)))
RLM_LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PUZZLE_OUTPUT_DIR = (
    PROJECT_ROOT / "openevolve" / "rlm_policy" / "openevolve_output" / "puzzle_outputs"
)
PUZZLE_OUTPUT_DIR = Path(os.getenv("RLM_POLICY_PUZZLE_OUTPUT_DIR", str(DEFAULT_PUZZLE_OUTPUT_DIR)))
SAVE_PUZZLE_OUTPUTS = os.getenv("RLM_POLICY_SAVE_PUZZLE_OUTPUTS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
if SAVE_PUZZLE_OUTPUTS:
    PUZZLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROOT_MODEL_NAME = os.getenv("RLM_POLICY_ROOT_MODEL", os.getenv("RLM_MODEL_NAME", "gpt-5-mini"))
JUDGE_MODEL_NAME = os.getenv("RLM_POLICY_JUDGE_MODEL", os.getenv("RLM_JUDGE_MODEL", "gpt-5.4"))
RLM_POLICY_ROOT_TEMPERATURE = os.getenv("RLM_POLICY_ROOT_TEMPERATURE", os.getenv("RLM_TEMPERATURE", ""))
RLM_POLICY_ROOT_TOP_P = os.getenv("RLM_POLICY_ROOT_TOP_P", os.getenv("RLM_TOP_P", ""))
RLM_POLICY_ROOT_SEED = os.getenv("RLM_POLICY_ROOT_SEED", os.getenv("RLM_SEED", ""))
SYSTEM_PROMPT_DELTA_MAX_CHARS = int(os.getenv("RLM_POLICY_SYSTEM_PROMPT_DELTA_MAX_CHARS", "1200"))
RLM_EXECUTION_TIMEOUT_SECONDS = float(os.getenv("RLM_POLICY_EXEC_TIMEOUT_SECONDS", "10"))

DEFAULT_ROOT_PROMPT_TEMPLATE = (
    "Solve the following puzzle based on the following question. "
    "The question may contain text and photos. For photos, a diagram and a text description may be provided. "
    "For each iteration explain your reasoning step by step. "
    "Question: {question}"
)


def _safe_label(value: str, max_len: int = 48) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "policy"
    return cleaned[:max_len]


def _build_root_backend_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_name": ROOT_MODEL_NAME,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    if RLM_POLICY_ROOT_TEMPERATURE.strip():
        kwargs["temperature"] = float(RLM_POLICY_ROOT_TEMPERATURE)
    if RLM_POLICY_ROOT_TOP_P.strip():
        kwargs["top_p"] = float(RLM_POLICY_ROOT_TOP_P)
    if RLM_POLICY_ROOT_SEED.strip():
        kwargs["seed"] = int(RLM_POLICY_ROOT_SEED)
    return kwargs


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


def _load_puzzles() -> list[dict[str, Any]]:
    puzzle_paths = sorted(PUZZLES_DIR.glob("*.json"))
    puzzles: list[dict[str, Any]] = []
    for path in puzzle_paths:
        with path.open("r", encoding="utf-8") as f:
            puzzles.append(json.load(f))
    return puzzles


def _question_for_feedback(puzzle: dict[str, Any]) -> str:
    text_only = question_text_only(puzzle).strip()
    if text_only:
        return text_only
    return json.dumps(format_question(puzzle), ensure_ascii=True)


def _extract_final_turn_response(trajectory: dict[str, Any] | None) -> str:
    if not isinstance(trajectory, dict):
        return ""
    iterations = trajectory.get("iterations", [])
    if not isinstance(iterations, list):
        return ""
    for iteration in reversed(iterations):
        response = str(iteration.get("response", ""))
        if response.strip():
            return response
    return ""


def _collect_run_metrics(result: RLMChatCompletion) -> dict[str, Any]:
    trajectory = result.metadata or {}
    iterations = trajectory.get("iterations", []) if isinstance(trajectory, dict) else []

    usage = result.usage_summary
    total_calls = sum(s.total_calls for s in usage.model_usage_summaries.values())

    return {
        "iterations": len(iterations),
        "total_llm_calls": total_calls,
        "total_input_tokens": usage.total_input_tokens,
        "total_output_tokens": usage.total_output_tokens,
        "execution_time_s": float(result.execution_time),
    }


def _make_eval_output_dir(program_path: str, policy_name: str) -> Path | None:
    if not SAVE_PUZZLE_OUTPUTS:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = uuid.uuid4().hex[:8]
    path = PUZZLE_OUTPUT_DIR / f"{timestamp}_{_safe_label(policy_name)}_{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "program_path": program_path,
        "policy_name": policy_name,
        "root_model": ROOT_MODEL_NAME,
        "judge_model": JUDGE_MODEL_NAME,
    }
    with (path / "evaluation_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def _save_puzzle_record(eval_output_dir: Path | None, puzzle: dict[str, Any], record: dict[str, Any]) -> None:
    if eval_output_dir is None:
        return
    puzzle_number = puzzle.get("number")
    if isinstance(puzzle_number, int):
        file_name = f"puzzle_{puzzle_number:03d}.json"
    else:
        file_name = f"puzzle_{_safe_label(str(puzzle_number or 'unknown'), max_len=16)}.json"
    payload = {
        "puzzle": puzzle,
        "record": record,
        "saved_at": datetime.now().isoformat(),
    }
    with (eval_output_dir / file_name).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_single_puzzle(
    puzzle: dict[str, Any],
    policy: dict[str, Any],
    judge: LLMJudge,
) -> dict[str, Any]:
    prompt_template = str(policy.get("root_prompt_template", DEFAULT_ROOT_PROMPT_TEMPLATE))
    question_payload = json.dumps(format_question(puzzle), indent=2)
    prompt = prompt_template.format(question=question_payload)
    root_prompt_suffix = str(policy.get("root_prompt_suffix", "")).strip() or None

    system_delta = str(policy.get("custom_system_prompt_delta", policy.get("custom_system_prompt", "")))
    custom_system_prompt = build_policy_system_prompt(
        system_delta,
        max_delta_chars=SYSTEM_PROMPT_DELTA_MAX_CHARS,
    )

    max_depth = int(policy.get("max_depth", 1))
    max_iterations = int(policy.get("max_iterations", 10))

    rlm = RLM(
        backend="openai",
        backend_kwargs=_build_root_backend_kwargs(),
        environment="docker",
        environment_kwargs={
            "execution_timeout_seconds": RLM_EXECUTION_TIMEOUT_SECONDS,
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

    question_text = _question_for_feedback(puzzle)
    judge_result = judge.judge(
        problem_statement=question_text,
        gold_answer=answer_key_text(puzzle),
        candidate_answer=result.response,
    )
    run_metrics = _collect_run_metrics(result)
    final_turn_response = _extract_final_turn_response(result.metadata or {})

    return {
        "puzzle_id": puzzle.get("number"),
        "question": question_text,
        "rlm_answer": result.response,
        "rlm_final_turn_response": final_turn_response,
        "judge": judge_result,
        "run_metrics": run_metrics,
    }


def _score_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "avg_correctness": 0.0,
            "avg_answer_closeness": 0.0,
            "avg_tokens": 0.0,
            "avg_latency_s": 9999.0,
            "avg_iterations": 9999.0,
        }

    correctness_values = [float(r["judge"].get("correctness", 0.0)) for r in records]
    closeness_values = [float(r["judge"].get("answer_closeness", 0.0)) for r in records]
    accuracy_values = [1.0 if bool(r["judge"].get("is_correct", False)) else 0.0 for r in records]
    tokens_values = [
        float(r["run_metrics"].get("total_input_tokens", 0.0))
        + float(r["run_metrics"].get("total_output_tokens", 0.0))
        for r in records
    ]
    latency_values = [float(r["run_metrics"].get("execution_time_s", 0.0)) for r in records]
    iterations_values = [float(r["run_metrics"].get("iterations", 0.0)) for r in records]

    avg_correctness = sum(correctness_values) / len(correctness_values)
    avg_answer_closeness = sum(closeness_values) / len(closeness_values)
    accuracy = sum(accuracy_values) / len(accuracy_values)
    avg_tokens = sum(tokens_values) / len(tokens_values)
    avg_latency_s = sum(latency_values) / len(latency_values)
    avg_iterations = sum(iterations_values) / len(iterations_values)

    # Basic objective: judge correctness.
    combined_score = max(0.0, min(1.0, avg_correctness))

    return {
        "combined_score": combined_score,
        "accuracy": accuracy,
        "avg_correctness": avg_correctness,
        "avg_answer_closeness": avg_answer_closeness,
        "avg_tokens": avg_tokens,
        "avg_latency_s": avg_latency_s,
        "avg_iterations": avg_iterations,
    }


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
            "evaluation_time_s": 0.0,
            "error": "OPENAI_API_KEY not set",
        }

    eval_started = time.time()
    policy = _load_policy(program_path)
    judge = LLMJudge(JUDGE_MODEL_NAME)
    eval_output_dir = _make_eval_output_dir(program_path, str(policy.get("name", "policy")))

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
            "evaluation_time_s": 0.0,
            "error": f"No puzzles found in {PUZZLES_DIR}",
        }

    LOGGER.info(
        "Starting simple policy evaluation: policy=%s puzzles=%d output_dir=%s",
        str(policy.get("name", "")),
        len(puzzles),
        str(eval_output_dir) if eval_output_dir else "(disabled)",
    )

    records: list[dict[str, Any]] = []
    for idx, puzzle in enumerate(puzzles, start=1):
        LOGGER.info(
            "Evaluating puzzle %d/%d (id=%s)",
            idx,
            len(puzzles),
            str(puzzle.get("number", "unknown")),
        )
        try:
            record = _run_single_puzzle(puzzle, policy, judge)
        except Exception as exc:
            record = {
                "puzzle_id": puzzle.get("number"),
                "question": _question_for_feedback(puzzle),
                "rlm_answer": "",
                "rlm_final_turn_response": "",
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
                },
            }
        records.append(record)
        _save_puzzle_record(eval_output_dir, puzzle, record)

    score = _score_records(records)
    elapsed = time.time() - eval_started

    metrics: dict[str, Any] = {
        **score,
        "evaluation_time_s": float(elapsed),
        "policy_name": str(policy.get("name", "")),
        "root_model": ROOT_MODEL_NAME,
        "judge_model": JUDGE_MODEL_NAME,
        "puzzle_output_dir": str(eval_output_dir) if eval_output_dir else "",
        "mutator_feedback_json": json.dumps(records, ensure_ascii=True),
    }

    LOGGER.info(
        "Evaluation complete: policy=%s score=%.4f",
        metrics["policy_name"],
        float(metrics["combined_score"]),
    )
    return metrics
