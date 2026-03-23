"""
Outer-loop evaluator for evolving RLM policies with OpenEvolve.

Each candidate policy is evaluated on staged puzzle batches:
- Stage A: tiny subset, cheap budget, early reject
- Stage B: medium subset for survivors
- Stage C: full dev subset for top survivors
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import re
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
    detect_finalization_mode,
    format_question,
    question_text_only,
)

PUZZLES_DIR = PROJECT_ROOT / "rlm" / "experiments" / "moscow_puzzles" / "puzzles"
DEFAULT_RLM_LOG_DIR = PROJECT_ROOT / "rlm" / "logs"
RLM_LOG_DIR = Path(os.getenv("RLM_POLICY_LOG_DIR", str(DEFAULT_RLM_LOG_DIR)))
RLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PUZZLE_OUTPUT_DIR = (
    PROJECT_ROOT / "openevolve" / "rlm_policy" / "openevolve_output" / "puzzle_outputs"
)
PUZZLE_OUTPUT_DIR = Path(os.getenv("RLM_POLICY_PUZZLE_OUTPUT_DIR", str(DEFAULT_PUZZLE_OUTPUT_DIR)))
SAVE_PUZZLE_OUTPUTS = os.getenv("RLM_POLICY_SAVE_PUZZLE_OUTPUTS", "1").lower() not in {
    "0",
    "false",
    "no",
}
if SAVE_PUZZLE_OUTPUTS:
    PUZZLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_ALL_STAGE_SPECS: list[tuple[str, int]] = [
    ("A", int(os.getenv("RLM_POLICY_STAGE_A_SIZE", "3"))),
    ("B", int(os.getenv("RLM_POLICY_STAGE_B_SIZE", "7"))),
    ("C", int(os.getenv("RLM_POLICY_STAGE_C_SIZE", "12"))),
]
_ACTIVE_STAGE_NAMES = {
    s.strip().upper()
    for s in os.getenv("RLM_POLICY_ACTIVE_STAGES", "A,B,C").split(",")
    if s.strip()
}
STAGE_SPECS: list[tuple[str, int]] = [
    (name, size) for name, size in _ALL_STAGE_SPECS if name in _ACTIVE_STAGE_NAMES
]
if not STAGE_SPECS:
    STAGE_SPECS = [("A", _ALL_STAGE_SPECS[0][1])]

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
RLM_POLICY_ROOT_TEMPERATURE = os.getenv(
    "RLM_POLICY_ROOT_TEMPERATURE", os.getenv("RLM_TEMPERATURE", "")
)
RLM_POLICY_ROOT_TOP_P = os.getenv("RLM_POLICY_ROOT_TOP_P", os.getenv("RLM_TOP_P", ""))
RLM_POLICY_ROOT_SEED = os.getenv("RLM_POLICY_ROOT_SEED", os.getenv("RLM_SEED", ""))
SYSTEM_PROMPT_DELTA_MAX_CHARS = int(os.getenv("RLM_POLICY_SYSTEM_PROMPT_DELTA_MAX_CHARS", "1200"))
FINAL_VAR_LITERAL_PATTERN = re.compile(r"^\s*FINAL_VAR\((.*?)\)\s*$", re.DOTALL)
FINAL_VAR_LINE_PATTERN = re.compile(r"^\s*FINAL_VAR\((.*?)\)\s*$", re.MULTILINE | re.DOTALL)

DEFAULT_RECOVERY_VAR_CANDIDATES = [
    "final_answer",
    "final_response",
    "answer",
    "result",
    "final_decision",
]

DEFAULT_FAILURE_TAG_WEIGHTS = {
    "final_var_missing": 0.35,
    "final_var_name_mismatch": 0.45,
    "literal_final_var_output": 0.60,
    "max_iter_no_finalization": 0.40,
    "loop_repetition": 0.30,
    "protocol_parroting": 0.45,
}

LOGGER = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


# Keep policy search focused on a small, interpretable surface.
LIMITED_GENOME_MODE = _env_flag("RLM_POLICY_LIMITED_GENOME", True)
HARD_GATE_PROTOCOL = _env_flag("RLM_POLICY_HARD_GATE_PROTOCOL", False)
HARD_GATE_SCORE_CAP = float(os.getenv("RLM_POLICY_HARD_GATE_SCORE_CAP", "0.02"))
HARD_GATE_FAILURE_TAGS = {"protocol_parroting", "literal_final_var_output"}

FIXED_ROOT_PROMPT_TEMPLATE = (
    "Solve the following puzzle based on the following question. "
    "The question may contain text and photos. For photos, a diagram and a text description may be provided. "
    "For each iteration explain your reasoning step by step. "
    "Question: {question}"
)
FIXED_STAGE_BUDGETS = {
    "A": {"max_depth": 1, "max_iterations": int(os.getenv("RLM_POLICY_STAGE_A_MAX_ITER", "6"))},
    "B": {"max_depth": 1, "max_iterations": int(os.getenv("RLM_POLICY_STAGE_B_MAX_ITER", "10"))},
    "C": {"max_depth": 1, "max_iterations": int(os.getenv("RLM_POLICY_STAGE_C_MAX_ITER", "14"))},
}
FIXED_SCORE_WEIGHTS = {
    "correctness": 1.0,
    "tokens_penalty": 0.10,
    "latency_penalty": 0.08,
    "finalization_penalty": 0.30,
    "protocol_penalty": 0.24,
    "iteration_penalty": 0.04,
    "recovery_used_penalty": 0.02,
    "recovery_bonus": 0.10,
}
FIXED_NORM_SCALES = {
    "tokens": 12000.0,
    "latency_s": 90.0,
    "iterations": 20.0,
}
FIXED_FAILURE_TAG_WEIGHTS = {
    **DEFAULT_FAILURE_TAG_WEIGHTS,
    "runtime_error": 0.70,
}
HINT_BY_FAILURE_TAG = {
    "protocol_parroting": "Stop meta acknowledgements. Output only puzzle-solution content.",
    "literal_final_var_output": "Do not output literal FINAL_VAR(...). Create `final_answer` in REPL first.",
    "final_var_name_mismatch": "Use one stable variable name: `final_answer`, then call FINAL_VAR(final_answer).",
    "max_iter_no_finalization": "Finalize earlier. Solve and call FINAL_VAR(final_answer) within the iteration budget.",
    "final_var_missing": "Always finalize with FINAL_VAR(final_answer) once answer is computed.",
    "loop_repetition": "Avoid repeated planning loops. Switch to executing REPL steps quickly.",
    "runtime_error": "Reduce brittle REPL code and keep finalization path simple.",
}


def _normalize(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return min(1.0, max(0.0, value / scale))


def _avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _build_root_backend_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_name": ROOT_MODEL_NAME,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    if RLM_POLICY_ROOT_TEMPERATURE.strip() != "":
        kwargs["temperature"] = float(RLM_POLICY_ROOT_TEMPERATURE)
    if RLM_POLICY_ROOT_TOP_P.strip() != "":
        kwargs["top_p"] = float(RLM_POLICY_ROOT_TOP_P)
    if RLM_POLICY_ROOT_SEED.strip() != "":
        kwargs["seed"] = int(RLM_POLICY_ROOT_SEED)
    return kwargs


def _extract_final_var_name(text: str) -> str | None:
    match = FINAL_VAR_LINE_PATTERN.search(str(text))
    if not match:
        return None
    return match.group(1).strip().strip("'\"")


def _extract_latest_locals(trajectory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(trajectory, dict):
        return {}
    iterations = trajectory.get("iterations", [])
    if not isinstance(iterations, list):
        return {}

    for iteration in reversed(iterations):
        for block in reversed(iteration.get("code_blocks", [])):
            result = block.get("result", {})
            locals_dict = result.get("locals", {})
            if isinstance(locals_dict, dict) and locals_dict:
                return locals_dict
    return {}


def _coerce_local_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            try:
                parsed = ast.literal_eval(text)
                return str(parsed).strip()
            except Exception:
                return text
        return text
    return str(value).strip()


def _detect_loop_repetition(iterations: list[dict[str, Any]]) -> tuple[bool, int]:
    max_consecutive = 1
    current = 1
    previous = None

    for iteration in iterations:
        response = " ".join(str(iteration.get("response", "")).split())
        if not response:
            continue
        if previous == response:
            current += 1
        else:
            previous = response
            current = 1
        if current > max_consecutive:
            max_consecutive = current

    return max_consecutive >= 2, max_consecutive


def _recover_candidate_answer(
    latest_locals: dict[str, Any],
    recovery_var_candidates: list[str],
) -> tuple[str | None, str]:
    for var_name in recovery_var_candidates:
        if var_name in latest_locals:
            value = _coerce_local_value(latest_locals[var_name])
            if value and not value.lower().startswith("error: variable"):
                return value, var_name

    fallback_keys = [
        key
        for key in latest_locals.keys()
        if any(token in key.lower() for token in ("final", "answer", "result"))
    ]
    for key in fallback_keys:
        value = _coerce_local_value(latest_locals[key])
        if value and not value.lower().startswith("error: variable"):
            return value, key

    return None, ""


def _protocol_failure_tags(
    trajectory: dict[str, Any] | None,
    candidate_answer: str,
    max_iterations: int,
    finalization_mode: str,
    latest_locals: dict[str, Any],
    recovery_var_candidates: list[str],
) -> tuple[list[str], str]:
    tags: list[str] = []
    iterations = trajectory.get("iterations", []) if isinstance(trajectory, dict) else []
    final_var_call_name = _extract_final_var_name(candidate_answer) or ""
    answer_lower = str(candidate_answer).strip().lower()

    if finalization_mode != "FINAL_VAR":
        tags.append("final_var_missing")

    if FINAL_VAR_LITERAL_PATTERN.fullmatch(str(candidate_answer).strip()):
        tags.append("literal_final_var_output")

    if max_iterations > 0 and len(iterations) >= max_iterations and finalization_mode != "FINAL_VAR":
        tags.append("max_iter_no_finalization")

    loop_repetition, _ = _detect_loop_repetition(iterations if isinstance(iterations, list) else [])
    if loop_repetition:
        tags.append("loop_repetition")

    if (
        "finalization protocol" in answer_lower
        or "i have examined the context and understand the instructions" in answer_lower
        or "i will follow the finalization protocol" in answer_lower
    ):
        tags.append("protocol_parroting")

    if final_var_call_name and final_var_call_name not in latest_locals and latest_locals:
        if any(k in latest_locals for k in recovery_var_candidates):
            tags.append("final_var_name_mismatch")

    return sorted(set(tags)), final_var_call_name


def _protocol_failure_score(tags: list[str], failure_tag_weights: dict[str, float]) -> float:
    return min(1.0, sum(float(failure_tag_weights.get(tag, 0.15)) for tag in tags))


def _safe_label(value: str, max_len: int = 48) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "policy"
    return cleaned[:max_len]


def _truncate_text(value: str, max_chars: int) -> str:
    collapsed = " ".join(str(value).split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


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
        "stage_specs": STAGE_SPECS,
        "stage_min_combined": STAGE_MIN_COMBINED,
    }
    with (path / "evaluation_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def _save_puzzle_record(
    eval_output_dir: Path | None,
    stage_name: str,
    puzzle: dict[str, Any],
    record: dict[str, Any],
) -> None:
    if eval_output_dir is None:
        return

    stage_dir = eval_output_dir / f"stage_{stage_name}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    puzzle_number = puzzle.get("number")
    if isinstance(puzzle_number, int):
        file_name = f"puzzle_{puzzle_number:03d}.json"
    else:
        file_name = f"puzzle_{_safe_label(str(puzzle_number or 'unknown'), max_len=16)}.json"

    payload = {
        "stage": stage_name,
        "puzzle": puzzle,
        "record": record,
        "saved_at": datetime.now().isoformat(),
    }
    with (stage_dir / file_name).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


def _apply_policy_constraints(policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    constrained = dict(policy)
    evolvable_fields = [
        "name",
        "root_prompt_suffix",
        "custom_system_prompt_delta",
        "recovery_var_candidates",
    ]
    locked_fields: list[str] = []

    if LIMITED_GENOME_MODE:
        locked_fields = [
            "root_prompt_template",
            "max_depth",
            "max_iterations",
            "stage_budgets",
            "judge_pass_threshold",
            "score_weights",
            "failure_tag_weights",
            "norm_scales",
        ]
        constrained["root_prompt_template"] = FIXED_ROOT_PROMPT_TEMPLATE
        constrained["max_depth"] = 1
        constrained["max_iterations"] = int(FIXED_STAGE_BUDGETS["A"]["max_iterations"])
        constrained["stage_budgets"] = FIXED_STAGE_BUDGETS
        constrained["judge_pass_threshold"] = 0.95
        constrained["score_weights"] = FIXED_SCORE_WEIGHTS
        constrained["failure_tag_weights"] = FIXED_FAILURE_TAG_WEIGHTS
        constrained["norm_scales"] = FIXED_NORM_SCALES

    return constrained, {
        "limited_genome_mode": LIMITED_GENOME_MODE,
        "locked_fields": locked_fields,
        "evolvable_fields": evolvable_fields,
    }


def _derive_primary_hint(failure_tag_counts: dict[str, int]) -> tuple[str, str]:
    if not failure_tag_counts:
        return (
            "none",
            "No dominant protocol failure detected. Improve reasoning quality and answer correctness.",
        )

    priority = [
        "protocol_parroting",
        "literal_final_var_output",
        "final_var_name_mismatch",
        "max_iter_no_finalization",
        "final_var_missing",
        "loop_repetition",
        "runtime_error",
    ]
    top_tag = sorted(
        failure_tag_counts.items(),
        key=lambda item: (
            -item[1],
            priority.index(item[0]) if item[0] in priority else len(priority),
            item[0],
        ),
    )[0][0]
    hint = HINT_BY_FAILURE_TAG.get(
        top_tag,
        "Focus on stable REPL finalization and avoid protocol-related output errors.",
    )
    return top_tag, f"{hint} (dominant tag: {top_tag})"


def _collect_run_metrics(result: RLMChatCompletion) -> dict[str, Any]:
    trajectory = result.metadata or {}
    iterations = trajectory.get("iterations", []) if isinstance(trajectory, dict) else []
    finalization = detect_finalization_mode(trajectory if isinstance(trajectory, dict) else {})

    usage = result.usage_summary
    total_calls = sum(s.total_calls for s in usage.model_usage_summaries.values())
    loop_repetition, max_consecutive_repeats = _detect_loop_repetition(
        iterations if isinstance(iterations, list) else []
    )

    return {
        "iterations": len(iterations),
        "total_llm_calls": total_calls,
        "total_input_tokens": usage.total_input_tokens,
        "total_output_tokens": usage.total_output_tokens,
        "execution_time_s": float(result.execution_time),
        "finalization_mode": finalization["finalization_mode"],
        "finalization_fail": 0.0 if finalization["finalization_mode"] == "FINAL_VAR" else 1.0,
        "loop_repetition": 1.0 if loop_repetition else 0.0,
        "max_consecutive_repeats": float(max_consecutive_repeats),
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
    recovery_var_candidates = policy.get("recovery_var_candidates", DEFAULT_RECOVERY_VAR_CANDIDATES)
    if not isinstance(recovery_var_candidates, list) or not recovery_var_candidates:
        recovery_var_candidates = DEFAULT_RECOVERY_VAR_CANDIDATES

    rlm = RLM(
        backend="openai",
        backend_kwargs=_build_root_backend_kwargs(),
        environment="docker",
        environment_kwargs={
            "execution_timeout_seconds": 10,
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

    trajectory = result.metadata or {}
    latest_locals = _extract_latest_locals(trajectory if isinstance(trajectory, dict) else {})
    run_metrics = _collect_run_metrics(result)
    failure_tags, final_var_call_name = _protocol_failure_tags(
        trajectory=trajectory if isinstance(trajectory, dict) else {},
        candidate_answer=result.response,
        max_iterations=max_iterations,
        finalization_mode=str(run_metrics.get("finalization_mode", "")),
        latest_locals=latest_locals,
        recovery_var_candidates=[str(v) for v in recovery_var_candidates],
    )

    question_for_judge = question_text_only(puzzle).strip() or json.dumps(
        format_question(puzzle),
        ensure_ascii=True,
    )
    judge_raw = judge.judge(
        problem_statement=question_for_judge,
        gold_answer=answer_key_text(puzzle),
        candidate_answer=result.response,
    )

    recovered_answer, recovered_from_var = _recover_candidate_answer(
        latest_locals=latest_locals,
        recovery_var_candidates=[str(v) for v in recovery_var_candidates],
    )
    judge_recovered: dict[str, Any] | None = None
    recovery_attempted = 0.0
    recovery_used = 0.0
    recovery_success = 0.0
    effective_answer = result.response
    effective_judge = judge_raw

    should_attempt_recovery = (
        bool(failure_tags)
        or str(run_metrics.get("finalization_mode")) != "FINAL_VAR"
        or FINAL_VAR_LITERAL_PATTERN.fullmatch(str(result.response).strip()) is not None
    )
    if should_attempt_recovery and recovered_answer:
        recovery_attempted = 1.0
        judge_recovered = judge.judge(
            problem_statement=question_for_judge,
            gold_answer=answer_key_text(puzzle),
            candidate_answer=recovered_answer,
        )
        raw_correctness = float(judge_raw.get("correctness", 0.0))
        recovered_correctness = float(judge_recovered.get("correctness", 0.0))
        if recovered_correctness > raw_correctness:
            effective_answer = recovered_answer
            effective_judge = judge_recovered
            recovery_used = 1.0
        if (not bool(judge_raw.get("is_correct"))) and bool(judge_recovered.get("is_correct")):
            recovery_success = 1.0

    run_metrics["final_var_call_name"] = final_var_call_name
    run_metrics["failure_tags"] = failure_tags
    run_metrics["failure_tag_count"] = float(len(failure_tags))
    run_metrics["latest_locals_keys"] = sorted(list(latest_locals.keys()))[:40]
    run_metrics["recovery_attempted"] = recovery_attempted
    run_metrics["recovery_used"] = recovery_used
    run_metrics["recovery_success"] = recovery_success
    run_metrics["recovered_from_var"] = recovered_from_var

    return {
        "puzzle_id": puzzle.get("number"),
        "question_text": question_for_judge,
        "answer": effective_answer,
        "answer_raw": result.response,
        "answer_recovered": recovered_answer or "",
        "judge": effective_judge,
        "judge_raw": judge_raw,
        "judge_recovered": judge_recovered or {},
        "run_metrics": run_metrics,
    }


def _score_records(records: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    correctness_values = [float(r["judge"]["correctness"]) for r in records]
    answer_close_values = [float(r["judge"]["answer_closeness"]) for r in records]
    input_tokens = [float(r["run_metrics"]["total_input_tokens"]) for r in records]
    output_tokens = [float(r["run_metrics"]["total_output_tokens"]) for r in records]
    times = [float(r["run_metrics"]["execution_time_s"]) for r in records]
    iterations = [float(r["run_metrics"]["iterations"]) for r in records]
    finalization_fails = [float(r["run_metrics"]["finalization_fail"]) for r in records]
    recovery_attempted_values = [float(r["run_metrics"].get("recovery_attempted", 0.0)) for r in records]
    recovery_used_values = [float(r["run_metrics"].get("recovery_used", 0.0)) for r in records]
    recovery_success_values = [float(r["run_metrics"].get("recovery_success", 0.0)) for r in records]
    failure_tag_values = [r["run_metrics"].get("failure_tags", []) for r in records]

    avg_correctness = _avg(correctness_values)
    avg_answer_closeness = _avg(answer_close_values)
    avg_tokens = _avg([a + b for a, b in zip(input_tokens, output_tokens, strict=True)])
    avg_latency_s = _avg(times)
    avg_iterations = _avg(iterations)
    finalization_fail_rate = _avg(finalization_fails)
    accuracy = _avg([1.0 if r["judge"]["is_correct"] else 0.0 for r in records])
    recovery_attempt_rate = _avg(recovery_attempted_values)
    recovery_used_rate = _avg(recovery_used_values)
    recovery_success_rate = _avg(recovery_success_values)

    weights = policy.get("score_weights", {})
    failure_tag_weights = dict(DEFAULT_FAILURE_TAG_WEIGHTS)
    failure_tag_weights.update(
        {
            str(k): float(v)
            for k, v in dict(policy.get("failure_tag_weights", {})).items()
            if isinstance(v, (int, float))
        }
    )

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
    protocol_fail_rate = _avg(
        [_protocol_failure_score(tags if isinstance(tags, list) else [], failure_tag_weights) for tags in failure_tag_values]
    )
    protocol_penalty = float(weights.get("protocol_penalty", 0.24)) * protocol_fail_rate
    recovery_used_penalty = float(weights.get("recovery_used_penalty", 0.02)) * recovery_used_rate
    recovery_bonus = float(weights.get("recovery_bonus", 0.10)) * recovery_success_rate

    combined_score = float(weights.get("correctness", 1.0)) * avg_correctness
    combined_score -= (
        tokens_penalty
        + latency_penalty
        + iter_penalty
        + finalization_penalty
        + protocol_penalty
        + recovery_used_penalty
    )
    combined_score += recovery_bonus
    combined_score = max(0.0, min(1.0, combined_score))

    # Guardrail: if accuracy is very low, cap score to discourage cheap wrong answers.
    if accuracy < 0.5:
        combined_score *= 0.35

    hard_gate_triggered = 0.0
    hard_gate_tags: list[str] = []
    if HARD_GATE_PROTOCOL:
        observed_tags = {
            str(tag)
            for tags in failure_tag_values
            if isinstance(tags, list)
            for tag in tags
        }
        hard_gate_tags = sorted(observed_tags.intersection(HARD_GATE_FAILURE_TAGS))
        if hard_gate_tags:
            hard_gate_triggered = 1.0
            combined_score = min(combined_score, HARD_GATE_SCORE_CAP)

    return {
        "combined_score": combined_score,
        "accuracy": accuracy,
        "avg_correctness": avg_correctness,
        "avg_answer_closeness": avg_answer_closeness,
        "avg_tokens": avg_tokens,
        "avg_latency_s": avg_latency_s,
        "avg_iterations": avg_iterations,
        "finalization_fail_rate": finalization_fail_rate,
        "protocol_fail_rate": protocol_fail_rate,
        "recovery_attempt_rate": recovery_attempt_rate,
        "recovery_used_rate": recovery_used_rate,
        "recovery_success_rate": recovery_success_rate,
        "hard_gate_triggered": hard_gate_triggered,
        "hard_gate_tags_csv": ",".join(hard_gate_tags),
    }


def _summarize_failure_diagnostics(
    records: list[dict[str, Any]],
    max_examples: int = 2,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    tag_counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for record in records:
        tags = record.get("run_metrics", {}).get("failure_tags", [])
        if not isinstance(tags, list):
            continue
        for tag in tags:
            key = str(tag)
            tag_counts[key] = tag_counts.get(key, 0) + 1

        if tags and len(examples) < max_examples:
            answer_raw = str(record.get("answer_raw", ""))
            examples.append(
                {
                    "puzzle_id": record.get("puzzle_id"),
                    "tags": tags,
                    "answer_raw_preview": answer_raw[:220],
                    "judge_correctness_raw": float(record.get("judge_raw", {}).get("correctness", 0.0)),
                    "judge_correctness_effective": float(record.get("judge", {}).get("correctness", 0.0)),
                    "recovered_from_var": record.get("run_metrics", {}).get("recovered_from_var", ""),
                }
            )

    return tag_counts, examples


def _build_mutator_feedback(records: list[dict[str, Any]], max_items: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda record: float(record.get("judge", {}).get("correctness", 0.0)),
    )
    selected = ranked[: min(max_items, len(ranked))]

    feedback: list[dict[str, Any]] = []
    for record in selected:
        feedback.append(
            {
                "puzzle_id": record.get("puzzle_id"),
                "question": _truncate_text(str(record.get("question_text", "")), 900),
                "solver_answer": _truncate_text(str(record.get("answer_raw", "")), 900),
                "score": float(record.get("judge", {}).get("correctness", 0.0)),
                "is_correct": bool(record.get("judge", {}).get("is_correct", False)),
                "failure_tags": list(record.get("run_metrics", {}).get("failure_tags", [])),
                "judge_reasoning": _truncate_text(
                    str(record.get("judge", {}).get("reasoning", "")),
                    700,
                ),
            }
        )
    return feedback


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
    policy_raw = _load_policy(program_path)
    policy, policy_constraints = _apply_policy_constraints(policy_raw)
    judge = LLMJudge(JUDGE_MODEL_NAME)
    eval_output_dir = _make_eval_output_dir(program_path, str(policy.get("name", "policy")))
    LOGGER.info(
        "Starting policy evaluation: policy=%s output_dir=%s stages=%s limited_genome_mode=%s",
        str(policy.get("name", "")),
        str(eval_output_dir) if eval_output_dir else "(disabled)",
        STAGE_SPECS,
        LIMITED_GENOME_MODE,
    )
    if policy_constraints.get("limited_genome_mode"):
        LOGGER.info(
            "Policy constraints active: evolvable=%s locked=%s",
            policy_constraints.get("evolvable_fields", []),
            policy_constraints.get("locked_fields", []),
        )

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
    stage_metrics: dict[str, dict[str, Any]] = {}
    stopped_stage: str | None = None

    for stage_name, stage_size in STAGE_SPECS:
        subset = _choose_stage_subset(puzzles, stage_size)
        records: list[dict[str, Any]] = []
        for puzzle_idx, puzzle in enumerate(subset, start=1):
            LOGGER.info(
                "Evaluating stage %s puzzle %d/%d (id=%s)",
                stage_name,
                puzzle_idx,
                len(subset),
                str(puzzle.get("number", "unknown")),
            )
            try:
                record = _run_single_puzzle(puzzle, policy, stage_name, judge)
                records.append(record)
            except Exception as exc:
                record = {
                    "puzzle_id": puzzle.get("number"),
                    "question_text": question_text_only(puzzle).strip()
                    or json.dumps(format_question(puzzle), ensure_ascii=True),
                    "answer": "",
                    "answer_raw": "",
                    "answer_recovered": "",
                    "judge": {
                        "is_correct": False,
                        "correctness": 0.0,
                        "answer_closeness": 0.0,
                        "failure_type": "other",
                        "reasoning": f"runtime_error: {exc}",
                    },
                    "judge_raw": {
                        "is_correct": False,
                        "correctness": 0.0,
                        "answer_closeness": 0.0,
                        "failure_type": "other",
                        "reasoning": f"runtime_error: {exc}",
                    },
                    "judge_recovered": {},
                    "run_metrics": {
                        "iterations": 0,
                        "total_llm_calls": 0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "execution_time_s": 0.0,
                        "finalization_mode": "error",
                        "finalization_fail": 1.0,
                        "loop_repetition": 0.0,
                        "max_consecutive_repeats": 0.0,
                        "final_var_call_name": "",
                        "failure_tags": [
                            "runtime_error",
                            "final_var_missing",
                            "max_iter_no_finalization",
                        ],
                        "failure_tag_count": 3.0,
                        "latest_locals_keys": [],
                        "recovery_attempted": 0.0,
                        "recovery_used": 0.0,
                        "recovery_success": 0.0,
                        "recovered_from_var": "",
                    },
                }
                records.append(record)

            _save_puzzle_record(eval_output_dir, stage_name, puzzle, record)

        all_records.extend(records)
        stage_score = _score_records(records, policy)
        stage_metrics[stage_name] = stage_score

        if stage_name in STAGE_MIN_COMBINED and stage_score["combined_score"] < STAGE_MIN_COMBINED[stage_name]:
            stopped_stage = stage_name
            break

    final_stage = list(stage_metrics.keys())[-1]
    final_score = stage_metrics[final_stage]
    elapsed = time.time() - eval_started
    failure_tag_counts, _failure_examples = _summarize_failure_diagnostics(all_records, max_examples=2)
    mutator_feedback = _build_mutator_feedback(all_records, max_items=5)

    metrics = {
        "combined_score": final_score["combined_score"],
        "accuracy": final_score["accuracy"],
        "avg_correctness": final_score["avg_correctness"],
        "avg_answer_closeness": final_score["avg_answer_closeness"],
        "avg_tokens": final_score["avg_tokens"],
        "avg_latency_s": final_score["avg_latency_s"],
        "avg_iterations": final_score["avg_iterations"],
        "finalization_fail_rate": final_score["finalization_fail_rate"],
        "protocol_fail_rate": final_score.get("protocol_fail_rate", 1.0),
        "recovery_attempt_rate": final_score.get("recovery_attempt_rate", 0.0),
        "recovery_used_rate": final_score.get("recovery_used_rate", 0.0),
        "recovery_success_rate": final_score.get("recovery_success_rate", 0.0),
        "hard_gate_triggered": final_score.get("hard_gate_triggered", 0.0),
        "hard_gate_tags_csv": str(final_score.get("hard_gate_tags_csv", "")),
        # Extra tracking metrics:
        "stage_count_reached": float(len(stage_metrics)),
        "evaluation_time_s": float(elapsed),
    }

    hint_tag, hint_primary = _derive_primary_hint(failure_tag_counts)
    metrics["policy_name"] = str(policy.get("name", ""))
    metrics["root_model"] = ROOT_MODEL_NAME
    metrics["judge_model"] = JUDGE_MODEL_NAME
    metrics["stopped_stage"] = stopped_stage or ""
    metrics["failure_tag_counts_json"] = json.dumps(failure_tag_counts, sort_keys=True)
    metrics["mutator_feedback_json"] = json.dumps(mutator_feedback, ensure_ascii=True)
    metrics["puzzle_output_dir"] = str(eval_output_dir) if eval_output_dir else ""
    metrics["hint_primary"] = hint_primary
    metrics["hint_tag"] = hint_tag
    LOGGER.info(
        "Evaluation complete: policy=%s score=%.4f hint_tag=%s hard_gate=%s",
        metrics["policy_name"],
        float(metrics["combined_score"]),
        hint_tag,
        str(metrics["hard_gate_triggered"]),
    )
    return metrics
