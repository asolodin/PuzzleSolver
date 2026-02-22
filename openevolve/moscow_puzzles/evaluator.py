"""
Evaluator for Moscow puzzles with LLM judging and baseline comparison.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

from openevolve.config import load_config
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.templates import TemplateManager


EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(EVALUATOR_DIR, "config.yaml")
SYSTEM_PROMPT_PATH = os.path.join(EVALUATOR_DIR, "evaluation_system.txt")
USER_PROMPT_PATH = os.path.join(EVALUATOR_DIR, "evaluation_user.txt")
DEFAULT_PROBLEM_PATH = os.path.join(EVALUATOR_DIR, "problem_250.json")

NUM_TIMING_RUNS = 3
WARMUP_RUNS = 1

_BASELINE_CACHE: Optional[Dict[str, Any]] = None

logger = logging.getLogger(__name__)

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_problem() -> Dict[str, Any]:
    problem_path = os.environ.get("PROBLEM_FILE") or DEFAULT_PROBLEM_PATH
    if os.path.exists(problem_path):
        with open(problem_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    if loop:
        return loop.run_until_complete(coro)
    return asyncio.run(coro)


def _run_program(program_path: str, timeout_seconds: int) -> Dict[str, Any]:
    times_ms = []
    outputs = []
    errors = []

    runs = max(1, NUM_TIMING_RUNS)
    for run_idx in range(runs + max(0, WARMUP_RUNS)):
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                ["python", program_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if run_idx >= WARMUP_RUNS:
                times_ms.append(elapsed_ms)
                outputs.append(proc.stdout.strip())

            if proc.returncode != 0:
                stderr = proc.stderr.strip()
                errors.append(stderr or f"Non-zero exit code: {proc.returncode}")
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if run_idx >= WARMUP_RUNS:
                times_ms.append(elapsed_ms)
                outputs.append("")
            errors.append(f"Timeout after {timeout_seconds}s")

    avg_time_ms = sum(times_ms) / len(times_ms) if times_ms else float(timeout_seconds) * 1000.0
    output = ""
    for out in reversed(outputs):
        if out:
            output = out
            break

    return {
        "avg_time_ms": float(avg_time_ms),
        "output": output,
        "errors": errors,
    }


def _performance_from_time_ms(time_ms: Optional[float]) -> float:
    if time_ms is None or time_ms <= 0:
        return 0.0
    time_sec = time_ms / 1000.0
    return float(1.0 / (1.0 + time_sec))


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _evaluate_single(
    program_path: str,
    llm: LLMEnsemble,
    system_template: str,
    user_template: str,
    problem_statement: str,
    answer_key: str,
    timeout_seconds: int,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    program_text = _read_text(program_path)
    run_info = _run_program(program_path, timeout_seconds=timeout_seconds)

    tm = TemplateManager()
    tm.add_template("evaluation_system", system_template)
    tm.add_template("evaluation_user", user_template)

    user_prompt = tm.get_template("evaluation_user").format(
        problem_statement=problem_statement,
        answer_key=answer_key,
        current_program=program_text,
    )
    user_prompt += (
        "\n\nExecution time (ms): "
        f"{run_info['avg_time_ms']:.3f}"
        "\n\nProgram output:\n```\n"
        f"{run_info['output']}\n```"
    )

    async def _call_llm():
        return await llm.generate_with_context(
            system_message=tm.get_template("evaluation_system"),
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
        )

    response_text = _run_async(_call_llm())
    eval_json = _extract_json(response_text)

    scores = {
        "correctness": _clamp_score(eval_json.get("correctness", 0.0)),
        "efficiency": _clamp_score(eval_json.get("efficiency", 0.0)),
        "answer_closeness": _clamp_score(eval_json.get("answer_closeness", 0.0)),
        "robustness": _clamp_score(eval_json.get("robustness", 0.0)),
        "creativity": _clamp_score(eval_json.get("creativity", 0.0)),
        "reasoning": eval_json.get("reasoning", "").strip(),
    }

    performance = _performance_from_time_ms(run_info["avg_time_ms"])
    scores["performance"] = performance
    scores["execution_time_ms"] = run_info["avg_time_ms"]
    scores["output"] = run_info["output"]
    scores["errors"] = run_info["errors"]

    return scores


def evaluate(program_path: str) -> Dict[str, Any]:
    global _BASELINE_CACHE

    config = load_config(CONFIG_PATH)
    evaluator_models = config.llm.evaluator_models or config.llm.models
    llm = LLMEnsemble(evaluator_models)

    system_template = _read_text(SYSTEM_PROMPT_PATH)
    user_template = _read_text(USER_PROMPT_PATH)

    problem = _load_problem()
    problem_statement = problem.get("text", "")
    answer_key = problem.get("answer_text", "")

    timeout_seconds = int(getattr(config.evaluator, "timeout", 300))
    max_tokens = getattr(config.llm, "max_tokens", None)

    current_scores = _evaluate_single(
        program_path,
        llm=llm,
        system_template=system_template,
        user_template=user_template,
        problem_statement=problem_statement,
        answer_key=answer_key,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
    )

    if _BASELINE_CACHE is None:
        _BASELINE_CACHE = current_scores

    baseline_scores = _BASELINE_CACHE

    logger.info(f"Current Scores: {current_scores}")

    return {
        "performance": current_scores["performance"],
        "correctness": current_scores["correctness"],
        "efficiency": current_scores["efficiency"],
        "answer_closeness": current_scores["answer_closeness"],
        "robustness": current_scores["robustness"],
        "creativity": current_scores["creativity"],
        "baseline": baseline_scores,
        "current": current_scores,
    }
