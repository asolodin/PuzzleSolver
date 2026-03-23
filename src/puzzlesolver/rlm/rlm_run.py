"""
Configurable RLM runner for Moscow puzzles.

Supports:
- built-in baseline policy (preserved for baseline comparisons)
- loading an evolved policy from an OpenEvolve `best_program.py` (build_policy)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parents[2]
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
from rlm import RLM  # noqa: E402
from rlm.core.types import RLMChatCompletion  # noqa: E402
from rlm.logger import RLMLogger  # noqa: E402

load_dotenv()

RLM_DIR = PROJECT_ROOT / "rlm"
EXPERIMENT_DIR = RLM_DIR / "experiments" / "moscow_puzzles"
PUZZLES_DIR = EXPERIMENT_DIR / "puzzles"
ANSWERS_DIR = EXPERIMENT_DIR / "answers"
REPORTS_DIR = EXPERIMENT_DIR / "run_reports"
LOG_DIR = RLM_DIR / "logs"

ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

RLM_MODEL_NAME = os.getenv("RLM_MODEL_NAME", "gpt-5.4-mini")
JUDGE_MODEL_NAME = os.getenv("RLM_JUDGE_MODEL", "gpt-5.4")
RLM_EXECUTION_TIMEOUT_SECONDS = float(os.getenv("RLM_EXECUTION_TIMEOUT_SECONDS", "20"))
RLM_MAX_DEPTH_DEFAULT = int(os.getenv("RLM_MAX_DEPTH", "1"))
RLM_MAX_ITERATIONS_DEFAULT = int(os.getenv("RLM_MAX_ITERATIONS", "10"))
SYSTEM_PROMPT_DELTA_MAX_CHARS = int(os.getenv("RLM_SYSTEM_PROMPT_DELTA_MAX_CHARS", "1200"))

BASELINE_POLICY: dict[str, Any] = {
    "name": "baseline_v1",
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
    "max_depth": 1,
    "max_iterations": 10,
}

BUILTIN_POLICIES: dict[str, dict[str, Any]] = {
    "baseline": BASELINE_POLICY,
}

logger = RLMLogger(log_dir=str(LOG_DIR))


@dataclass
class RunHooks:
    subcall_start_events: int = 0
    subcall_complete_events: int = 0
    subcall_error_events: int = 0
    subcall_total_duration_s: float = 0.0
    subcall_models: dict[str, int] = field(default_factory=dict)
    iteration_start_events: int = 0
    iteration_complete_events: int = 0
    iteration_total_duration_s: float = 0.0

    def on_subcall_start(self, depth: int, model: str, prompt_preview: str) -> None:
        del depth, prompt_preview
        self.subcall_start_events += 1
        self.subcall_models[model] = self.subcall_models.get(model, 0) + 1

    def on_subcall_complete(
        self, depth: int, model: str, duration: float, error_or_none: str | None
    ) -> None:
        del depth, model
        self.subcall_complete_events += 1
        self.subcall_total_duration_s += float(duration)
        if error_or_none:
            self.subcall_error_events += 1

    def on_iteration_start(self, depth: int, iteration_num: int) -> None:
        del depth, iteration_num
        self.iteration_start_events += 1

    def on_iteration_complete(self, depth: int, iteration_num: int, duration: float) -> None:
        del depth, iteration_num
        self.iteration_complete_events += 1
        self.iteration_total_duration_s += float(duration)


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "run"


def _build_backend_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_name": RLM_MODEL_NAME,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    return kwargs


def _load_policy_from_python_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    module_name = f"rlm_run_policy_{path.stem}_{int(path.stat().st_mtime_ns)}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load policy module from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "build_policy"):
        raise RuntimeError(f"Policy module must define build_policy(): {path}")

    policy = module.build_policy()
    if not isinstance(policy, dict):
        raise RuntimeError("build_policy() must return dict")
    return policy


def _resolve_policy(policy_name: str, policy_file: str | None) -> tuple[dict[str, Any], str]:
    if policy_file:
        path = Path(policy_file).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return _load_policy_from_python_file(path), str(path)

    key = policy_name.strip().lower()
    if key not in BUILTIN_POLICIES:
        valid = ", ".join(sorted(BUILTIN_POLICIES.keys()))
        raise ValueError(f"Unknown policy_name '{policy_name}'. Valid values: {valid}")
    return dict(BUILTIN_POLICIES[key]), f"builtin:{key}"


def create_rlm(run_hooks: RunHooks, policy: dict[str, Any]) -> RLM:
    system_delta = str(
        policy.get("custom_system_prompt_delta", policy.get("custom_system_prompt", ""))
    ).strip()
    custom_system_prompt = None
    if system_delta:
        custom_system_prompt = build_policy_system_prompt(
            system_delta,
            max_delta_chars=SYSTEM_PROMPT_DELTA_MAX_CHARS,
        )

    max_depth = int(policy.get("max_depth", RLM_MAX_DEPTH_DEFAULT))
    max_iterations = int(policy.get("max_iterations", RLM_MAX_ITERATIONS_DEFAULT))

    return RLM(
        backend="openai",
        backend_kwargs=_build_backend_kwargs(),
        environment="docker",
        environment_kwargs={
            "execution_timeout_seconds": RLM_EXECUTION_TIMEOUT_SECONDS,
            "memory_limit": "4g",
            "cpu_limit": 4,
        },
        max_depth=max_depth,
        max_iterations=max_iterations,
        logger=logger,
        verbose=True,
        custom_system_prompt=custom_system_prompt,
        on_subcall_start=run_hooks.on_subcall_start,
        on_subcall_complete=run_hooks.on_subcall_complete,
        on_iteration_start=run_hooks.on_iteration_start,
        on_iteration_complete=run_hooks.on_iteration_complete,
    )


def collect_run_metrics(result: RLMChatCompletion, hooks: RunHooks) -> dict[str, Any]:
    trajectory = result.metadata or {}
    iterations = trajectory.get("iterations", []) if isinstance(trajectory, dict) else []
    code_blocks = []
    for iteration in iterations:
        code_blocks.extend(iteration.get("code_blocks", []))

    repl_error_blocks = 0
    repl_llm_calls = 0
    for block in code_blocks:
        block_result = block.get("result", {})
        stderr = str(block_result.get("stderr", "")).strip()
        if stderr:
            repl_error_blocks += 1
        repl_llm_calls += len(block_result.get("rlm_calls", []))

    finalization = detect_finalization_mode(trajectory if isinstance(trajectory, dict) else {})

    usage_summary = result.usage_summary
    per_model_usage = {
        model: summary.to_dict()
        for model, summary in usage_summary.model_usage_summaries.items()
    }
    total_llm_calls = sum(summary["total_calls"] for summary in per_model_usage.values())

    return {
        "execution_time_s": float(result.execution_time),
        "iterations": len(iterations),
        "code_blocks": len(code_blocks),
        "repl_error_blocks": repl_error_blocks,
        "repl_llm_calls": repl_llm_calls,
        "final_var_events": finalization["final_var_events"],
        "explicit_final_tag_events": finalization["explicit_final_tag_events"],
        "finalization_mode": finalization["finalization_mode"],
        "total_llm_calls": total_llm_calls,
        "total_input_tokens": usage_summary.total_input_tokens,
        "total_output_tokens": usage_summary.total_output_tokens,
        "total_cost_usd": usage_summary.total_cost,
        "usage_by_model": per_model_usage,
        "hook_metrics": {
            "subcall_start_events": hooks.subcall_start_events,
            "subcall_complete_events": hooks.subcall_complete_events,
            "subcall_error_events": hooks.subcall_error_events,
            "subcall_total_duration_s": hooks.subcall_total_duration_s,
            "subcall_models": hooks.subcall_models,
            # NOTE: these counters currently stay zero because this rlm version defines
            # iteration callbacks but does not invoke them in the loop.
            "iteration_start_events": hooks.iteration_start_events,
            "iteration_complete_events": hooks.iteration_complete_events,
            "iteration_total_duration_s": hooks.iteration_total_duration_s,
        },
    }


def summarize(
    records: list[dict[str, Any]], policy_name: str, policy_source: str, run_id: str
) -> dict[str, Any]:
    judged = [r for r in records if not r["judge"].get("skipped") and "error" not in r["judge"]]
    correct = [r for r in judged if r["judge"].get("is_correct")]

    def avg(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    final_var_runs = [r for r in records if r["run_metrics"].get("finalization_mode") == "FINAL_VAR"]
    incorrect_no_final_var = [
        r
        for r in judged
        if (not r["judge"].get("is_correct"))
        and r["run_metrics"].get("finalization_mode") != "FINAL_VAR"
    ]

    return {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "policy_name": policy_name,
        "policy_source": policy_source,
        "rlm_model": RLM_MODEL_NAME,
        "judge_model": JUDGE_MODEL_NAME,
        "num_puzzles": len(records),
        "num_judged": len(judged),
        "num_correct": len(correct),
        "accuracy": (len(correct) / len(judged)) if judged else None,
        "avg_iterations": avg([float(r["run_metrics"].get("iterations", 0)) for r in records]),
        "avg_execution_time_s": avg(
            [float(r["run_metrics"].get("execution_time_s", 0.0)) for r in records]
        ),
        "avg_total_input_tokens": avg(
            [float(r["run_metrics"].get("total_input_tokens", 0)) for r in records]
        ),
        "avg_total_output_tokens": avg(
            [float(r["run_metrics"].get("total_output_tokens", 0)) for r in records]
        ),
        "avg_total_llm_calls": avg(
            [float(r["run_metrics"].get("total_llm_calls", 0)) for r in records]
        ),
        "final_var_rate": (len(final_var_runs) / len(records)) if records else None,
        "incorrect_without_final_var": len(incorrect_no_final_var),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline/evolved RLM policies on puzzle set.")
    parser.add_argument(
        "--policy-name",
        default="baseline",
        help="Built-in policy name. Default: baseline",
    )
    parser.add_argument(
        "--policy-file",
        default=None,
        help="Path to a Python file containing build_policy() (e.g. OpenEvolve best_program.py).",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional label for output folder/report naming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy, policy_source = _resolve_policy(args.policy_name, args.policy_file)
    policy_name = str(policy.get("name", args.policy_name))

    judge = LLMJudge(
        JUDGE_MODEL_NAME,
        include_usage=True,
        include_raw_response=True,
    )
    puzzle_paths = sorted(PUZZLES_DIR.glob("*.json"))
    records: list[dict[str, Any]] = []

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_stamp}_{_safe_name(args.run_label or policy_name)}"
    run_answers_dir = ANSWERS_DIR / run_id
    run_answers_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = str(policy.get("root_prompt_template", BASELINE_POLICY["root_prompt_template"]))
    root_prompt_suffix = str(policy.get("root_prompt_suffix", "")).strip() or None

    print(f"Running policy='{policy_name}' source='{policy_source}' run_id='{run_id}'")

    for puzzle_path in puzzle_paths:
        with puzzle_path.open("r", encoding="utf-8") as f:
            puzzle = json.load(f)

        formatted_question = format_question(puzzle)
        q_string = json.dumps(formatted_question, indent=2)
        gold_answer = answer_key_text(puzzle)

        hooks = RunHooks()
        rlm = create_rlm(hooks, policy)
        prompt = prompt_template.format(question=q_string)
        try:
            result: RLMChatCompletion = rlm.completion(prompt, root_prompt_suffix)
        finally:
            rlm.close()

        run_metrics = collect_run_metrics(result, hooks)
        judge_result = judge.judge(
            problem_statement=question_text_only(puzzle),
            gold_answer=gold_answer,
            candidate_answer=result.response,
        )

        output_payload = {
            "policy": {
                "name": policy_name,
                "source": policy_source,
                "config": policy,
            },
            "question": puzzle,
            "result": result.to_dict(),
            "answer": result.response,
            "judge": judge_result,
            "run_metrics": run_metrics,
            "gold_answer_text": gold_answer,
        }

        answer_path = run_answers_dir / puzzle_path.name
        with answer_path.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        record = {
            "puzzle_file": puzzle_path.name,
            "judge": judge_result,
            "run_metrics": run_metrics,
            "answer_path": str(answer_path),
        }
        records.append(record)

        print(
            f"{puzzle_path.name}: correct={judge_result.get('is_correct')} "
            f"correctness={judge_result.get('correctness')} "
            f"iterations={run_metrics.get('iterations')} "
            f"calls={run_metrics.get('total_llm_calls')} "
            f"finalization={run_metrics.get('finalization_mode')}"
        )

    run_summary = summarize(records, policy_name=policy_name, policy_source=policy_source, run_id=run_id)
    report_path = REPORTS_DIR / f"{run_id}_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": run_summary,
                "records": records,
            },
            f,
            indent=2,
        )

    print("\nRun summary:")
    print(json.dumps(run_summary, indent=2))
    print(f"Saved answers to: {run_answers_dir}")
    print(f"Saved summary to: {report_path}")


if __name__ == "__main__":
    main()
