"""
Shared utilities for RLM puzzle experiments and OpenEvolve evaluators.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from rlm.clients import get_client
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

FINAL_TAG_PATTERN = re.compile(r"^\s*FINAL(?:_VAR)?\(.*\)\s*$", re.MULTILINE | re.DOTALL)

RLM_POLICY_FIXED_SYSTEM_SCAFFOLD = """
You are running inside an automated benchmark harness, not a human chat.
Treat the prompt as programmatic task input.
Always inspect the full `context` first, then solve using REPL/tool calls.
Prefer FINAL_VAR when you computed a concrete result in REPL.
Do not replace a computed symbolic/programmatic result with a regenerated natural-language guess.
""".strip()

JUDGE_SYSTEM_PROMPT = """
You are a strict evaluator for puzzle answers.
Compare the candidate answer against the gold answer and return JSON only.
Scoring rubric:
- correctness: 1.0 only when the candidate is fully correct; otherwise < 1.0.
- answer_closeness: semantic closeness to the gold answer.
Required JSON fields:
{
  "is_correct": boolean,
  "correctness": number,
  "answer_closeness": number,
  "failure_type": "none|wrong_answer|partial|format_issue|other",
  "reasoning": string
}
Do not include markdown.
""".strip()

JUDGE_USER_TEMPLATE = """
Problem:
{problem_statement}

Gold answer:
{gold_answer}

Candidate answer:
{candidate_answer}

Return JSON only.
""".strip()


def extract_json(text: str) -> dict[str, Any]:
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


def clamp_01(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def format_part(puzzle: dict[str, Any], part_name: str) -> list[dict[str, Any]]:
    """
    Format puzzle part into the same structure expected by RLM prompt context.
    """
    items: list[dict[str, Any]] = []
    text_part = {"text": ""}
    for part in puzzle.get(part_name, []):
        if part.get("type") == "text":
            value = str(part.get("value", ""))
            text_part["text"] = text_part["text"] + ("\n" if text_part["text"] else "") + value
        elif part.get("type") == "photo":
            if text_part["text"]:
                items.append(text_part)
                text_part = {"text": ""}
            photo_part = {
                "photo": {k: v for k, v in part.items() if k not in ["type", "image_path", "bbox"]}
            }
            items.append(photo_part)
    if text_part["text"]:
        items.append(text_part)
    return items


def format_question(puzzle: dict[str, Any]) -> list[dict[str, Any]]:
    return format_part(puzzle, "question")


def question_text_only(puzzle: dict[str, Any]) -> str:
    return "\n".join(
        str(part.get("value", "")).strip()
        for part in puzzle.get("question", [])
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def answer_key_text(puzzle: dict[str, Any]) -> str:
    return "\n".join(
        str(part.get("value", "")).strip()
        for part in puzzle.get("answer", [])
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def detect_finalization_mode(trajectory: dict[str, Any] | None) -> dict[str, Any]:
    """
    Detect finalization path from RLM trajectory metadata.
    """
    iterations = trajectory.get("iterations", []) if isinstance(trajectory, dict) else []
    final_var_events = 0
    explicit_final_tag_events = 0

    for iteration in iterations:
        for block in iteration.get("code_blocks", []):
            block_result = block.get("result", {})
            if block_result.get("final_answer") is not None:
                final_var_events += 1
        response = str(iteration.get("response", ""))
        if FINAL_TAG_PATTERN.search(response):
            explicit_final_tag_events += 1

    if final_var_events > 0:
        mode = "FINAL_VAR"
    elif explicit_final_tag_events > 0:
        mode = "FINAL"
    else:
        mode = "fallback_or_plain_text"

    return {
        "finalization_mode": mode,
        "final_var_events": final_var_events,
        "explicit_final_tag_events": explicit_final_tag_events,
    }


class LLMJudge:
    """
    Shared LLM judge with optional usage/raw-response capture.
    """

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
        include_usage: bool = False,
        include_raw_response: bool = False,
    ):
        self.model_name = model_name
        self.include_usage = include_usage
        self.include_raw_response = include_raw_response
        self.client = get_client(
            "openai",
            {
                "model_name": model_name,
                "api_key": os.getenv(api_key_env),
            },
        )

    def judge(self, problem_statement: str, gold_answer: str, candidate_answer: str) -> dict[str, Any]:
        if not gold_answer.strip():
            return {
                "judge_model": self.model_name,
                "skipped": True,
                "is_correct": False,
                "correctness": 0.0,
                "answer_closeness": 0.0,
                "failure_type": "other",
                "reasoning": "No gold answer text found for this puzzle.",
            }

        user_prompt = JUDGE_USER_TEMPLATE.format(
            problem_statement=problem_statement.strip(),
            gold_answer=gold_answer.strip(),
            candidate_answer=candidate_answer.strip(),
        )

        try:
            response_text = self.client.completion(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model_name,
            )
            parsed = extract_json(response_text)
            correctness = clamp_01(parsed.get("correctness"))
            answer_closeness = clamp_01(parsed.get("answer_closeness"))
            is_correct_raw = parsed.get("is_correct")
            is_correct = bool(is_correct_raw) if isinstance(is_correct_raw, bool) else correctness >= 0.95

            result = {
                "judge_model": self.model_name,
                "is_correct": bool(is_correct),
                "correctness": correctness,
                "answer_closeness": answer_closeness,
                "failure_type": str(parsed.get("failure_type", "other")),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
            }
            if self.include_usage:
                try:
                    result["usage"] = self.client.get_last_usage().to_dict()
                except Exception:
                    pass
            if self.include_raw_response:
                result["raw_response"] = response_text
            return result
        except Exception as exc:
            return {
                "judge_model": self.model_name,
                "error": str(exc),
                "is_correct": False,
                "correctness": 0.0,
                "answer_closeness": 0.0,
                "failure_type": "other",
                "reasoning": "Judge call failed.",
            }


def _escape_prompt_fragment(fragment: str) -> str:
    """
    Escape braces in appended fragments so `str.format(custom_tools_section=...)`
    used by RLM does not treat them as format placeholders.
    """
    return fragment.replace("{", "{{").replace("}", "}}")


def build_policy_system_prompt(
    custom_system_prompt_delta: str | None,
    max_delta_chars: int = 1200,
) -> str:
    """
    Build a system prompt as:
      fixed RLM scaffold + fixed policy scaffold + bounded evolvable delta.

    The base RLM scaffold stays fixed; evolution can only change `custom_system_prompt_delta`.
    """
    delta = str(custom_system_prompt_delta or "").strip()
    if max_delta_chars > 0 and len(delta) > max_delta_chars:
        delta = delta[:max_delta_chars].rstrip()

    pieces = [RLM_SYSTEM_PROMPT.rstrip(), _escape_prompt_fragment(RLM_POLICY_FIXED_SYSTEM_SCAFFOLD)]
    if delta:
        pieces.append("Additional policy hints:\n" + _escape_prompt_fragment(delta))
    return "\n\n".join(pieces)
