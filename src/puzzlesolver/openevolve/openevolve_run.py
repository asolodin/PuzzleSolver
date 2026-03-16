#!/usr/bin/env python
"""
Project-local OpenEvolve runner with rlm_policy defaults.

Usage examples:
  python src/puzzlesolver/openevolve/openevolve_run.py
  python src/puzzlesolver/openevolve/openevolve_run.py --iterations 20
  python src/puzzlesolver/openevolve/openevolve_run.py --checkpoint openevolve_output/checkpoints/checkpoint_10

Behavior:
  - If you pass positional initial/evaluator paths explicitly, it behaves like normal openevolve-run.
  - Otherwise, it injects defaults for this project:
      openevolve/rlm_policy/initial_program.py,
      src/puzzlesolver/openevolve/rlm_policy_evaluator.py,
      openevolve/rlm_policy/config.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

from openevolve.cli import main as openevolve_main


def _default_paths() -> list[str]:
    here = Path(__file__).resolve()
    project_root = here.parents[3]
    return [
        str(project_root / "openevolve" / "rlm_policy" / "initial_program.py"),
        str(project_root / "src" / "puzzlesolver" / "openevolve" / "rlm_policy_evaluator.py"),
        "--config",
        str(project_root / "openevolve" / "rlm_policy" / "config.yaml"),
    ]


def _has_explicit_program_paths(args: list[str]) -> bool:
    if len(args) < 2:
        return False
    return not args[0].startswith("-") and not args[1].startswith("-")


def main() -> int:
    args = sys.argv[1:]
    if _has_explicit_program_paths(args):
        return openevolve_main()

    sys.argv = [sys.argv[0], *_default_paths(), *args]
    return openevolve_main()


if __name__ == "__main__":
    raise SystemExit(main())

