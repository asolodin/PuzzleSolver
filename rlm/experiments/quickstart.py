"""
Compatibility wrapper for the refactored baseline runner.

Primary entrypoint now:
  python src/puzzlesolver/rlm/rlm_run.py
or
  python -m puzzlesolver.rlm.rlm_run
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puzzlesolver.rlm.rlm_run import main  # noqa: E402


if __name__ == "__main__":
    main()

