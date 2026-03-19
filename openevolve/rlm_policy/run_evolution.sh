#!/bin/bash
set -euo pipefail

# Optional overrides:
#   export OPENAI_API_KEY=...
#   export RLM_POLICY_STAGE_A_SIZE=3
#   export RLM_POLICY_STAGE_B_SIZE=7
#   export RLM_POLICY_STAGE_C_SIZE=12

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v openevolve-run >/dev/null 2>&1; then
  RUNNER="openevolve-run"
elif [ -x "$SCRIPT_DIR/../../.venv/bin/openevolve-run" ]; then
  RUNNER="$SCRIPT_DIR/../../.venv/bin/openevolve-run"
else
  echo "Could not find openevolve-run in PATH or .venv/bin." >&2
  exit 1
fi

HAS_OUTPUT=0
for ARG in "$@"; do
  case "$ARG" in
    --output|--output=*|-o)
      HAS_OUTPUT=1
      break
      ;;
  esac
done

if [ "$HAS_OUTPUT" -eq 1 ]; then
  "$RUNNER" initial_program.py ../../src/puzzlesolver/openevolve/rlm_policy_evaluator.py --config config.yaml "$@"
else
  OUTPUT_DIR="openevolve_output_$(date +%Y%m%d_%H%M%S)"
  echo "Using output directory: $OUTPUT_DIR"
  "$RUNNER" initial_program.py ../../src/puzzlesolver/openevolve/rlm_policy_evaluator.py --config config.yaml --output "$OUTPUT_DIR" "$@"
fi
