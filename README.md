# PuzzleSolver

Research sandbox for integrating **Recursive Language Models (RLM)** with **OpenEvolve** on logical/mathematical puzzles.

## What is in this repo

- Baseline RLM puzzle runner with LLM judging and run metrics
- Shared utilities for puzzle formatting, judging, and finalization checks
- OpenEvolve outer-loop setup to evolve RLM policy parameters/prompts across puzzle batches

## Setup

```bash
uv sync
```

Set your API key:

```bash
export OPENAI_API_KEY="..."
```

## Run baseline RLM

Primary entrypoint:

```bash
python src/puzzlesolver/rlm/rlm_run.py
```

Compatibility entrypoint (same behavior):

```bash
python rlm/experiments/quickstart.py
```

Outputs are written under:
- `rlm/experiments/moscow_puzzles/answers`
- `rlm/experiments/moscow_puzzles/run_reports`

## Run OpenEvolve policy evolution

Recommended Python runner:

```bash
python src/puzzlesolver/openevolve/openevolve_run.py --iterations 10
```

Optional fixed model controls:

```bash
export RLM_POLICY_ROOT_MODEL="gpt-5-mini"
export RLM_POLICY_JUDGE_MODEL="gpt-5.4"
```

Alternative launcher:

```bash
cd openevolve/rlm_policy
./run_evolution.sh --iterations 10
```

## Key paths

- Baseline runner: `src/puzzlesolver/rlm/rlm_run.py`
- Shared RLM utilities: `src/puzzlesolver/rlm_shared.py`
- OpenEvolve runner: `src/puzzlesolver/openevolve/openevolve_run.py`
- OpenEvolve evaluator: `src/puzzlesolver/openevolve/rlm_policy_evaluator.py`
- Policy genome seed: `openevolve/rlm_policy/initial_program.py`
