# RLM Policy Evolution Example

This example evolves an **RLM agent policy** (prompt templates + stop/budget rules + score weights) using OpenEvolve.

It does **outer-loop** evolution:
- OpenEvolve mutates `build_policy()` in `initial_program.py`.
- `src/puzzlesolver/openevolve/rlm_policy_evaluator.py` runs the candidate policy on staged puzzle subsets (A/B/C).
- Each puzzle run calls RLM to produce an answer.
- A judge LLM scores semantic correctness.
- Evaluator computes `combined_score` from correctness minus efficiency/finalization penalties.

## Files

- `initial_program.py`: candidate policy genome (`build_policy()` in EVOLVE-BLOCK)
- `src/puzzlesolver/openevolve/rlm_policy_evaluator.py`: staged RLM+judge evaluation and scoring
- `config.yaml`: OpenEvolve config with MAP-Elites dimensions
- `run_evolution.sh`: convenience launcher
- `src/puzzlesolver/rlm_shared.py`: shared formatting/judging/finalization logic (also used by baseline quickstart)

## Run

```bash
cd openevolve/rlm_policy
export OPENAI_API_KEY="..."
./run_evolution.sh --iterations 10
```

Python runner (no shell script required):

```bash
python src/puzzlesolver/openevolve/openevolve_run.py --iterations 10
```

Optional stage-size overrides:

```bash
export RLM_POLICY_STAGE_A_SIZE=3
export RLM_POLICY_STAGE_B_SIZE=7
export RLM_POLICY_STAGE_C_SIZE=12
```

Model control (fixed, not evolved):

```bash
export RLM_POLICY_ROOT_MODEL="gpt-5.2"
export RLM_POLICY_JUDGE_MODEL="gpt-5-mini"
```

If unset, it falls back to `RLM_MODEL_NAME` / `RLM_JUDGE_MODEL`, then defaults.

## Notes

- This is expensive because evaluator executes RLM runs; use small stage sizes first.
- Early stage cutoffs prune weak candidates quickly.
- `combined_score` is the fitness objective; feature dimensions are logged for diversity.
