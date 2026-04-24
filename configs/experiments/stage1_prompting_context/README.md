# Stage 1 Prompting Context

Stage 1 screens lightweight inference-time prompt changes on top of the trained
`f04_candidate_yes_no` artifact before spending more A100 training budget.

## Active Eval-Only Screens

| experiment_id | parent | change | purpose |
|---|---|---|---|
| `x11_f04_candidate_choices` | `f04_candidate_yes_no` | add full choices list | test whether comparison context helps multi-option questions |
| `x12_f04_candidate_hint` | `f04_candidate_yes_no` | add hint | test whether compact reasoning context helps without large prompt drift |

## Run Policy

- These are artifact-backed eval-only screens first, not retraining runs.
- Use the trained `full_f04/f04_candidate_yes_no_seed42` adapter.
- Keep `training.epochs=0`.
- On T4, use `training.bf16=false` and `training.fp16=true`.
- If either ablation improves validation despite train/eval prompt mismatch, it
  is a strong candidate for a later retrain.

Archived Stage 1 configs live under `configs/archive/stage1_prompting_context/`.
