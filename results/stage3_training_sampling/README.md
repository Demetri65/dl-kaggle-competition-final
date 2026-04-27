# Stage 3 Training Sampling Results

This folder tracks Stage 3 runs that kept the selected `f04_candidate_yes_no`
formulation and changed training/sampling behavior.

Current selected final config:

- `c03_balanced_answer`
- parent: `f04_candidate_yes_no`
- formulation: `candidate_yes_no`
- sampling: `balanced_answer_index`
- LoRA: rank `16`, alpha `32`

The full summary is in `2026-04-27_stage3_training_sampling_summary.md`.

Final public leaderboard result:

- `c03_balanced_answer` train+val submission: approximately `0.71`
  user-reported public accuracy

