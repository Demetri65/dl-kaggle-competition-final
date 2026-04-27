# Stage 3 Training Sampling

Stage 3 keeps the selected `f04_candidate_yes_no` formulation and changes
training/sampling settings.

## Active Configs

| experiment_id | parent | change | status |
|---|---|---|---|
| `c03_balanced_answer` | `f04_candidate_yes_no` | rank-16 LoRA, balanced answer-index sampling | selected final config |
| `c04_task_choice_stratified` | `f04_candidate_yes_no` | rank-16 LoRA, stratified by num choices + task | not selected |
| `t02_rank16_attn` | none | rank-16 attention LoRA overlay | pending confirmed metrics |
| `t04_rank16_attn_connector` | none | rank-16 attention + connector LoRA overlay | pending confirmed metrics |
| `t05_lr_1e4` | none | learning rate `1e-4` overlay | pending confirmed metrics |

## Full Validation Results

| experiment_id | formulation | sampling | val_accuracy | two_choice | five_choice | decision |
|---|---|---|---:|---:|---:|---|
| `c03_balanced_answer` | candidate yes/no | balanced answer index | 0.694656 | 0.737705 | 0.454545 | selected |
| `c04_task_choice_stratified` | candidate yes/no | stratified num choices + task | 0.681298 | 0.811475 | 0.318182 | not selected |
| `f04_candidate_yes_no` | candidate yes/no | uniform | 0.693702 | 0.754098 | 0.136364 | Stage 0 baseline |

## Decision

Use `c03_balanced_answer` for final single-model submission. It narrowly beats
the original `f04` baseline overall and substantially improves the weakest
5-choice slice. `c04` is better on 2-choice examples, but loses too much on
overall and 5-choice accuracy.

## Public Leaderboard Note

The final train+val `c03_balanced_answer` submission scored approximately
`0.71` on the public leaderboard. This matches the validation range and should
be treated as a real modeling ceiling for the current low-context candidate
yes/no setup, not as a broken submission.

If another ablation is run, prioritize adding allowed context fields to the
candidate yes/no formulation over seed sweeps.

Full run notes are tracked in
`results/stage3_training_sampling/2026-04-27_stage3_training_sampling_summary.md`.
