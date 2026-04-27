# Stage 3 Training Sampling Summary - 2026-04-27

## Run Info

- Scope: Stage 3 sampling and LoRA-capacity ablations on the selected
  `f04_candidate_yes_no` formulation
- Base model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Seed: `42`
- Final selected config: `c03_balanced_answer`
- Selected final training policy: train on `train.csv + val.csv`, predict
  `test.csv`, do not report validation accuracy for the final submission run

## Corrected Full Runs

These are the relevant Stage 3 runs after `c03` and `c04` were confirmed to
inherit `f04_candidate_yes_no` and use rank-16 LoRA.

| experiment_id | parent | formulation | sampling | trainable_params | val_accuracy | two_choice | three_choice | four_choice | five_choice | note |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `c03_balanced_answer` | `f04_candidate_yes_no` | candidate yes/no | balanced answer index | 4,456,448 | 0.694656 | 0.737705 | 0.627953 | 0.829365 | 0.454545 | selected final single-model config |
| `c04_task_choice_stratified` | `f04_candidate_yes_no` | candidate yes/no | stratified num choices + task | 4,456,448 | 0.681298 | 0.811475 | 0.598425 | 0.785714 | 0.318182 | stronger 2-choice, weaker overall and 5-choice |
| `f04_candidate_yes_no` | none | candidate yes/no | uniform | 4,456,448 | 0.693702 | 0.754098 | 0.653543 | 0.813492 | 0.136364 | Stage 0 baseline; strong overall, poor 5-choice |

## Subject And Task Slices

| experiment_id | closed_choice | true_false | language_science | natural_science | social_science |
|---|---:|---:|---:|---:|---:|
| `c03_balanced_answer` | 0.697537 | 0.606061 | 0.321429 | 0.656371 | 0.860082 |
| `c04_task_choice_stratified` | 0.682759 | 0.636364 | 0.285714 | 0.651223 | 0.823045 |
| `f04_candidate_yes_no` | 0.695567 | 0.636364 | 0.321429 | 0.657658 | 0.851852 |

## Prediction Distribution

| experiment_id | label_0 | label_1 | label_2 | label_3 | label_4 |
|---|---:|---:|---:|---:|---:|
| `c03_balanced_answer` | 0.345420 | 0.341603 | 0.230916 | 0.079198 | 0.002863 |
| `c04_task_choice_stratified` | 0.332061 | 0.357824 | 0.229962 | 0.072519 | 0.007634 |
| `f04_candidate_yes_no` | 0.337786 | 0.350191 | 0.230916 | 0.072519 | 0.008588 |

## Superseded Rank-8 / Restricted-Index Runs

Earlier Stage 3 runs used `restricted_index_scoring` with rank-8 LoRA
(`2,228,224` trainable parameters). Those runs were useful sanity checks, but
they are not the final c03/c04 comparison because they did not use the selected
`f04_candidate_yes_no` formulation.

| experiment_id | formulation | sampling | trainable_params | val_accuracy | two_choice | three_choice | four_choice | five_choice | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `c03_balanced_answer` | restricted index scoring | balanced answer index | 2,228,224 | 0.594466 | 0.663934 | 0.501969 | 0.785714 | 0.181818 | superseded |
| `c04_task_choice_stratified` | restricted index scoring | stratified num choices + task | 2,228,224 | 0.565840 | 0.655738 | 0.507874 | 0.662698 | 0.181818 | superseded |

## Smoke Runs

The smoke runs were short validation checks and should not drive final model
selection.

| experiment_id | sampling | val_accuracy | two_choice | three_choice | four_choice | five_choice |
|---|---|---:|---:|---:|---:|---:|
| `c03_balanced_answer` | balanced answer index | 0.320313 | 0.562500 | 0.361111 | 0.176471 | 0.162791 |
| `c04_task_choice_stratified` | stratified num choices + task | 0.289063 | 0.468750 | 0.416667 | 0.176471 | 0.093023 |

## Decision

Use `c03_balanced_answer` for the final single-model submission.

Rationale:

- `c03` has the best corrected full-run validation accuracy (`0.694656`).
- `c03` is close to the original `f04` overall score while fixing the major
  5-choice weakness (`0.454545` vs `0.136364`).
- `c04` is better on 2-choice examples, but its lower overall, lower 4-choice,
  and lower 5-choice accuracy make it a worse single-model final choice.
- Final submission should train `c03` on `train + val` and predict `test`
  only, with no validation metric reported for that final run.

## Public Leaderboard Outcome

The final `c03_balanced_answer` train+val submission produced a user-reported
public leaderboard score of approximately `0.71`.

This is directionally consistent with the held-out validation results
(`0.694656`) and does not indicate a submission-format failure. The score gap
to reported stronger public leaderboard submissions around `0.88` suggests the
next useful work is not seed tuning, but a higher-information context ablation.

Recommended next ablation if more training budget is available:

- keep `candidate_yes_no`
- keep rank-16 attention LoRA under the 5M trainable-parameter cap
- keep `balanced_answer_index`
- add more allowed competition context fields such as `choices`, `hint`,
  `lecture`, `subject`, `topic`, `category`, and `skill`
- continue to exclude `solution` from inference paths

## Artifact Paths

- Corrected `c03` validation run:
  `/content/drive/MyDrive/p2p_runs/stage3_balanced_answer_index_a100/full/c03_balanced_answer_seed42`
- Corrected `c04` validation run:
  `/content/drive/MyDrive/p2p_runs/stage3_stratified_num_choices_task_a100/full/c04_task_choice_stratified_seed42`
- Final train+val run root:
  `/content/drive/MyDrive/p2p_runs/final_c03_trainval_submission_a100/final_train_val_test`
- Test-only recovery output root:
  `/content/drive/MyDrive/p2p_runs/final_c03_trainval_submission_a100/recover_test_eval`
- User-reported public leaderboard score:
  approximately `0.71`

## Pending Results

No confirmed metrics have been recorded here yet for:

- `t02_rank16_attn`
- `t04_rank16_attn_connector`
- `t05_lr_1e4`
- Stage 4 seed configs `l01_seed1`, `l01_seed2`, `l01_seed3`
