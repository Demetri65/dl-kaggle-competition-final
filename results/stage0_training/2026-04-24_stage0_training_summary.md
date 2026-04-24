# Stage 0 A100 Training Summary - 2026-04-24

## Run Info

- Scope: full Stage 0 LoRA training and validation
- Hardware: A100 for training, A100 artifact-backed rerun for `f03`
- Seed: `42`
- Selected formulation: `f04_candidate_yes_no`
- Comparator: `f03_multimodal_letter`

## Results

| experiment_id | run type | val_accuracy | two_choice_accuracy | note |
|---|---|---:|---:|---|
| `f04_candidate_yes_no` | full A100 LoRA train/eval | 0.693702 | 0.754098 | selected Stage 1 parent |
| `f03_multimodal_letter` | full A100 LoRA train/eval | 0.646947 | 0.688525 | restricted-output comparator |
| `f03_multimodal_letter` | artifact-backed eval rerun | 0.645992 | 0.676230 | sanity check of saved artifact and inference batching |

## Key Slice Notes

`f04_candidate_yes_no` was strongest overall and on 2-choice examples. Its main
known weakness is 5-choice accuracy:

| slice | f04 accuracy |
|---|---:|
| 2 choices | 0.754098 |
| 3 choices | 0.653543 |
| 4 choices | 0.813492 |
| 5 choices | 0.136364 |

Prediction distribution for `f04` underuses the final option:

| label | prediction share |
|---|---:|
| `0` | 0.337786 |
| `1` | 0.350191 |
| `2` | 0.230916 |
| `3` | 0.072519 |
| `4` | 0.008588 |

## Takeaway

Promote `f04_candidate_yes_no` into Stage 1. Keep `f03_multimodal_letter` as a
stable restricted-output comparator, but do not spend more Stage 1 budget on old
Stage 0 alternatives unless a later failure requires revisiting formulation.

## Artifact Paths

- `f04` run: `/content/drive/MyDrive/p2p_runs/notebook_c_a100/full_f04/f04_candidate_yes_no_seed42`
- `f03` run: `/content/drive/MyDrive/p2p_runs/notebook_c_a100/full_f03/f03_multimodal_letter_seed42`
- `f03` eval rerun: `/content/drive/MyDrive/p2p_runs/notebook_c_a100/re_eval_f03_b16_full/f03_multimodal_letter_seed42`

## Raw Files

- `2026-04-24_stage0_training_summary.json`
