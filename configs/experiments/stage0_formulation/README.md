# Stage 0 Formulations

Stage 0 selected `f04_candidate_yes_no` as the primary formulation after full
A100 LoRA training. `f03_multimodal_letter` remains active as a cleaner
restricted-output comparator.

## Active Configs

| experiment_id | role | formulation | included fields | status |
|---|---|---|---|---|
| `f04_candidate_yes_no` | selected parent | candidate yes/no | image, question, candidate answer | promote to Stage 1 |
| `f03_multimodal_letter` | comparator | restricted letter scoring | image, question, choices, hint, lecture | keep for reference |

## Selected Parent

`f04_candidate_yes_no` asks the model to verify one candidate answer at a time
with `Yes` or `No`. It outperformed the restricted-letter comparator on full
validation:

| experiment_id | run type | val_accuracy |
|---|---|---:|
| `f04_candidate_yes_no` | full A100 LoRA train/eval | 0.693702 |
| `f03_multimodal_letter` | full A100 LoRA train/eval | 0.646947 |
| `f03_multimodal_letter` | artifact-backed eval sanity rerun | 0.645992 |

## Notes

- Use `f04_candidate_yes_no` as the Stage 1 parent.
- Keep `f03_multimodal_letter` only as the restricted-output comparator.
- `f04` is weak on 5-choice examples and underuses the final option, so later
  stages should watch `accuracy_by_num_choices` and prediction distribution.
- Archived Stage 0 configs live under `configs/archive/stage0_formulation/`.
