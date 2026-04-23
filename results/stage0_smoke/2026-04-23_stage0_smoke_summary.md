# Stage 0 Smoke Run - 2026-04-23

## Run Info
- Scope: Stage 0 zero-shot smoke comparison
- Notebook A: `f02_multimodal_index` vs `f03_multimodal_letter`
- Notebook B: `f04_candidate_yes_no` vs `f05_candidate_yes_no_full_context`
- Hardware: T4
- Seed: `42`
- Validation subset: `32` examples

## Commands
```bash
python3 scripts/run_stage.py f02_multimodal_index f03_multimodal_letter \
  --output-dir /content/drive/MyDrive/p2p_runs/notebook_a_t4/smoke \
  --set training.epochs=0 \
  --set lora.enabled=false \
  --set training.bf16=false \
  --set training.fp16=true \
  --set training.eval_batch_size=8 \
  --set runtime.max_val_examples=32

python3 scripts/run_stage.py f04_candidate_yes_no f05_candidate_yes_no_full_context \
  --output-dir /content/drive/MyDrive/p2p_runs/notebook_b_t4/smoke \
  --set training.epochs=0 \
  --set lora.enabled=false \
  --set training.bf16=false \
  --set training.fp16=true \
  --set training.eval_batch_size=8 \
  --set runtime.max_val_examples=32
```

## Results
| experiment_id | formulation family | val_accuracy | two_choice_accuracy | note |
|---|---|---:|---:|---|
| `f04_candidate_yes_no` | candidate yes/no | 0.31250 | 0.60 | best smoke result of the four |
| `f03_multimodal_letter` | restricted scoring | 0.28125 | 0.80 | best restricted-output smoke result |
| `f02_multimodal_index` | restricted scoring | 0.21875 | 0.20 | weaker than `f03` on this subset |
| `f05_candidate_yes_no_full_context` | candidate yes/no | 0.15625 | 0.20 | weakest smoke result of the four |

## Takeaway
On this `32`-example smoke subset, `f03_multimodal_letter` beat `f02_multimodal_index`, and `f04_candidate_yes_no` beat `f05_candidate_yes_no_full_context`. The strongest smoke performer overall was `f04_candidate_yes_no`.

These are screening results only. They are useful for prioritization, but they are too small to determine the final trained formulation by themselves.

## Raw Files
- Notebook A raw summary: `2026-04-23_notebook_a_t4_smoke.json`
- Notebook B raw summary: `2026-04-23_notebook_b_t4_smoke.json`
- Notebook C raw summary: `2026-04-23_notebook_c_a100_smoke_f06_meta.json`
- Notebook C raw summary: `2026-04-23_notebook_c_a100_smoke_f03.json`
- Notebook C raw summary: `2026-04-23_notebook_c_a100_smoke_f04.json`

## A100 Smoke Train Follow-Up
Three A100 LoRA smoke-train follow-ups were run with a `64`-example train subset and `32`-example validation subset.

| experiment_id | run type | val_accuracy | note |
|---|---|---:|---|
| `f06_multimodal_letter_meta_full` | smoke train | 0.18750 | did not justify promoting the metadata-rich `f06` variant to the default LoRA baseline |
| `f03_multimodal_letter` | smoke train | 0.25000 | cleaner baseline than `f06`, but still too small to over-interpret |
| `f04_candidate_yes_no` | smoke train | 0.37500 | strongest A100 smoke-train result so far |

Interpretation:
- this run was weaker than the earlier zero-shot smoke for `f03_multimodal_letter`
- `f03` still looks like the better default A100 baseline candidate than `f06`
- `f04_candidate_yes_no` was the best zero-shot formulation and also produced the best A100 smoke-train result
- the current full A100 baseline recommendation should move to `f04_candidate_yes_no`

## Next Step
- use `f04_candidate_yes_no` as the default full A100 simple LoRA baseline
- keep `f03_multimodal_letter` as the cleaner restricted-output comparator if you have budget for a second full run
