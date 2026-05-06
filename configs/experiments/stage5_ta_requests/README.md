# Stage 5 TA Requests

These configs keep the selected `c03_balanced_answer` strategy while adding TA
suggestions that fit the repo's config-driven experiment policy.

| experiment_id | purpose |
|---|---|
| `ta01_dora_caption_context_512_aug` | cap-safe final candidate: candidate yes/no, balanced answer sampling, rank-16 attention DoRA, captions, richer context, 512 aspect-pad images, train-only augmentation |
| `ta02_rank16_dora_qv_mlp_capcheck` | parameter-cap check for rank-16 q/v + MLP DoRA targets |

`ta02` is not the default final submission config. If it exceeds the 5M
trainable-parameter cap, use `ta01` for the final train+val test run.
