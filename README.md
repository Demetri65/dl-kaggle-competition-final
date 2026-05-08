# Pixels to Predictions: Visual Science QA (Kaggle Competition)

This repository is the final surface for the DL Spring 2026 Pixels to Predictions Kaggle project. The official final experiment is the comprehensive training and evaluation pipeline in `DL Final Notebook.ipynb`, which achieves 76.66% on the public leaderboard. 

## Public Links
- **GitHub repository:** https://github.com/Demetri65/dl-kaggle-competition-final
- **Public weights:** https://drive.google.com/drive/folders/1y5Q16JywjRZMH_NkxQeShVDV58daA4z3?usp=sharing


## Canonical Final Model
**Training Formulation:** Binary Candidate Yes/No Verification
**Base model:** `HuggingFaceTB/SmolVLM-500M-Instruct`
**LoRA targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `out_proj`
**LoRA Configuration:** Rank=16, Alpha=32, Dropout=0.05

**Prompt Format:**
```text
<image>
Question:
{question text}
Candidate answer:
{choice text}
Reply with Yes or No only.
```

**Training hyperparameters:** 
- 1 epoch (trained on train + validation sets combined)
- Batch size 4, gradient accumulation 1 (Effective BS = 4)
- Learning rate 5e-5, linear schedule, warmup_ratio=0.05, weight decay 0.01
- Full BF16, TF32 enabled
- `image_seq_len` = 64
- Images stretched to 384x384 (BICUBIC)
- Optimizer: AdamW

**Inference Strategy:** Fast scoring using next-token `log P(Yes)`. The model scores each candidate independently, and the choice with the highest yes score is selected.
**Constraints/Guardrails:** Max 5,000,000 trainable parameters. The final model uses ~4,456,448 trainable parameters.

## Reproduction
Run the `DL Final Notebook.ipynb` notebook from start to finish. Ensure the appropriate paths are changed for inputs and outputs if running outside the provided Colab/Kaggle environment. The notebook processes the data, applies the parameter-efficient fine-tuning, runs inference using fast scoring, and generates the final submission file.

## Archive Policy
Historical configurations, the original starter notebook (`starter_notebook.ipynb`), the configuration-driven sweep pipeline (`configs/`, `scripts/`), and early reports are preserved for chronology and report evidence. They are not intended for official reproduction of the final score. The only canonical source for the final 76.66% reproduction is `DL Final Notebook.ipynb`.
