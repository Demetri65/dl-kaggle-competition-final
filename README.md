# Pixels to Predictions: Visual Science Question Answering

A multimodal fine-tuning and evaluation pipeline that adapts `HuggingFaceTB/SmolVLM-500M-Instruct` with LoRA for multiple-choice visual science questions.

## Result

| Model | Formulation | Adaptation | Public leaderboard |
| --- | --- | --- | ---: |
| SmolVLM-500M-Instruct | Binary candidate verification | LoRA | 76.66% |

`DL Final Notebook.ipynb` is the canonical final experiment for this result. Earlier configuration-driven experiments and reports remain in the repository as development history.

## Problem formulation

Each answer option is evaluated independently as a binary verification problem. Given an image, question, and candidate answer, the model predicts whether the candidate is correct:

```text
<image>
Question:
{question text}
Candidate answer:
{choice text}
Is the candidate answer correct? Reply with Yes or No only.
```

At inference time, the pipeline scores the next-token log-probability of `Yes` for every candidate and selects the highest-scoring option. This converts a variable-size multiple-choice task into a consistent binary scoring procedure.

## Model and training configuration

The canonical notebook uses:

- Base model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- LoRA rank 16, alpha 32, and dropout 0.05
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, and `out_proj`
- 4,456,448 trainable parameters, below the 5,000,000-parameter competition limit
- One epoch over the combined training and validation splits
- Per-device batch size 4 with one gradient-accumulation step
- Learning rate `5e-5`, linear schedule, warmup ratio `0.05`, and weight decay `0.01`
- Images stretched to 384 × 384 with bicubic resampling and an image sequence length of 64
- Mixed precision selected from BF16 or FP16 according to accelerator support, with TF32 enabled in the training arguments

## Evaluation pipeline

The final notebook loads the trained LoRA adapter, scores every candidate answer, chooses the candidate with the highest `Yes` score, and writes the competition submission. The repository also contains a configuration-driven experiment pipeline for prompt, image-processing, sampling, and fine-tuning studies, plus saved smoke-test and training summaries under `results/`.

## Reproduce the final experiment

1. Install the dependencies in `requirements.txt` in an accelerator-enabled Python environment.
2. Download the competition data and update `DATA_DIR` in `DL Final Notebook.ipynb` to point to it.
3. Run `DL Final Notebook.ipynb` from start to finish to fine-tune the adapter, score the test examples, and generate the submission.

The final adapter weights are available in this [public Google Drive folder](https://drive.google.com/drive/folders/1y5Q16JywjRZMH_NkxQeShVDV58daA4z3?usp=sharing).

Full reproduction requires the external competition dataset, model weights, and suitable accelerator hardware. Paths and output locations may need adjustment outside the original notebook environment.

## Repository structure

| Path | Purpose |
| --- | --- |
| `DL Final Notebook.ipynb` | Canonical final training, evaluation, and submission workflow |
| `configs/` | Base, experiment, and archived configurations from iterative studies |
| `src/` | Reusable data, prompting, modeling, training, evaluation, and scoring modules |
| `scripts/` | Experiment, reporting, data, and submission utilities |
| `results/` | Saved smoke-test and training summaries |
| `tests/` | CPU-safe unit tests for the reusable pipeline |
| `notebooks/` | Earlier formulation, inference, and sampling experiments |

## Tests

Run the unit suite with:

```bash
python -m pytest -q
```

The tests validate configuration loading, prompt construction, scoring, evaluation artifacts, result records, stage summaries, and submission construction without downloading model weights or reproducing the leaderboard run.

## Limitations

- The 76.66% result is a public leaderboard score, not an independent benchmark or estimate of generalization beyond the competition data.
- The competition dataset is not included in this repository.
- Reproducing fine-tuning requires external model assets and suitable accelerator hardware; the unit tests cover pipeline behavior only.
- Historical configurations, notebooks, and reports document the development process but are not the canonical recipe for the final score.
