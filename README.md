# dl-kaggle-competition-final

Foundation scaffold for a Kaggle competition project that needs to work in three places without changing path variables:

- local development
- Google Colab
- GitHub collaboration

GitHub repo:

- `https://github.com/Demetri65/dl-kaggle-competition-final`

## Working Contract

Keep these rules stable from day one:

- GitHub stores code, notebooks, and lightweight project files.
- `data/` is the single repo-local data directory.
- Keep the real `.env` local only.
- In VSCode Colab, the setup cell syncs the latest `origin/main` and writes your uploaded `.env` to `/content/.env`.
- Model weights, checkpoints, outputs, and submissions stay out of Git.

That matches the existing notebook, which already expects:

```python
REPO_ROOT = Path("/content/dl-kaggle-competition-final")
DATA_DIR = REPO_ROOT / "data"
```

## Files Added

- `requirements.txt`: shared Python dependencies for local and Colab.
- `scripts/download_data.py`: downloads the competition data into repo-local `data/` with the official Kaggle CLI using `KAGGLE_API_TOKEN`.
- `scripts/bootstrap_colab.sh`: reads `KAGGLE_API_TOKEN` from an uploaded `.env`, installs dependencies, and downloads the data without using `kaggle.json`.
- `.env.example`: template for the modern Kaggle API token used in Colab.
- `.gitignore`: keeps data, credentials, and training artifacts out of Git.

## Current State

- The repo is initialized and pushed to GitHub on `main`.
- GitHub is the source of truth for code, notebooks, and config files.
- `data/`, `.env`, legacy `kaggle.json`, checkpoints, and outputs stay out of Git.

## Remaining Setup Flow

1. Both partners clone the same repo locally.
2. Open the notebook from GitHub in Colab when you want GPU time.
3. Run the setup cell once to show the upload widget, choose your local `.env`, then rerun the same cell.
4. Commit code, notebooks, and configs only. Never commit `data/`, `.env`, or legacy `kaggle.json`.

## Local Setup

```bash
cp .env.example .env
# edit .env and set KAGGLE_API_TOKEN from Kaggle Settings > API > Generate New Token
set -a
source .env
set +a
python3 -m pip install -r requirements.txt
python3 scripts/download_data.py
```

## Colab Setup

Prepare a local `.env` from `.env.example` and set:

```bash
KAGGLE_API_TOKEN=...
```

That `.env` stays local and should not be committed.
The setup cell writes the selected file into the runtime as `/content/.env`.
The repo uses the official Kaggle CLI with `KAGGLE_API_TOKEN`; `kaggle.json` is not required.
Each setup-cell run syncs the Colab checkout to the latest `origin/main`.

In the notebook:
1. Run the setup cell once to display the `.env` upload widget.
2. Choose your local `.env`.
3. Rerun the same setup cell to save the file, sync the repo, and finish bootstrap.

## Partner Workflow

Use GitHub as the source of truth:

- `main` stays runnable.
- Each person works on short-lived branches.
- Merge through pull requests, even if the team is small.
- Avoid both editing the same notebook at the same time.

If notebook conflicts become a problem, the next step is to move reusable logic into Python modules and keep notebooks thin. That can happen later without changing the `data/` contract.

## Experiment Framework

The repo now includes a config-driven train/eval pipeline for SmolVLM ablations:

- Base config: `configs/base.yaml`
- Experiment overrides: `configs/experiments/<stage>/*.yaml`
- Single-run entrypoint: `scripts/run_experiment.py`
- Stage sweep entrypoint: `scripts/run_stage.py`
- Final submission utility: `scripts/build_final_submission.py`
- Ensemble submission utility: `scripts/build_ensemble_submission.py`
- Results summary: `scripts/summarize_results.py`

Experiment configs are organized into stage folders, but you can still run them by unique experiment id such as `f03_multimodal_letter`.
Config composition supports a primary `parent_experiment_id` plus optional `merge_experiment_ids` overlays.

### Run One Experiment

```bash
python3 scripts/run_experiment.py --experiment f02_multimodal_index
```

Example with a small override:

```bash
python3 scripts/run_experiment.py \
  --experiment f03_multimodal_letter \
  --set training.epochs=2 \
  --seed 7
```

Stage 0 eval-only screening run:

```bash
python3 scripts/run_experiment.py \
  --experiment f02_multimodal_index \
  --set lora.enabled=false \
  --set training.epochs=0
```

Artifact-backed eval-only rerun:

```bash
python3 scripts/run_experiment.py \
  --experiment f02_multimodal_index \
  --set training.epochs=0 \
  --set runtime.eval_artifact_dir=outputs/f02_multimodal_index_seed42
```

### Run A Whole Stage

```bash
python3 scripts/run_stage.py \
  f02_multimodal_index f03_multimodal_letter f04_candidate_yes_no
```

Example stage sweep that also writes test predictions and an averaged ensemble submission:

```bash
python3 scripts/run_stage.py \
  l01_seed1 l01_seed2 l01_seed3 \
  --predict-test \
  --ensemble-name seed_ensemble
```

Stage sweep with best-parent selection and child-config promotion:

```bash
python3 scripts/run_stage.py \
  f02_multimodal_index f03_multimodal_letter f04_candidate_yes_no \
  --stage-name stage0_screen \
  --select-best-by val_accuracy \
  --promote-best-to stage1_parent_plus_lr \
  --child-set training.learning_rate=1.0e-4
```

### Summarize Results

```bash
python3 scripts/summarize_results.py --sort-by val_accuracy --top 10
```

### Build Submission CSVs

Build a final submission from one saved run:

```bash
python3 scripts/build_final_submission.py \
  --run-dir outputs/f02_multimodal_index_seed42
```

Build an ensemble submission from multiple saved runs:

```bash
python3 scripts/build_ensemble_submission.py \
  --run-dir outputs/l01_seed1_seed1 \
  --run-dir outputs/l01_seed2_seed2 \
  --run-dir outputs/l01_seed3_seed3 \
  --ensemble-name seed_ensemble
```

### Outputs

- Per-run artifacts are written under `outputs/<experiment_id>_seed<seed>/`
- Validation and test prediction artifacts are written to `artifacts/*.jsonl`
- Central run metadata is appended to `results/experiments.csv`
- Stage summaries are always written under `outputs/stages/<stage_name>/`
- Child configs can inherit via `parent_experiment_id` and optionally merge extra overlays via `merge_experiment_ids`
