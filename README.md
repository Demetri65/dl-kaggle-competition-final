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
