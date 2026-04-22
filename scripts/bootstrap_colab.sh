#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE=/content/.env

cd "${REPO_DIR}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Kaggle credentials not found."
  echo "In Colab, use the notebook FileUpload widget to save your local .env to /content/.env, then rerun this script."
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is missing from ${ENV_FILE}."
  exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 scripts/download_data.py
