from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize results/experiments.csv.")
    parser.add_argument("--results-file", default="results/experiments.csv", help="Path to the results CSV.")
    parser.add_argument("--sort-by", default="val_accuracy", help="Column to sort by.")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending instead of descending.")
    parser.add_argument("--top", type=int, default=20, help="Number of rows to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = REPO_ROOT / args.results_file
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    dataframe = pd.read_csv(results_path)
    if args.sort_by not in dataframe.columns:
        raise ValueError(f"Unknown sort column: {args.sort_by}")

    dataframe = dataframe.sort_values(args.sort_by, ascending=args.ascending)
    columns = [
        "timestamp",
        "experiment_id",
        "parent_experiment_id",
        "seed",
        "run_mode",
        "trainable_parameter_count",
        "formulation_mode",
        "prompt_template",
        "sampling_mode",
        "val_accuracy",
        "two_choice_accuracy",
        "two_choice_support",
        "yes_no_true_false_accuracy",
        "yes_no_true_false_support",
        "eval_artifact_dir",
        "output_dir",
    ]
    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = None
    print(dataframe[columns].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
