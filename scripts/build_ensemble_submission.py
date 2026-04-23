from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.submission_utils import ensemble_prediction_files, resolve_test_prediction_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ensemble submission from saved test prediction files.")
    parser.add_argument("--run-dir", dest="run_dirs", action="append", default=[], help="Run output directory containing artifacts/test_predictions.jsonl.")
    parser.add_argument("--predictions", dest="prediction_paths", action="append", default=[], help="Path to a test_predictions.jsonl file.")
    parser.add_argument("--output", help="Optional submission CSV path.")
    parser.add_argument("--ensemble-name", help="If set and --output is omitted, write to outputs/ensembles/<ensemble_name>.csv.")
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    args = parse_args()
    if not args.run_dirs and not args.prediction_paths:
        raise ValueError("Provide at least one --run-dir or --predictions input.")

    prediction_paths = [_resolve_path(path_value) for path_value in args.prediction_paths]
    prediction_paths.extend(resolve_test_prediction_path(_resolve_path(path_value)) for path_value in args.run_dirs)
    output_path = _resolve_path(args.output) if args.output else None
    if output_path is None:
        if not args.ensemble_name:
            raise ValueError("Provide --output or --ensemble-name for the ensemble submission path.")
        output_path = REPO_ROOT / "outputs" / "ensembles" / f"{args.ensemble_name}.csv"

    ensemble_prediction_files(prediction_paths, output_path)
    print(
        json.dumps(
            {
                "prediction_paths": [str(path) for path in prediction_paths],
                "submission_path": str(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
