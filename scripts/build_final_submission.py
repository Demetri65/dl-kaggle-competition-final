from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.submission_utils import resolve_test_prediction_path, save_submission_from_prediction_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a final submission CSV from one saved test prediction file.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run-dir", help="Run output directory containing artifacts/test_predictions.jsonl.")
    inputs.add_argument("--predictions", help="Path to a test_predictions.jsonl file.")
    parser.add_argument("--output", help="Optional submission CSV path.")
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _default_output_path(run_dir: Path | None, prediction_path: Path) -> Path:
    if run_dir is not None:
        return run_dir.parent / "submissions" / f"{run_dir.name}.csv"
    return prediction_path.with_suffix(".csv")


def main() -> None:
    args = parse_args()
    run_dir = _resolve_path(args.run_dir) if args.run_dir else None
    prediction_path = _resolve_path(args.predictions) if args.predictions else None
    if run_dir is not None:
        prediction_path = resolve_test_prediction_path(run_dir)
    assert prediction_path is not None

    output_path = _resolve_path(args.output) if args.output else _default_output_path(run_dir, prediction_path)
    save_submission_from_prediction_file(prediction_path, output_path)
    print(
        json.dumps(
            {
                "prediction_path": str(prediction_path),
                "submission_path": str(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
