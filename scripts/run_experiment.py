from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_experiment_config
from src.training.train import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one config-driven SmolVLM experiment.")
    parser.add_argument("--experiment", required=True, help="Unique experiment id or nested config reference from configs/experiments.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Config override in key=value form.")
    parser.add_argument("--seed", type=int, help="Optional seed override.")
    parser.add_argument("--output-dir", help="Optional output root override.")
    parser.add_argument("--predict-test", action="store_true", help="Also run test prediction and write submission.csv.")
    parser.add_argument("--final-retrain", action="store_true", help="Train on train+val and skip validation metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    if args.predict_test:
        overrides.append("runtime.predict_test=true")
    if args.final_retrain:
        overrides.append("runtime.final_retrain=true")

    config = load_experiment_config(
        repo_root=REPO_ROOT,
        experiment_id=args.experiment,
        cli_overrides=overrides,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    summary = run_experiment(REPO_ROOT, config)
    payload = {
        "experiment_id": config.experiment_id,
        "run_name": config.run_name,
        "output_dir": str(config.output_dir),
        **summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
