from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


COMPETITION = "pixels-to-predictions"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REQUIRED_FILES = ("train.csv", "val.csv", "test.csv")
EXPECTED_ENTRIES = (*REQUIRED_FILES, "sample_submission.csv", "images")
MIN_KAGGLE_VERSION = (2, 0, 0)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kaggle competition data into the repo-local data directory."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download into the output directory.",
    )
    return parser.parse_args()


def parse_version(version_output: str) -> tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_output)
    if not match:
        raise RuntimeError(f"Unable to parse Kaggle CLI version from: {version_output!r}")
    return tuple(int(part) for part in match.groups())


def build_kaggle_env(config_dir: str) -> dict[str, str]:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise EnvironmentError("KAGGLE_API_TOKEN must be set in the environment before downloading data.")

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["KAGGLE_CONFIG_DIR"] = config_dir
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    return env


def verify_kaggle_cli(env: dict[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kaggle.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    version_output = (result.stdout or result.stderr).strip()
    version = parse_version(version_output)
    if version < MIN_KAGGLE_VERSION:
        raise RuntimeError(
            "Kaggle CLI is too old for access-token auth. "
            f"Found {version_output}, need >= {'.'.join(str(part) for part in MIN_KAGGLE_VERSION)}."
        )

    print(f"Using Kaggle CLI {version_output}")


def extract_archives(root: Path) -> None:
    processed: set[Path] = set()
    while True:
        archives = sorted(path for path in root.rglob("*.zip") if path not in processed)
        if not archives:
            return

        for zip_path in archives:
            print(f"Extracting {zip_path.relative_to(root)}")
            if not zipfile.is_zipfile(zip_path):
                raise zipfile.BadZipFile(f"Invalid zip archive: {zip_path}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(zip_path.parent)
            processed.add(zip_path)


def run_kaggle_download(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            "Kaggle competition download failed. "
            "Confirm that KAGGLE_API_TOKEN is valid and that your Kaggle account has accepted "
            f"the competition rules for '{COMPETITION}'."
        )


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaggle-config-") as config_dir:
        kaggle_env = build_kaggle_env(config_dir)
        verify_kaggle_cli(kaggle_env)

        command = [
            sys.executable,
            "-m",
            "kaggle.cli",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(DATA_DIR),
        ]
        if args.force:
            command.append("-o")

        run_kaggle_download(command, kaggle_env)

        try:
            extract_archives(DATA_DIR)
        except zipfile.BadZipFile as exc:
            if args.force:
                raise RuntimeError(
                    f"Encountered an invalid zip archive after a forced download: {exc}"
                ) from exc
            print(
                f"Detected an invalid zip archive under {DATA_DIR}. "
                "Retrying the Kaggle download once with overwrite enabled.",
                file=sys.stderr,
            )
            retry_command = [*command, "-o"] if "-o" not in command else command
            run_kaggle_download(retry_command, kaggle_env)
            extract_archives(DATA_DIR)

    nested_dirs = [path for path in DATA_DIR.iterdir() if path.is_dir()]
    if len(nested_dirs) == 1 and not any((DATA_DIR / name).exists() for name in EXPECTED_ENTRIES):
        nested_root = nested_dirs[0]
        if any((nested_root / name).exists() for name in EXPECTED_ENTRIES):
            print(f"Normalizing extracted layout from {nested_root.name}/ into {DATA_DIR}")
            for child in nested_root.iterdir():
                shutil.move(str(child), DATA_DIR / child.name)
            nested_root.rmdir()

    missing_files = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
    if missing_files:
        discovered = sorted(path.relative_to(DATA_DIR).as_posix() for path in DATA_DIR.rglob("*.csv"))
        raise FileNotFoundError(
            "Competition download did not produce the expected files in "
            f"{DATA_DIR}. Missing {missing_files}. Found CSV files: {discovered or 'none'}. "
            "Confirm that your Kaggle credentials are valid and that your Kaggle account has accepted "
            f"the competition rules for '{COMPETITION}'."
        )

    image_count = sum(1 for path in DATA_DIR.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if image_count == 0:
        raise FileNotFoundError(
            f"No extracted image files were found under {DATA_DIR}. "
            "The competition data download appears incomplete."
        )

    print(f"Competition files downloaded to: {DATA_DIR}")


if __name__ == "__main__":
    main()
