from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
OUTPUT_DIR = Path("/output")
EXPECTED_CHECKPOINTS = 2


def count_checkpoints(artifacts_dir: Path) -> int:
    return len(list(artifacts_dir.glob("deepset_sum_*.pt")))


def run_command(args: list[str]) -> None:
    completed = subprocess.run(args, cwd=SRC_DIR)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def copy_artifacts_to_output() -> None:
    dest = OUTPUT_DIR / "artifacts"
    dest.mkdir(parents=True, exist_ok=True)
    for pt_file in ARTIFACTS_DIR.glob("deepset_sum_*.pt"):
        shutil.copy2(pt_file, dest / pt_file.name)
    print(f"Copied artifacts to {dest}", flush=True)


def main() -> None:
    checkpoint_count = count_checkpoints(ARTIFACTS_DIR)

    if checkpoint_count < EXPECTED_CHECKPOINTS:
        print(
            f"Found {checkpoint_count} checkpoint(s) in {ARTIFACTS_DIR}. "
            "Training model before running inference.",
            flush=True,
        )
        run_command([sys.executable, "training_pipeline.py"])
        copy_artifacts_to_output()
    else:
        print(
            f"Found {checkpoint_count} checkpoint(s) in {ARTIFACTS_DIR}. "
            "Running inference from existing artifacts.",
            flush=True,
        )

    run_command([sys.executable, "inference_from_artifacts.py", *sys.argv[1:]])

    feature_importance_dir = OUTPUT_DIR / "feature_importance"
    print(f"Generating feature importance plots → {feature_importance_dir}", flush=True)
    run_command([sys.executable, "generate_feature_importance.py", "--output-dir", str(feature_importance_dir)])


if __name__ == "__main__":
    main()
