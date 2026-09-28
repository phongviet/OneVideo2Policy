"""Run reproducible policy experiments used by the five-page project report.

The runner is deliberately resumable: an evaluation is skipped only when its
report exists and contains the requested number of episodes. Raw checkpoints and
rollouts stay under the ignored ``results/`` tree; the aggregate JSON and figures
are written to ``reports/cvpr/`` for inclusion in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DATA = ROOT / "results/simulation/ball_localization_robust_4000/demonstrations.npz"
BACKGROUND = ROOT / "results/gaussian_scene/hoi4d_ball_to_bowl_animated/object-trajectory.mp4"
RESULTS = ROOT / "results/report_experiments"
PAPER = ROOT / "reports/cvpr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=260928)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    return parser.parse_args()


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "egl"
    environment["PYTHONPATH"] = str(ROOT / "scripts")
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def complete_report(path: Path, episodes: int) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("episodes") == episodes
    except (json.JSONDecodeError, OSError):
        return False


def train(seed: int) -> Path:
    output = RESULTS / "multiseed" / f"seed_{seed}"
    checkpoint = output / "visual_waypoint.pt"
    if checkpoint.exists():
        print(f"reuse checkpoint seed={seed}", flush=True)
        return checkpoint
    print(f"train seed={seed}", flush=True)
    run(
        [
            str(PYTHON),
            "scripts/train_diagnostic_policies.py",
            "--data",
            str(DATA),
            "--output",
            str(output),
            "--models",
            "visual_waypoint",
            "--epochs",
            "60",
            "--batch-size",
            "128",
            "--seed",
            str(seed),
            "--shuffle-episodes",
        ],
        output / "train.log",
    )
    return checkpoint


def evaluate(
    label: str,
    checkpoint: Path,
    episodes: int,
    eval_seed: int,
    *,
    camera_jitter_m: float = 0.0,
    lighting_scale: float = 1.0,
    gaussian: bool = False,
    oracle: bool = False,
) -> Path:
    output = RESULTS / "evaluation" / label
    report = output / "report.json"
    if complete_report(report, episodes):
        print(f"reuse evaluation {label}", flush=True)
        return report
    command = [
        str(PYTHON),
        "scripts/evaluate_ball_bowl_waypoint.py",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--episodes",
        str(episodes),
        "--seed",
        str(eval_seed),
        "--camera-jitter-m",
        str(camera_jitter_m),
        "--lighting-scale",
        str(lighting_scale),
    ]
    if gaussian:
        command += ["--gaussian-background-video", str(BACKGROUND)]
    if oracle:
        command.append("--oracle-ball-position")
    print(f"evaluate {label}", flush=True)
    run(command, output / "run.log")
    return report


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not DATA.exists() or not BACKGROUND.exists():
        raise FileNotFoundError("Required frozen dataset or Gaussian background is missing")
    RESULTS.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    checkpoints = {seed: train(seed) for seed in args.training_seeds}
    reports: dict[str, dict[str, str]] = {"multiseed": {}, "coverage": {}, "severity": {}}

    # E1: independent training seeds, paired combined-shift test episodes.
    for seed, checkpoint in checkpoints.items():
        report = evaluate(
            f"multiseed_seed_{seed}_combined_{args.episodes}",
            checkpoint,
            args.episodes,
            args.eval_seed,
            camera_jitter_m=0.02,
            lighting_scale=0.5,
            gaussian=True,
        )
        reports["multiseed"][str(seed)] = str(report.relative_to(ROOT))

    # E2: cumulative coverage ladder. These are frozen checkpoints from the
    # completed pipeline, all tested on the same combined-shift episodes.
    coverage = {
        "one_demo": ROOT / "results/policy/ball_bowl_one_demo/visual_waypoint.pt",
        "clean_500": ROOT / "results/policy/ball_localization_500/visual_waypoint.pt",
        "appearance_pose_2000": ROOT
        / "results/policy/ball_localization_mixed_pose_2000/visual_waypoint.pt",
        "plus_camera_3000": ROOT
        / "results/policy/ball_localization_robust_3000/visual_waypoint.pt",
        "plus_combined_4000": checkpoints[args.training_seeds[0]],
    }
    for label, checkpoint in coverage.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        report = evaluate(
            f"coverage_{label}_combined_{args.episodes}",
            checkpoint,
            args.episodes,
            args.eval_seed,
            camera_jitter_m=0.02,
            lighting_scale=0.5,
            gaussian=True,
        )
        reports["coverage"][label] = str(report.relative_to(ROOT))
    oracle_report = evaluate(
        f"coverage_oracle_combined_{args.episodes}",
        checkpoints[args.training_seeds[0]],
        args.episodes,
        args.eval_seed,
        camera_jitter_m=0.02,
        lighting_scale=0.5,
        gaussian=True,
        oracle=True,
    )
    reports["coverage"]["oracle"] = str(oracle_report.relative_to(ROOT))

    # E3: controlled camera severity, with all other factors held nominal.
    for jitter_cm in (0, 1, 2, 3, 4):
        label = f"camera_{jitter_cm}cm_{args.episodes}"
        report = evaluate(
            label,
            checkpoints[args.training_seeds[0]],
            args.episodes,
            args.eval_seed,
            camera_jitter_m=jitter_cm / 100,
        )
        reports["severity"][str(jitter_cm)] = str(report.relative_to(ROOT))

    manifest = {
        "episodes_per_evaluation": args.episodes,
        "paired_evaluation_seed": args.eval_seed,
        "training_seeds": args.training_seeds,
        "frozen_training_data": str(DATA.relative_to(ROOT)),
        "gaussian_background": str(BACKGROUND.relative_to(ROOT)),
        "reports": reports,
    }
    manifest_path = RESULTS / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(manifest_path, PAPER / "experiment-manifest.json")
    subprocess.run(
        [str(PYTHON), "scripts/summarize_report_experiments.py", "--manifest", str(manifest_path)],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
