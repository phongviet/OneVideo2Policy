"""Aggregate report experiments and create paper-ready figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports/cvpr"


def wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / episodes
    denominator = 1 + z * z / episodes
    center = (proportion + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / episodes + z * z / (4 * episodes * episodes)
    ) / denominator
    return [center - radius, center + radius]


def read_report(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def exact_mcnemar(first: list[bool], second: list[bool]) -> dict[str, float | int]:
    """Two-sided exact McNemar test for paired binary outcomes."""
    first_only = sum(a and not b for a, b in zip(first, second, strict=True))
    second_only = sum(b and not a for a, b in zip(first, second, strict=True))
    discordant = first_only + second_only
    if discordant == 0:
        probability = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
        probability = min(1.0, 2 * tail / (2**discordant))
    return {
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "two_sided_exact_p": probability,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = manifest["reports"]
    aggregate: dict[str, object] = {
        "protocol": manifest,
        "multiseed": {},
        "coverage": {},
        "severity": {},
    }

    seed_rates = []
    seed_errors = []
    for seed, path in groups["multiseed"].items():
        report = read_report(path)
        rate = report["successes"] / report["episodes"]
        seed_rates.append(rate)
        seed_errors.append(report["median_initial_xy_localization_error_m"])
        aggregate["multiseed"][seed] = {
            "successes": report["successes"],
            "episodes": report["episodes"],
            "success_rate": rate,
            "wilson_95": wilson(report["successes"], report["episodes"]),
            "median_initial_xy_error_m": report["median_initial_xy_localization_error_m"],
        }
    total_episodes = len(seed_rates) * manifest["episodes_per_evaluation"]
    total_successes = sum(item["successes"] for item in aggregate["multiseed"].values())
    aggregate["multiseed_summary"] = {
        "mean_seed_success_rate": float(np.mean(seed_rates)),
        "std_seed_success_rate": float(np.std(seed_rates, ddof=1)),
        "t95_seed_mean": [
            float(
                np.mean(seed_rates)
                - 2.776445 * np.std(seed_rates, ddof=1) / np.sqrt(len(seed_rates))
            ),
            float(
                np.mean(seed_rates)
                + 2.776445 * np.std(seed_rates, ddof=1) / np.sqrt(len(seed_rates))
            ),
        ],
        "min_seed_success_rate": float(np.min(seed_rates)),
        "max_seed_success_rate": float(np.max(seed_rates)),
        "pooled_successes": int(total_successes),
        "pooled_episodes": int(total_episodes),
        "pooled_wilson_95": wilson(total_successes, total_episodes),
        "mean_seed_median_xy_error_m": float(np.mean(seed_errors)),
    }

    paired_outcomes: dict[str, dict[str, list[bool]]] = {"coverage": {}, "severity": {}}
    for group in ("coverage", "severity"):
        for label, path in groups[group].items():
            report = read_report(path)
            paired_outcomes[group][label] = [
                bool(episode["success"]) for episode in report["episodes_detail"]
            ]
            aggregate[group][label] = {
                "successes": report["successes"],
                "episodes": report["episodes"],
                "success_rate": report["successes"] / report["episodes"],
                "wilson_95": wilson(report["successes"], report["episodes"]),
                "median_initial_xy_error_m": report["median_initial_xy_localization_error_m"],
            }

    full = paired_outcomes["coverage"]["plus_combined_4000"]
    aggregate["coverage_paired_tests"] = {
        label: exact_mcnemar(outcomes, full)
        for label, outcomes in paired_outcomes["coverage"].items()
        if label != "plus_combined_4000"
    }

    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "experiment-results.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    summary = aggregate["multiseed_summary"]
    macro_lines = [
        f"\\newcommand{{\\SeedMean}}{{{100 * summary['mean_seed_success_rate']:.1f}}}",
        f"\\newcommand{{\\SeedStd}}{{{100 * summary['std_seed_success_rate']:.1f}}}",
        f"\\newcommand{{\\SeedMin}}{{{100 * summary['min_seed_success_rate']:.0f}}}",
        f"\\newcommand{{\\SeedMax}}{{{100 * summary['max_seed_success_rate']:.0f}}}",
        f"\\newcommand{{\\SeedPooledSuccess}}{{{summary['pooled_successes']}}}",
        f"\\newcommand{{\\SeedPooledEpisodes}}{{{summary['pooled_episodes']}}}",
    ]
    coverage_commands = {
        "one_demo": "OneDemo",
        "clean_500": "CleanFiveHundred",
        "appearance_pose_2000": "AppearancePoseTwoThousand",
        "plus_camera_3000": "PlusCameraThreeThousand",
        "plus_combined_4000": "PlusCombinedFourThousand",
        "oracle": "Oracle",
    }
    for label, value in aggregate["coverage"].items():
        command = coverage_commands[label]
        macro_lines.append(
            f"\\newcommand{{\\Coverage{command}}}"
            f"{{{100 * value['success_rate']:.0f}}}"
        )
    number_names = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four"}
    for label, value in aggregate["severity"].items():
        macro_lines.append(
            f"\\newcommand{{\\Camera{number_names[label]}cm}}"
            f"{{{100 * value['success_rate']:.0f}}}"
        )
    (PAPER / "results_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 8, "figure.dpi": 180})
    coverage_order = [
        "one_demo",
        "clean_500",
        "appearance_pose_2000",
        "plus_camera_3000",
        "plus_combined_4000",
        "oracle",
    ]
    coverage_labels = [
        "1 demo",
        "Clean\n500",
        "+3DGS/pose\n2000",
        "+camera\n3000",
        "+light\n4000",
        "Oracle",
    ]
    values = [100 * aggregate["coverage"][key]["success_rate"] for key in coverage_order]
    fig, axis = plt.subplots(figsize=(3.25, 1.9))
    colors = ["#9aa0a6"] * 2 + ["#4e79a7"] * 3 + ["#59a14f"]
    bars = axis.bar(range(len(values)), values, color=colors)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Success rate (%)")
    axis.set_xticks(range(len(values)), coverage_labels)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}",
            ha="center",
            fontsize=7,
        )
    fig.tight_layout(pad=0.4)
    fig.savefig(PAPER / "coverage-ablation.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "coverage-ablation.png", bbox_inches="tight")
    plt.close(fig)

    jitters = sorted(int(key) for key in aggregate["severity"])
    rates = [100 * aggregate["severity"][str(key)]["success_rate"] for key in jitters]
    errors = [
        1000 * aggregate["severity"][str(key)]["median_initial_xy_error_m"]
        for key in jitters
    ]
    fig, axis = plt.subplots(figsize=(3.25, 1.9))
    axis.plot(jitters, rates, "o-", color="#4e79a7", label="success")
    axis.set_xlabel("Camera translation range (cm)")
    axis.set_ylabel("Success rate (%)", color="#4e79a7")
    axis.set_ylim(0, 105)
    axis.tick_params(axis="y", labelcolor="#4e79a7")
    second = axis.twinx()
    second.plot(jitters, errors, "s--", color="#e15759", label="XY error")
    second.set_ylabel("Median initial XY error (mm)", color="#e15759")
    second.tick_params(axis="y", labelcolor="#e15759")
    axis.grid(alpha=0.25)
    fig.tight_layout(pad=0.4)
    fig.savefig(PAPER / "camera-severity.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "camera-severity.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(aggregate["multiseed_summary"], indent=2))


if __name__ == "__main__":
    main()
