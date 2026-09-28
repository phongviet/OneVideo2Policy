"""Plot the committed HOI4D cross-action SAM2 and prompt-ablation results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/experiments/hoi4d-cross-action-sam2.json"
ABLATION = ROOT / "docs/experiments/hoi4d-articulated-prompt-ablation.json"
OUTPUT = ROOT / "docs/assets/hoi4d-cross-action-sam2.png"


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["sequences"]
    ablation = {
        row["name"]: row for row in json.loads(ABLATION.read_text(encoding="utf-8"))["results"]
    }
    names = [row["name"] for row in baseline]
    single = np.asarray([row["mean_iou"] for row in baseline])
    part_aware = np.asarray(
        [ablation.get(name, {}).get("part_aware_five_point_mean_iou", np.nan) for name in names]
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axis = plt.subplots(figsize=(7.4, 3.3), constrained_layout=True)
    positions = np.arange(len(names))
    width = 0.34
    bars = axis.bar(
        positions - width / 2,
        single,
        width,
        label="one positive point",
        color="#617D98",
    )
    alternate = axis.bar(
        positions + width / 2,
        np.nan_to_num(part_aware, nan=0),
        width,
        label="part-aware: +4 negatives",
        color="#2A9D8F",
    )
    for index, value in enumerate(part_aware):
        if np.isnan(value):
            alternate[index].set_visible(False)
    axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    for index, value in enumerate(part_aware):
        if not np.isnan(value):
            axis.text(index + width / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Mean mask IoU (100 frames)")
    axis.set_title(
        "SAM2.1 Small generalizes to rigid objects; part prompting is task dependent",
        loc="left",
        fontweight="bold",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        frameon=False,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
