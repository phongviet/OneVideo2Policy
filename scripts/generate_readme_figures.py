#!/usr/bin/env python3
"""Build the compact, evidence-backed figures used by the README."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
sys.path.insert(0, str(ROOT / "scripts"))

BLUE = "#2864DC"
ORANGE = "#D45A16"
GREEN = "#2E7D32"
RED = "#C62828"
INK = "#172033"
MUTED = "#5F6B7A"
PALE = "#F4F7FB"


def load_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(ROOT / path).convert("RGB"))


def center_crop(image: np.ndarray, ratio: float = 1.55) -> np.ndarray:
    height, width = image.shape[:2]
    current = width / height
    if current > ratio:
        new_width = int(height * ratio)
        start = (width - new_width) // 2
        return image[:, start : start + new_width]
    new_height = int(width / ratio)
    start = (height - new_height) // 2
    return image[start : start + new_height]


def save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline_figure() -> None:
    frame = load_image("data/interim/hoi4d_ball_to_bowl_gate/frames/000000.jpg")
    mask = np.asarray(
        Image.open(ROOT / "data/interim/hoi4d_ball_to_bowl_gate/masks/target/000000.png")
        .convert("L")
        .resize((frame.shape[1], frame.shape[0]))
    )
    perception = frame.copy()
    perception[mask > 127] = (
        0.42 * perception[mask > 127] + 0.58 * np.array([40, 100, 220])
    ).astype(np.uint8)
    images = [
        frame,
        perception,
        load_image("results/gaussian_scene/hoi4d_ball_to_bowl_fused/held-out-render.png"),
        load_image("results/gaussian_scene/metric_robot_registration/registered-simulator.png"),
    ]
    titles = [
        "1  RGB-D demonstration",
        "2  Segment and track",
        "3  Metric scene + SE(3) skill",
        "4  Generate robot data",
        "5  Train and evaluate",
    ]
    subtitles = [
        "one human sequence",
        "objects and motion",
        "3DGS appearance + metric physics",
        "pose, camera, and lighting shifts",
        "closed-loop simulation",
    ]

    fig = plt.figure(figsize=(15.2, 3.25))
    grid = fig.add_gridspec(1, 5, wspace=0.23)
    axes = [fig.add_subplot(grid[0, index]) for index in range(5)]
    for index, axis in enumerate(axes):
        axis.set_axis_off()
        axis.add_patch(
            FancyBboxPatch(
                (0, 0),
                1,
                1,
                transform=axis.transAxes,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor="white",
                edgecolor="#CBD4E1",
                linewidth=1.2,
                zorder=-10,
                clip_on=False,
            )
        )
        axis.text(
            0.05,
            0.93,
            titles[index],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
        axis.text(
            0.05,
            0.81,
            subtitles[index],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=MUTED,
        )
        if index < 4:
            image_axis = axis.inset_axes([0.03, 0.05, 0.94, 0.64])
            image_axis.imshow(center_crop(images[index]))
            image_axis.set_axis_off()
        else:
            inset = axis.inset_axes([0.12, 0.12, 0.78, 0.54])
            labels = ["1 demo", "+ pose", "+ camera", "+ combined"]
            values = [6, 20, 50, 82]
            inset.bar(range(4), values, color=["#A8B3C5", "#8FA0B9", "#6E8FCC", BLUE])
            inset.set_ylim(0, 100)
            inset.set_xticks(range(4), labels, rotation=24, ha="right", fontsize=6.8)
            inset.set_yticks([0, 50, 100])
            inset.tick_params(axis="y", labelsize=7)
            inset.set_ylabel("success (%)", fontsize=7)
            inset.grid(axis="y", alpha=0.25)
            inset.spines[["top", "right"]].set_visible(False)
            for position, value in enumerate(values):
                inset.text(position, value + 3, str(value), ha="center", fontsize=7, color=INK)
    for left, right in zip(axes[:-1], axes[1:], strict=True):
        start = left.get_position().x1 + 0.003
        end = right.get_position().x0 - 0.003
        y = (left.get_position().y0 + left.get_position().y1) / 2
        fig.add_artist(
            FancyArrowPatch(
                (start, y),
                (end, y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                color=ORANGE,
            )
        )
    save(fig, "pipeline-overview.png")


def tracking_figure() -> None:
    import cv2
    from ablate_record3d_rotation import (
        ROI_PLACED,
        ROI_REFERENCE,
        apply_transform,
        crop,
        mask_only_transform,
    )

    from onevideo2policy.video.record3d import Record3DArchive

    archive = Record3DArchive(ROOT / "data/raw/sources/EM1-0406.r3d")
    reference = crop(archive.read_rgb(0), ROI_REFERENCE)
    observed = crop(archive.read_rgb(330), ROI_PLACED)
    detector = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.015)
    key_a, des_a = detector.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None)
    key_b, des_b = detector.detectAndCompute(cv2.cvtColor(observed, cv2.COLOR_RGB2GRAY), None)
    pairs = cv2.BFMatcher().knnMatch(des_a, des_b, k=2)
    matches = [a for a, b in pairs if a.distance < 0.78 * b.distance]
    first = np.float32([key_a[item.queryIdx].pt for item in matches])
    second = np.float32([key_b[item.trainIdx].pt for item in matches])
    _, inliers = cv2.estimateAffinePartial2D(
        first, second, method=cv2.RANSAC, ransacReprojThreshold=3
    )
    first, second = first[inliers[:, 0] > 0], second[inliers[:, 0] > 0]
    order = np.argsort(first[:, 0] + 0.01 * first[:, 1])
    held_indices = order[::3]
    fit_indices = np.setdiff1d(order, held_indices)
    tracked, _ = cv2.estimateAffinePartial2D(
        first[fit_indices], second[fit_indices], method=cv2.RANSAC, ransacReprojThreshold=3
    )
    held, target = first[held_indices], second[held_indices]
    baseline_prediction = apply_transform(held, mask_only_transform(reference, observed))
    tracked_prediction = apply_transform(held, tracked)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.65))
    panels = [
        (reference, "Reference", "held-out points", None),
        (observed, "Mask-only transform", "35.6 px error", baseline_prediction),
        (observed, "Tracked transform", "0.3 px error", tracked_prediction),
    ]
    for index, (axis, (image, title, metric, prediction)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
        if index == 0:
            axis.scatter(
                held[:, 0], held[:, 1], s=24, c="#F5C542", edgecolors="black", linewidths=0.4
            )
        else:
            axis.scatter(
                target[:, 0], target[:, 1], s=28, c=GREEN, edgecolors="white", linewidths=0.5
            )
            color = RED if index == 1 else BLUE
            for truth, predicted in zip(target, prediction, strict=True):
                axis.plot([truth[0], predicted[0]], [truth[1], predicted[1]], color=color, lw=2)
                axis.scatter(predicted[0], predicted[1], s=17, c=color, marker="x")
        axis.text(
            0.5,
            -0.10,
            metric,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=RED if index == 1 else (BLUE if index == 2 else MUTED),
        )
    fig.suptitle(
        "Correspondence tracking resolves rotation", fontsize=15, fontweight="bold", color=INK
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.15, top=0.82, wspace=0.08)
    save(fig, "em1-0406-tracking-ablation.png")


def h2o_figure() -> None:
    place_path = ROOT / "results/h2o_oracle_place/oracle-place-trajectory.npz"
    pour_path = ROOT / "results/h2o_oracle_pour/oracle-pour-trajectory.npz"
    with np.load(place_path) as data:
        object_world = data["object_to_world"]
        object_hand = data["object_to_hand"]
        place_motion = data["object_relative_motion"]
    with np.load(pour_path) as data:
        pour_motion = data["object_relative_motion"]
    hand_world = object_world @ object_hand
    object_xyz = 100 * (object_world[:, :3, 3] - object_world[0, :3, 3])
    hand_xyz = 100 * (hand_world[:, :3, 3] - hand_world[0, :3, 3])
    place_xyz = 100 * place_motion[:, :3, 3]

    demos = np.load(ROOT / "results/h2o_place_demos_100/demonstrations.npz")
    robot_a = demos["pose_images"][100]
    robot_b = demos["pose_images_front"][100]
    robot_pair = np.hstack((robot_a, robot_b))

    fig = plt.figure(figsize=(15.4, 4.1))
    grid = fig.add_gridspec(1, 4, wspace=0.30)
    axes = [
        fig.add_subplot(grid[0], projection="3d"),
        fig.add_subplot(grid[1], projection="3d"),
        fig.add_subplot(grid[2]),
        fig.add_subplot(grid[3]),
    ]
    ax = axes[0]
    ax.plot(*object_xyz.T, color=ORANGE, lw=2.5, label="object")
    ax.plot(*hand_xyz.T, color=BLUE, lw=2.0, label="hand")
    ax.scatter(*object_xyz[[0, -1]].T, color=[GREEN, RED], s=28)
    ax.set_title("1  Human demonstration", fontsize=11.5, fontweight="bold", color=INK)
    ax.text2D(
        0.5,
        -0.10,
        "H2O measured pose sequence · cm",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)

    ax = axes[1]
    ax.plot(*place_xyz.T, color=BLUE, lw=2.5, label="Place translation")
    sample = np.linspace(0, len(pour_motion) - 1, 5, dtype=int)
    base = 100 * pour_motion[:, :3, 3]
    ax.plot(*base.T, color=ORANGE, lw=2.2, label="Pour SE(3)")
    for idx in sample:
        origin = base[idx]
        direction = 2.7 * pour_motion[idx, :3, 2]
        ax.quiver(*origin, *direction, color=ORANGE, linewidth=1.3, arrow_length_ratio=0.25)
    ax.set_title("2  Recovered skill", fontsize=11.5, fontweight="bold", color=INK)
    ax.text2D(
        0.5,
        -0.10,
        "object-relative SE(3) · cm",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    ax.legend(loc="upper left", fontsize=7.2, frameon=False)

    for axis in axes[:2]:
        axis.tick_params(labelsize=6)
        axis.grid(alpha=0.25)

    ax = axes[2]
    ax.imshow(robot_pair)
    ax.set_axis_off()
    ax.set_title("3  Panda retargeting", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(0.25, -0.06, "agent view", transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
    ax.text(0.75, -0.06, "front view", transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
    ax.text(
        0.5,
        -0.22,
        "robosuite simulation",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=MUTED,
    )

    ax = axes[3]
    labels = ["Place\ndemos", "Place\nexecution", "Pour\nSE(3)", "Pour\nXYZ"]
    values = [100, 100, 100, 0]
    colors = [GREEN, GREEN, BLUE, "#A8B3C5"]
    bars = ax.bar(range(4), values, color=colors, width=0.68)
    ax.set_ylim(0, 112)
    ax.set_xticks(range(4), labels, fontsize=8)
    ax.set_yticks([0, 50, 100], ["0", "50", "100"], fontsize=8)
    ax.set_ylabel("success (%)", fontsize=8.5)
    ax.set_title("4  Closed-loop results", fontsize=11.5, fontweight="bold", color=INK)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    counts = ["100/100", "20/20", "20/20", "0/20"]
    for bar, label in zip(bars, counts, strict=True):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(height + 3, 4),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
    fig.suptitle(
        "From H2O motion to robot execution", fontsize=16, fontweight="bold", color=INK, y=1.01
    )
    save(fig, "h2o-retargeting-results.png")


def policy_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.4), gridspec_kw={"width_ratios": [1.12, 1]})
    ax = axes[0]
    labels = ["1 demo", "Clean\n500", "+ pose\n2,000", "+ camera\n3,000", "+ combined\n4,000"]
    values = [6, 2, 20, 50, 82]
    bars = ax.bar(range(5), values, color=["#A8B3C5", "#A8B3C5", "#93A5C2", "#6487C2", BLUE])
    ax.set_ylim(0, 105)
    ax.set_ylabel("success (%)")
    ax.set_title("Training coverage under combined shift", fontweight="bold", color=INK)
    ax.set_xticks(range(5), labels)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value}%",
            ha="center",
            fontweight="bold",
        )
    ax.axhline(100, color=GREEN, lw=1.5, ls="--")
    ax.text(
        4.45, 98, "oracle 100%", ha="right", va="top", fontsize=9, color=GREEN, fontweight="bold"
    )
    ax.text(
        0.02,
        0.87,
        "five-seed combined: 76.0 ± 5.8%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )

    ax = axes[1]
    distances = np.arange(5)
    success = [98, 94, 76, 60, 50]
    ax.plot(distances, success, color=RED, marker="o", lw=2.6, ms=7)
    ax.fill_between(distances, success, 40, color=RED, alpha=0.10)
    ax.set_ylim(40, 102)
    ax.set_xticks(distances, [f"{x} cm" for x in distances])
    ax.set_ylabel("success (%)")
    ax.set_title("Sensitivity to camera displacement", fontweight="bold", color=INK)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for x, value in zip(distances, success, strict=True):
        ax.text(x, value + 2, f"{value}%", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("Closed-loop ball-to-bowl evaluation", fontsize=16, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "policy-experiment-summary.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    pipeline_figure()
    tracking_figure()
    h2o_figure()
    policy_figure()
    print(
        json.dumps(
            {
                "generated": [
                    "pipeline-overview.png",
                    "em1-0406-tracking-ablation.png",
                    "h2o-retargeting-results.png",
                    "policy-experiment-summary.png",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
