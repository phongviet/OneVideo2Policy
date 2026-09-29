"""Evaluate every implemented pipeline stage on four additional HOI4D actions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from onevideo2policy.pipeline import (
    evaluate_local_policy,
    fit_ridge_policy,
    generate_local_demonstrations,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/raw/sources/hoi4d/HOI4D_processed"
ANNOTATIONS = ROOT / "data/raw/sources/hoi4d/HOI4D_annotations"
PERCEPTION = ROOT / "results/hoi4d_cross_action_sam2"
WORK = ROOT / "results/hoi4d_whole_pipeline"
OUTPUT = ROOT / "docs/experiments/hoi4d-whole-pipeline-results.json"
FIGURE = ROOT / "docs/assets/hoi4d-whole-pipeline-results.png"

SEQUENCES = [
    {
        "name": "Mug",
        "directory": "mug",
        "id": "ZY20210800004_H4_C2_N40_S10_s05_T1",
        "pose_label": "Watercup",
        "mask_dir": "sam2_masks",
        "skill": "place",
        "skill_supported": True,
        "blocker": "no separately tracked placement target is present in this sequence",
    },
    {
        "name": "Kettle",
        "directory": "kettle",
        "id": "ZY20210800004_H4_C12_N44_S184_s05_T2",
        "pose_label": "kettle",
        "mask_dir": "sam2_masks",
        "skill": "pour",
        "skill_supported": False,
        "blocker": "current skill and controller omit demonstrated object orientation",
    },
    {
        "name": "Trash-can lid",
        "directory": "trash-can_lid",
        "id": "ZY20210800004_H4_C14_N28_S211_s02_T2",
        "pose_label": "Dustbincover",
        "mask_dir": "sam2_masks",
        "skill": "open/close",
        "skill_supported": False,
        "blocker": "part segmentation fails and no articulated joint state is modeled",
    },
    {
        "name": "Storage door",
        "directory": "storage_door",
        "id": "ZY20210800004_H4_C4_N41_S217_s03_T3",
        "pose_label": "Lockersldingdoor",
        "mask_dir": "sam2_part_aware_masks",
        "skill": "close",
        "skill_supported": False,
        "blocker": "no articulated joint state or door manipulation controller",
    },
]


def nested(root: Path, sequence_id: str) -> Path:
    return root.joinpath(*sequence_id.split("_"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_fusion_masks(sequence: dict[str, Any]) -> Path:
    source_dir = PERCEPTION / sequence["directory"] / sequence["mask_dir"]
    destination = WORK / sequence["directory"] / "fusion_masks"
    (destination / "source").mkdir(parents=True, exist_ok=True)
    (destination / "target").mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        cv2.imwrite(str(destination / "source" / path.name), image)
        cv2.imwrite(str(destination / "target" / path.name), np.zeros_like(image))
    return destination


def run_metric_stages(sequence: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    name = sequence["directory"]
    processed = PROCESSED / sequence["id"]
    manifest = PERCEPTION / name / "manifest.json"
    masks = prepare_fusion_masks(sequence)
    trajectory = WORK / name / "rgbd_odometry"
    gaussian = WORK / name / "gaussian_fused_rgbd"
    camera_info = processed / "camera/recon/split_0/info.json"
    depth = processed / "raw_depth"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "onevideo2policy.cli",
            "estimate-rgbd-trajectory",
            "--manifest",
            str(manifest),
            "--depth-dir",
            str(depth),
            "--camera-info",
            str(camera_info),
            "--masks",
            str(masks),
            "--output",
            str(trajectory),
        ],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/fuse_hoi4d_gaussians.py",
            "--manifest",
            str(manifest),
            "--depth-dir",
            str(depth),
            "--masks",
            str(masks),
            "--trajectory",
            str(trajectory / "camera-trajectory.npz"),
            "--output",
            str(gaussian),
            "--frame-ids",
            "1",
            "12",
            "24",
            "36",
            "48",
            "60",
            "71",
            "--reference-frame",
            "1",
            "--validation-frame",
            "30",
            "--stride",
            "6",
            "--voxel-size-m",
            "0.01",
            "--min-static-observations",
            "2",
        ],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=subprocess.DEVNULL,
    )
    return load_json(trajectory / "report.json"), load_json(gaussian / "report.json")


def recover_metric_motion(sequence: dict[str, Any]) -> dict[str, Any]:
    name = sequence["directory"]
    manifest = load_json(PERCEPTION / name / "manifest.json")
    processed = PROCESSED / sequence["id"]
    annotation = nested(ANNOTATIONS, sequence["id"])
    masks = WORK / name / "fusion_masks/source"
    with np.load(WORK / name / "rgbd_odometry/camera-trajectory.npz") as trajectory:
        intrinsics = trajectory["intrinsics"]
        camera_to_world = trajectory["camera_to_world"]
    predicted_world = []
    reference_world = []
    valid_ids = []
    for frame_index, frame in enumerate(manifest["frames"]):
        source_id = int(frame["source_frame_id"])
        depth = cv2.imread(
            str(processed / "raw_depth" / f"{source_id:05d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        mask = cv2.imread(str(masks / f"{frame_index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if depth is None or mask is None:
            continue
        depth_m = cv2.resize(
            depth.astype(np.float32) / 1000,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        ys, xs = np.nonzero((mask > 0) & (depth_m > 0.1) & (depth_m < 4.0))
        if len(xs) < 20:
            continue
        z = depth_m[ys, xs]
        points = np.column_stack(
            (
                (xs - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (ys - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            )
        )
        predicted_camera = np.median(points, axis=0)
        pose = load_json(annotation / "objpose" / f"{source_id}.json")
        item = next(entry for entry in pose["dataList"] if entry["label"] == sequence["pose_label"])
        reference_camera = np.asarray([item["center"][axis] for axis in "xyz"])
        transform = camera_to_world[frame_index]
        predicted_world.append((transform @ np.r_[predicted_camera, 1])[:3])
        reference_world.append((transform @ np.r_[reference_camera, 1])[:3])
        valid_ids.append(frame_index)
    prediction = np.asarray(predicted_world)
    reference = np.asarray(reference_world)
    offset = np.median(prediction[:5] - reference[:5], axis=0)
    errors = np.linalg.norm((prediction - offset) - reference, axis=1)
    relative_errors = np.linalg.norm(
        (prediction - prediction[0]) - (reference - reference[0]), axis=1
    )
    return {
        "valid_frames": len(prediction),
        "first_frame_surface_to_center_offset_m": np.round(offset, 5).tolist(),
        "median_calibrated_center_error_m": round(float(np.median(errors)), 4),
        "p90_calibrated_center_error_m": round(float(np.quantile(errors, 0.9)), 4),
        "median_relative_motion_error_m": round(float(np.median(relative_errors)), 4),
        "p90_relative_motion_error_m": round(float(np.quantile(relative_errors, 0.9)), 4),
        "recovered_world_xyz_m": prediction.tolist(),
        "reference_world_xyz_m": reference.tolist(),
        "frame_ids": valid_ids,
    }


def systems_place_baseline(motion: dict[str, Any]) -> dict[str, Any]:
    prediction = np.asarray(motion["recovered_world_xyz_m"])
    frame_ids = np.asarray(motion["frame_ids"])
    start = int(np.argmin(np.abs(frame_ids - round(3.324 * 5))))
    stop = int(np.argmin(np.abs(frame_ids - round(7.846 * 5))))
    initial_offset = prediction[start] - prediction[stop]
    initial_xy_distance = float(np.linalg.norm(initial_offset[:2]))
    observations, actions = generate_local_demonstrations(
        initial_offset,
        episodes=250,
        steps=48,
        randomization_m=0.06,
        seed=42,
    )
    weights = fit_ridge_policy(observations, actions)
    result = evaluate_local_policy(
        weights,
        initial_offset,
        episodes=100,
        steps=48,
        randomization_m=0.06,
        seed=10042,
        success_xy_m=0.03,
    )
    return {
        "type": "privileged-state point-robot systems baseline",
        "demonstrations": 250,
        "training_samples": len(observations),
        "observed_initial_to_goal_offset_m": np.round(initial_offset, 4).tolist(),
        "initial_xy_distance_m": round(initial_xy_distance, 4),
        "nontrivial_at_3cm_success_tolerance": initial_xy_distance > 0.03,
        **result,
    }


def plot_matrix(rows: list[dict[str, Any]]) -> None:
    stages = [
        "SAM2",
        "CoTracker",
        "RGB-D\nodometry",
        "Gaussian\nscene",
        "Metric\nmotion",
        "Skill\nextraction",
        "Synthetic\nbaseline",
        "Visual robot\npolicy",
    ]
    matrix = np.asarray([[int(row["gates"][stage]["pass"]) for stage in stages] for row in rows])
    figure, axis = plt.subplots(figsize=(10.5, 3.5), constrained_layout=True)
    colors = np.zeros((*matrix.shape, 3), dtype=float)
    colors[matrix == 1] = [0.18, 0.62, 0.55]
    colors[matrix == 0] = [0.82, 0.36, 0.32]
    axis.imshow(colors, aspect="auto")
    axis.set_xticks(np.arange(len(stages)), stages)
    axis.set_yticks(np.arange(len(rows)), [row["name"] for row in rows])
    axis.tick_params(length=0)
    for row_index in range(len(rows)):
        for column_index in range(len(stages)):
            axis.text(
                column_index,
                row_index,
                "PASS" if matrix[row_index, column_index] else "FAIL",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=8,
            )
    axis.set_title(
        "Where cross-action transfer breaks",
        loc="left",
        fontweight="bold",
    )
    axis.set_xticks(np.arange(-0.5, len(stages), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    for spine in axis.spines.values():
        spine.set_visible(False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    sam = {
        row["name"]: row
        for row in load_json(ROOT / "docs/experiments/hoi4d-cross-action-sam2.json")["sequences"]
    }
    tracking = {
        row["name"]: row
        for row in load_json(ROOT / "docs/experiments/hoi4d-cross-action-tracking.json")[
            "sequences"
        ]
    }
    ablation = {
        row["name"]: row
        for row in load_json(ROOT / "docs/experiments/hoi4d-articulated-prompt-ablation.json")[
            "results"
        ]
    }
    rows = []
    for sequence in SEQUENCES:
        metric_report, gaussian_report = run_metric_stages(sequence)
        motion = recover_metric_motion(sequence)
        name = sequence["name"]
        mask_iou = (
            ablation[name]["part_aware_five_point_mean_iou"]
            if sequence["mask_dir"] == "sam2_part_aware_masks"
            else sam[name]["mean_iou"]
        )
        tracking_pass = (
            tracking[name]["survival_gate_pass"] and tracking[name]["containment_gate_pass"]
        )
        odometry_pass = (
            metric_report["median_reprojection_error_px"] <= 5
            and metric_report["median_depth_residual_m"] <= 0.02
        )
        gaussian_pass = (
            gaussian_report["held_out_render"]["evaluated_coverage"] >= 0.5
            and gaussian_report["held_out_render"]["psnr_db"] >= 20
        )
        motion_pass = motion["p90_relative_motion_error_m"] <= 0.03
        baseline = systems_place_baseline(motion) if sequence["skill_supported"] else None
        gates = {
            "SAM2": {"pass": mask_iou >= 0.7, "value": mask_iou, "threshold": 0.7},
            "CoTracker": {
                "pass": tracking_pass,
                "survival": tracking[name]["track_survival"],
                "ground_truth_containment": tracking[name]["ground_truth_track_containment"],
                "thresholds": [0.8, 0.8],
            },
            "RGB-D\nodometry": {
                "pass": odometry_pass,
                "median_reprojection_error_px": metric_report["median_reprojection_error_px"],
                "median_depth_residual_m": metric_report["median_depth_residual_m"],
            },
            "Gaussian\nscene": {
                "pass": gaussian_pass,
                **gaussian_report["held_out_render"],
            },
            "Metric\nmotion": {
                "pass": motion_pass,
                "p90_relative_motion_error_m": motion["p90_relative_motion_error_m"],
            },
            "Skill\nextraction": {
                "pass": False,
                "skill": sequence["skill"],
                "primitive_implemented": sequence["skill_supported"],
                "blocker": sequence["blocker"],
            },
            "Synthetic\nbaseline": {
                "pass": (
                    baseline is not None
                    and baseline["success_rate"] >= 0.8
                    and baseline["nontrivial_at_3cm_success_tolerance"]
                ),
                "result": baseline,
            },
            "Visual robot\npolicy": {
                "pass": False,
                "reason": (
                    "no matching robosuite task, task-specific demonstrations, or "
                    "trained visual checkpoint"
                ),
            },
        }
        rows.append(
            {
                "name": name,
                "sequence_id": sequence["id"],
                "action": sequence["skill"],
                "gates": gates,
                "rgbd_odometry": {
                    key: value for key, value in metric_report.items() if key != "steps"
                },
                "gaussian_scene": gaussian_report,
                "metric_motion": {
                    key: value
                    for key, value in motion.items()
                    if key
                    not in {
                        "recovered_world_xyz_m",
                        "reference_world_xyz_m",
                        "frame_ids",
                    }
                },
                "strict_end_to_end_pass": all(stage["pass"] for stage in gates.values()),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "HOI4D whole-pipeline cross-action evaluation",
        "protocol": {
            "gate_order": [
                "SAM2",
                "CoTracker",
                "RGB-D odometry",
                "Gaussian scene",
                "Metric motion",
                "Skill extraction",
                "Synthetic baseline",
                "Visual robot policy",
            ],
            "strict_rule": (
                "a sequence passes end to end only if every stage passes with action "
                "semantics preserved"
            ),
            "policy_scope": (
                "hardware-free; the synthetic baseline is privileged-state and is not "
                "counted as the final visual robot policy"
            ),
        },
        "sequences": rows,
        "summary": {
            "strict_end_to_end_passes": sum(row["strict_end_to_end_pass"] for row in rows),
            "sequences": len(rows),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_matrix(rows)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
