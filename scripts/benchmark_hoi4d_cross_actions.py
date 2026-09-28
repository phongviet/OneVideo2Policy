"""Audit HOI4D action diversity and selected sequence geometry annotations.

The benchmark is deliberately model independent: it tests whether the inputs needed
by OneVideo2Policy exist beyond the original bowl sequence.  It measures official
action, motion-mask, and 3D object-pose annotations and produces a compact paper-style
figure plus a machine-readable record.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from onevideo2policy.video.hoi4d import hoi4d_color_map

CATEGORIES = {
    "C1": "Toy car",
    "C2": "Mug",
    "C4": "Storage furniture",
    "C5": "Bottle",
    "C7": "Bowl",
    "C12": "Kettle",
    "C14": "Trash can",
}

# The bowl is the existing reference. The other rows were selected before running
# the benchmark to cover rigid transfer, rotation, and articulated interaction.
SEQUENCES = [
    {
        "id": "ZY20210800001_H1_C7_N14_S280_s04_T5",
        "short": "Bowl",
        "motion": "pick/place",
        "key_events": ["Pickup", "carry", "putdown"],
        "object_labels": [1, 3],
    },
    {
        "id": "ZY20210800004_H4_C2_N40_S10_s05_T1",
        "short": "Mug",
        "motion": "pick/place ×2",
        "key_events": ["Pickup", "carry", "putdown"],
        "object_labels": [1, 3],
    },
    {
        "id": "ZY20210800004_H4_C12_N44_S184_s05_T2",
        "short": "Kettle",
        "motion": "pour",
        "key_events": ["dump"],
        "object_labels": [1, 3],
    },
    {
        "id": "ZY20210800004_H4_C14_N28_S211_s02_T2",
        "short": "Trash can",
        "motion": "open/close",
        "key_events": ["open", "close"],
        "object_labels": [1, 3, 4],
    },
    {
        "id": "ZY20210800004_H4_C4_N41_S217_s03_T3",
        "short": "Storage",
        "motion": "place + close",
        "key_events": ["putdown", "close"],
        "object_labels": [1, 3, 4, 5],
    },
]


def sequence_path(root: Path, sequence_id: str) -> Path:
    return root.joinpath(*sequence_id.split("_"))


def action_summary(annotation_root: Path) -> dict[str, Any]:
    files = sorted(annotation_root.glob("**/action/color.json"))
    event_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    event_seconds: Counter[str] = Counter()
    complete_pose = 0
    complete_masks = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = payload["events"]
        event_counts.update(event["event"] for event in events)
        for event in events:
            event_seconds[event["event"]] += event["endTime"] - event["startTime"]
        category = next((part for part in path.parts if part in CATEGORIES), "unknown")
        category_counts[category] += 1
        sequence = path.parents[1]
        complete_pose += len(list((sequence / "objpose").glob("*.json"))) == 300
        motion_root = sequence / "2Dseg"
        motion_dir = (
            motion_root / "mask" if (motion_root / "mask").is_dir() else motion_root / "shift_mask"
        )
        complete_masks += len(list(motion_dir.glob("*.png"))) == 300
    return {
        "sequence_count": len(files),
        "category_counts": {
            CATEGORIES.get(key, key): value for key, value in sorted(category_counts.items())
        },
        "event_counts": dict(event_counts.most_common()),
        "event_seconds": {key: round(value, 3) for key, value in event_seconds.most_common()},
        "sequences_with_300_pose_frames": complete_pose,
        "sequences_with_300_mask_frames": complete_masks,
    }


def read_pose_series(pose_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    frames = []
    labels: set[str] = set()
    for path in sorted(pose_dir.glob("*.json"), key=lambda item: int(item.stem)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames.append(payload)
        labels.update(item["label"] for item in payload.get("dataList", []))
    return frames, sorted(labels)


def pose_metrics(frames: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    effective = 0
    for frame in frames:
        effective += bool(frame.get("isEffective"))
        for item in frame.get("dataList", []):
            by_label[item["label"]].append(item)

    objects: dict[str, Any] = {}
    center_series: dict[str, np.ndarray] = {}
    for label, items in by_label.items():
        centers = np.array([[x["center"][axis] for axis in "xyz"] for x in items], dtype=float)
        center_series[label] = centers
        dimensions = np.array(
            [[x["dimensions"][axis] for axis in ("length", "width", "height")] for x in items],
            dtype=float,
        )
        steps = (
            np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.array([])
        )
        objects[label] = {
            "annotated_frames": len(items),
            "dimensions_m": np.round(np.median(dimensions, axis=0), 4).tolist(),
            "dimension_cv_percent": round(
                float(
                    np.mean(
                        np.std(dimensions, axis=0) / np.maximum(np.mean(dimensions, axis=0), 1e-9)
                    )
                )
                * 100,
                3,
            ),
            "camera_frame_path_m": round(float(steps.sum()), 3),
            "camera_frame_net_m": round(float(np.linalg.norm(centers[-1] - centers[0])), 3),
        }

    relative_ranges = []
    for index, first in enumerate(labels):
        for second in labels[index + 1 :]:
            count = min(len(center_series[first]), len(center_series[second]))
            if not count:
                continue
            distances = np.linalg.norm(
                center_series[first][:count] - center_series[second][:count], axis=1
            )
            relative_ranges.append(
                {
                    "pair": [first, second],
                    "distance_range_m": round(float(np.ptp(distances)), 3),
                    "median_distance_m": round(float(np.median(distances)), 3),
                }
            )
    relative_ranges.sort(key=lambda item: item["distance_range_m"], reverse=True)
    return {
        "pose_frames": len(frames),
        "effective_pose_fraction": round(effective / max(len(frames), 1), 4),
        "objects": objects,
        "largest_relative_center_change": relative_ranges[0] if relative_ranges else None,
    }


def mask_metrics(mask_dir: Path, object_labels: list[int], stride: int = 5) -> dict[str, Any]:
    paths = sorted(mask_dir.glob("*.png"), key=lambda item: int(item.stem))
    palette = hoi4d_color_map()
    fractions = []
    visible = Counter()
    for path in paths[::stride]:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        union = np.zeros(image.shape[:2], dtype=bool)
        for label in object_labels:
            current = np.all(image == palette[label], axis=-1)
            union |= current
            visible[str(label)] += bool(current.any())
        fractions.append(float(union.mean()))
    sampled = len(fractions)
    return {
        "mask_frames": len(paths),
        "sample_stride": stride,
        "sampled_frames": sampled,
        "object_visible_fraction": round(sum(value > 0 for value in fractions) / sampled, 4),
        "mean_object_image_fraction": round(float(np.mean(fractions)), 4),
        "p05_object_image_fraction": round(float(np.quantile(fractions, 0.05)), 4),
        "per_label_visibility": {
            label: round(count / sampled, 4) for label, count in visible.items()
        },
    }


def key_event_metrics(action: dict[str, Any], key_events: list[str]) -> dict[str, Any]:
    selected = [event for event in action["events"] if event["event"] in key_events]
    return {
        "event_count": len(action["events"]),
        "event_vocabulary": sorted({event["event"] for event in action["events"]}),
        "key_event_seconds": round(
            sum(event["endTime"] - event["startTime"] for event in selected), 3
        ),
        "key_event_instances": [
            {
                "event": event["event"],
                "start_s": round(event["startTime"], 3),
                "end_s": round(event["endTime"], 3),
            }
            for event in selected
        ],
    }


def thumbnail(video: Path, time_s: float, mask: Path, object_labels: list[int]) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read {video} at {time_s:.2f} s")
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_id = min(round(time_s * 15), 299)
    mask_image = cv2.imread(str(mask / f"{frame_id:05d}.png"))
    if mask_image is not None:
        rgb_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        palette = hoi4d_color_map()
        binary = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        for label in object_labels:
            binary |= np.all(rgb_mask == palette[label], axis=-1).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 225, 190), 6)
    return image


def make_figure(rows: list[dict[str, Any]], coverage: dict[str, Any], output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    figure = plt.figure(figsize=(12.2, 7.1), constrained_layout=True)
    grid = figure.add_gridspec(3, 5, height_ratios=[1.0, 0.82, 0.9])
    for index, row in enumerate(rows):
        axis = figure.add_subplot(grid[0, index])
        axis.imshow(row.pop("_thumbnail"))
        axis.set_title(f"{row['short']}\n{row['motion']}", fontweight="bold", fontsize=10)
        axis.axis("off")
        axis.text(
            0.02,
            0.03,
            f"{row['metrics']['events']['key_event_seconds']:.1f} s key action",
            transform=axis.transAxes,
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.7, "pad": 2, "edgecolor": "none"},
        )

    event_axis = figure.add_subplot(grid[1, :3])
    common = ["Reachout", "Grasp", "Pickup", "carry", "dump", "open", "close", "putdown"]
    counts = [coverage["event_counts"].get(event, 0) for event in common]
    colors = ["#627D98"] * 4 + ["#E08E45", "#2A9D8F", "#2A9D8F", "#627D98"]
    event_axis.bar(common, counts, color=colors, width=0.72)
    event_axis.set_ylabel("Annotated instances")
    event_axis.set_title(
        "A  Action coverage across all 61 local sequences", loc="left", fontweight="bold"
    )
    event_axis.tick_params(axis="x", rotation=28)
    event_axis.spines[["top", "right"]].set_visible(False)

    visibility_axis = figure.add_subplot(grid[1, 3:])
    names = [row["short"] for row in rows]
    visibility = [row["metrics"]["masks"]["mean_object_image_fraction"] * 100 for row in rows]
    visibility_axis.barh(
        names, visibility, color=["#6B7A8F", "#4C78A8", "#F2A541", "#2A9D8F", "#2A9D8F"]
    )
    visibility_axis.invert_yaxis()
    visibility_axis.set_xlabel("Object mask area (% image)")
    visibility_axis.set_title("B  Visual support", loc="left", fontweight="bold")
    visibility_axis.spines[["top", "right"]].set_visible(False)

    table_axis = figure.add_subplot(grid[2, :])
    table_axis.axis("off")
    column_labels = [
        "Sequence",
        "Motion",
        "Pose frames",
        "Mask frames",
        "Object visible",
        "3D parts",
        "Representation",
    ]
    cells = []
    for row in rows:
        metrics = row["metrics"]
        articulated = len(metrics["poses"]["objects"]) > 1 and row["short"] in {
            "Trash can",
            "Storage",
        }
        cells.append(
            [
                row["short"],
                row["motion"],
                str(metrics["poses"]["pose_frames"]),
                str(metrics["masks"]["mask_frames"]),
                f"{metrics['masks']['object_visible_fraction'] * 100:.0f}%",
                str(len(metrics["poses"]["objects"])),
                "part/joint state" if articulated else "rigid SE(3)",
            ]
        )
    table = table_axis.table(
        cellText=cells, colLabels=column_labels, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D9E2EC")
        if row == 0:
            cell.set_facecolor("#243B53")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F0F4F8")
    table_axis.set_title(
        "C  Annotation and task-representation audit",
        loc="left",
        fontweight="bold",
        pad=4,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    coverage = action_summary(args.annotations)
    rows = []
    for spec in SEQUENCES:
        annotation = sequence_path(args.annotations, spec["id"])
        rgb = sequence_path(args.rgb, spec["id"])
        action = json.loads((annotation / "action" / "color.json").read_text(encoding="utf-8"))
        events = key_event_metrics(action, spec["key_events"])
        poses, labels = read_pose_series(annotation / "objpose")
        motion_root = annotation / "2Dseg"
        mask_dir = (
            motion_root / "mask" if (motion_root / "mask").is_dir() else motion_root / "shift_mask"
        )
        midpoint = events["key_event_instances"][0]
        midpoint_s = (midpoint["start_s"] + midpoint["end_s"]) / 2
        row = {
            **spec,
            "metrics": {
                "events": events,
                "masks": mask_metrics(mask_dir, spec["object_labels"]),
                "poses": pose_metrics(poses, labels),
            },
            "_thumbnail": thumbnail(
                rgb / "align_rgb" / "image.mp4", midpoint_s, mask_dir, spec["object_labels"]
            ),
        }
        rows.append(row)

    make_figure(rows, coverage, args.figure)
    result = {
        "schema_version": 1,
        "experiment": "HOI4D cross-sequence and cross-action feasibility audit",
        "protocol": {
            "selection": (
                "one existing reference plus four preselected sequences spanning rigid "
                "transfer, pouring, and articulated interaction"
            ),
            "mask_sampling": "every fifth official 15 Hz motion-segmentation frame",
            "pose_measurement": "all 300 official 3D object-pose frames",
            "caveat": (
                "camera-frame path length includes egocentric camera motion and is "
                "descriptive, not a world-frame trajectory error"
            ),
        },
        "dataset_coverage": coverage,
        "selected_sequences": rows,
        "conclusion": {
            "rigid_supported": ["Bowl", "Mug", "Kettle"],
            "requires_part_state": ["Trash can", "Storage"],
            "interpretation": (
                "The current rigid object-relative skill can consume pick/place and pour "
                "sequences. Open/close actions have complete observations but require a "
                "part or joint state before policy synthesis."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, default=Path("data/raw/sources/hoi4d/HOI4D_release"))
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/raw/sources/hoi4d/HOI4D_annotations")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/hoi4d-cross-action-results.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("docs/assets/hoi4d-cross-action-results.png")
    )
    result = run(parser.parse_args())
    print(
        json.dumps(
            {
                "output_sequences": len(result["selected_sequences"]),
                "coverage": result["dataset_coverage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
