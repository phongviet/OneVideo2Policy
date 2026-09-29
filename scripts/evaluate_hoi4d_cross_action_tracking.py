"""Run CoTracker3 after the selected SAM2 protocol on four HOI4D actions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from onevideo2policy.video.cotracker_adapter import CoTracker3Adapter
from onevideo2policy.video.perception import (
    load_manifest_rgb,
    run_tracking_on_masks,
    save_perception_artifacts,
)
from onevideo2policy.video.sam2_adapter import Sam2PointPrompt

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "results/hoi4d_cross_action_sam2"
OUTPUT = ROOT / "docs/experiments/hoi4d-cross-action-tracking.json"
CHECKPOINT = ROOT / "weights/cotracker3/scaled_offline.pth"
SEQUENCES = [
    ("Mug", "mug", "sam2_masks"),
    ("Kettle", "kettle", "sam2_masks"),
    ("Trash-can lid", "trash-can_lid", "sam2_masks"),
    ("Storage door", "storage_door", "sam2_part_aware_masks"),
]


def load_masks(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No masks in {directory}")
    return np.stack([cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0 for path in paths])


def ground_truth_containment(xy: np.ndarray, visible: np.ndarray, reference: np.ndarray) -> float:
    height, width = reference.shape[1:]
    rounded = np.rint(xy).astype(int)
    in_bounds = (
        (rounded[..., 0] >= 0)
        & (rounded[..., 0] < width)
        & (rounded[..., 1] >= 0)
        & (rounded[..., 1] < height)
    )
    eligible = visible & in_bounds
    if not eligible.any():
        return 0.0
    frame_ids, point_ids = np.nonzero(eligible)
    inside = reference[
        frame_ids,
        rounded[frame_ids, point_ids, 1],
        rounded[frame_ids, point_ids, 0],
    ]
    return float(inside.mean())


def main() -> None:
    tracker = CoTracker3Adapter.from_checkpoint(CHECKPOINT, device="cuda")
    rows = []
    for name, directory, mask_name in SEQUENCES:
        sequence = WORK / directory
        frames = load_manifest_rgb(sequence / "manifest.json")
        predictions = load_masks(sequence / mask_name)
        references = load_masks(sequence / "ground_truth/source")
        prompt = Sam2PointPrompt(
            object_id=1,
            frame_idx=0,
            points_xy=np.asarray([[0, 0]], dtype=np.float32),
            labels=np.asarray([1], dtype=np.int32),
        )
        result = run_tracking_on_masks(
            frames,
            {"object": predictions},
            {"object": prompt},
            tracker,
            point_count=32,
            seed=42,
            border=4,
            reseed_interval=16,
        )["object"]
        save_perception_artifacts({"object": result}, sequence / "cotracker")
        survival = float(result.visible.mean())
        containment = ground_truth_containment(result.tracks_xy, result.visible, references)
        rows.append(
            {
                "name": name,
                "frames": len(frames),
                "points_per_window": 32,
                "reseed_interval_frames": 16,
                "track_survival": round(survival, 4),
                "ground_truth_track_containment": round(containment, 4),
                "survival_gate_pass": survival >= 0.8,
                "containment_gate_pass": containment >= 0.8,
            }
        )
        print(rows[-1])
    payload = {
        "schema_version": 1,
        "experiment": "CoTracker3 after selected SAM2 cross-action masks",
        "protocol": {
            "checkpoint": "scaled_offline.pth",
            "points": 32,
            "reseed_interval_frames": 16,
            "survival_threshold": 0.8,
            "ground_truth_containment_threshold": 0.8,
        },
        "sequences": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
