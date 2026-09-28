"""Evaluate one-point SAM2 propagation on four additional HOI4D actions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from onevideo2policy.video.hoi4d import import_hoi4d_rgb_video
from onevideo2policy.video.sam2_adapter import Sam2PointPrompt, Sam2VideoAdapter

ROOT = Path(__file__).resolve().parents[1]
RGB = ROOT / "data/raw/sources/hoi4d/HOI4D_release"
ANNOTATIONS = ROOT / "data/raw/sources/hoi4d/HOI4D_annotations"
WORK = ROOT / "results/hoi4d_cross_action_sam2"
OUTPUT = ROOT / "docs/experiments/hoi4d-cross-action-sam2.json"

SEQUENCES = [
    ("Mug", "ZY20210800004_H4_C2_N40_S10_s05_T1", 1),
    ("Kettle", "ZY20210800004_H4_C12_N44_S184_s05_T2", 1),
    ("Trash-can lid", "ZY20210800004_H4_C14_N28_S211_s02_T2", 3),
    ("Storage door", "ZY20210800004_H4_C4_N41_S217_s03_T3", 1),
]


def sequence_path(root: Path, sequence_id: str) -> Path:
    return root.joinpath(*sequence_id.split("_"))


def prompt_point(mask: np.ndarray) -> tuple[int, int]:
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    _, maximum, _, location = cv2.minMaxLoc(distance)
    if maximum <= 0:
        raise ValueError("First-frame ground-truth mask is empty")
    return location


def iou(prediction: np.ndarray, reference: np.ndarray) -> float:
    union = np.logical_or(prediction, reference).sum()
    return float(np.logical_and(prediction, reference).sum() / union) if union else 1.0


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    predictor = Sam2VideoAdapter.from_hugging_face(device="cuda")
    results = []
    for name, sequence_id, label in SEQUENCES:
        sequence_work = WORK / name.lower().replace(" ", "_")
        manifest_path = sequence_work / "manifest.json"
        if not manifest_path.exists():
            import_hoi4d_rgb_video(
                sequence_path(RGB, sequence_id),
                sequence_path(ANNOTATIONS, sequence_id),
                sequence_work,
                source_labels=[label],
                target_labels=[2],
                sample_fps=5.0,
                max_width=640,
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        references = []
        for path in sorted((sequence_work / "ground_truth/source").glob("*.png")):
            references.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0)
        references_array = np.stack(references)
        x, y = prompt_point(references_array[0])
        prompt = Sam2PointPrompt(
            object_id=1,
            frame_idx=0,
            points_xy=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        predictions = predictor.segment_video(
            sequence_work / "frames", [prompt], expected_frame_count=len(references_array)
        )[1]
        frame_ious = np.array(
            [
                iou(prediction, reference)
                for prediction, reference in zip(predictions, references_array, strict=True)
            ]
        )
        results.append(
            {
                "name": name,
                "sequence_id": sequence_id,
                "motion_label": label,
                "frames": len(frame_ious),
                "sample_fps": manifest["sample_fps"],
                "first_frame_prompt_xy": [x, y],
                "mean_iou": round(float(frame_ious.mean()), 4),
                "median_iou": round(float(np.median(frame_ious)), 4),
                "p10_iou": round(float(np.quantile(frame_ious, 0.1)), 4),
                "frames_iou_at_least_0_5": round(float((frame_ious >= 0.5).mean()), 4),
            }
        )
        print(name, results[-1])

    payload = {
        "schema_version": 1,
        "experiment": "SAM2.1 Hiera Small cross-action propagation on HOI4D",
        "protocol": {
            "model": "facebook/sam2.1-hiera-small",
            "input": "5 Hz, maximum width 640 px",
            "prompt": (
                "one positive point at the maximum distance-transform location of the "
                "official first-frame mask"
            ),
            "evaluation": (
                "IoU against official masks on every sampled frame, including the prompted frame"
            ),
        },
        "sequences": results,
        "macro_mean_iou": round(float(np.mean([row["mean_iou"] for row in results])), 4),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
