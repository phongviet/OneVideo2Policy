"""Test whether part-aware negative points resolve articulated SAM2 ambiguity."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from onevideo2policy.video.sam2_adapter import Sam2PointPrompt, Sam2VideoAdapter

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "results/hoi4d_cross_action_sam2"
OUTPUT = ROOT / "docs/experiments/hoi4d-articulated-prompt-ablation.json"
SEQUENCES = [("Trash-can lid", "trash-can_lid"), ("Storage door", "storage_door")]


def load_masks(directory: Path) -> np.ndarray:
    return np.stack(
        [
            cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
            for path in sorted(directory.glob("*.png"))
        ]
    )


def part_prompt(mask: np.ndarray) -> Sam2PointPrompt:
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    _, _, _, positive = cv2.minMaxLoc(distance)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    pad = max(6, round(max(x1 - x0, y1 - y0) * 0.08))
    height, width = mask.shape
    negatives = [
        (max(0, x0 - pad), center_y),
        (min(width - 1, x1 + pad), center_y),
        (center_x, max(0, y0 - pad)),
        (center_x, min(height - 1, y1 + pad)),
    ]
    return Sam2PointPrompt(
        object_id=1,
        points_xy=np.asarray([positive, *negatives], dtype=np.float32),
        labels=np.asarray([1, 0, 0, 0, 0], dtype=np.int32),
    )


def iou(prediction: np.ndarray, reference: np.ndarray) -> float:
    union = np.logical_or(prediction, reference).sum()
    return float(np.logical_and(prediction, reference).sum() / union) if union else 1.0


def main() -> None:
    baseline = json.loads(
        (ROOT / "docs/experiments/hoi4d-cross-action-sam2.json").read_text(encoding="utf-8")
    )
    baseline_by_name = {row["name"]: row for row in baseline["sequences"]}
    model = Sam2VideoAdapter.from_hugging_face(device="cuda")
    results = []
    for name, directory in SEQUENCES:
        sequence = WORK / directory
        references = load_masks(sequence / "ground_truth/source")
        prompt = part_prompt(references[0])
        predictions = model.segment_video(
            sequence / "frames", [prompt], expected_frame_count=len(references)
        )[1]
        values = np.asarray(
            [
                iou(prediction, reference)
                for prediction, reference in zip(predictions, references, strict=True)
            ]
        )
        row = {
            "name": name,
            "baseline_one_point_mean_iou": baseline_by_name[name]["mean_iou"],
            "part_aware_five_point_mean_iou": round(float(values.mean()), 4),
            "part_aware_median_iou": round(float(np.median(values)), 4),
            "part_aware_p10_iou": round(float(np.quantile(values, 0.1)), 4),
            "part_aware_frames_iou_at_least_0_5": round(float((values >= 0.5).mean()), 4),
            "prompt_points_xy": prompt.points_xy.astype(int).tolist(),
            "prompt_labels": prompt.labels.tolist(),
        }
        print(row)
        results.append(row)
    payload = {
        "schema_version": 1,
        "experiment": "Articulated-part prompt ablation for SAM2.1 Hiera Small",
        "protocol": (
            "one interior positive point plus four negatives just outside the official "
            "first-frame part bounding box; later frames remain unseen"
        ),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
