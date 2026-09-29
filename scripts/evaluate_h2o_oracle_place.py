"""Evaluate the annotation-free H2O oracle Place handoff before policy training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from onevideo2policy.video.h2o import (
    ACTION_NAMES,
    estimate_human_grasp,
    load_oracle_trajectory,
    read_action_index,
    relative_object_motion,
)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--action-index", required=True, type=Path)
    parser.add_argument("--sequence", default="subject1/h1/1")
    parser.add_argument("--action-id", default=13, type=int, choices=range(9, 17))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matches = [
        row
        for row in read_action_index(args.action_index)
        if row.sequence == args.sequence and row.action_id == args.action_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {ACTION_NAMES[args.action_id]!r} interval in {args.sequence}, "
            f"found {len(matches)}"
        )
    trajectory = load_oracle_trajectory(args.dataset_root, matches[0])
    grasp = estimate_human_grasp(trajectory)
    motion = relative_object_motion(trajectory)
    net_translation = float(np.linalg.norm(motion[-1, :3, 3]))
    net_rotation = rotation_angle_deg(motion[-1, :3, :3])
    result = {
        "schema_version": 1,
        "experiment": "H2O oracle Place handoff",
        "sequence": matches[0].sequence,
        "action": matches[0].action,
        "frames": len(matches[0].frame_ids),
        "object_id": trajectory.object_id,
        "object_name": trajectory.object_name,
        "selected_hand": grasp.hand,
        "stable_grasp_frames": int(grasp.stable_mask.sum()),
        "grasp_translation_rms_m": round(grasp.translation_rms_m, 5),
        "grasp_rotation_p90_deg": round(grasp.rotation_p90_deg, 3),
        "object_net_translation_m": round(net_translation, 5),
        "object_net_rotation_deg": round(net_rotation, 3),
        "gates": {
            "stable_human_grasp": grasp.translation_rms_m <= 0.02
            and grasp.rotation_p90_deg <= 20,
            "nontrivial_object_motion": net_translation >= 0.03 or net_rotation >= 15,
            "robot_trajectory": False,
        },
        "next_gate": (
            "retarget the saved object-relative SE(3) and object-to-hand transform to "
            "a Panda gripper, then require 10/20 successful perturbed MuJoCo rollouts"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "oracle-place-trajectory.npz",
        frame_ids=matches[0].frame_ids,
        object_to_world=trajectory.object_to_world,
        object_relative_motion=motion,
        object_to_hand=grasp.object_to_hand,
        stable_grasp_mask=grasp.stable_mask,
    )
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
