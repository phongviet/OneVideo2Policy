from __future__ import annotations

from pathlib import Path

import numpy as np

from onevideo2policy.video.h2o import (
    estimate_human_grasp,
    load_oracle_trajectory,
    read_action_index,
    relative_object_motion,
)


def _write(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(map(str, values.ravel())) + "\n", encoding="utf-8")


def _right_hand_joints(transform: np.ndarray) -> np.ndarray:
    joints = np.zeros((21, 3))
    joints[0] = [0, -0.04, 0]
    joints[5] = [0.03, 0, 0]
    joints[9] = [0, 0.02, 0]
    joints[13] = [-0.015, 0, 0]
    joints[17] = [-0.03, 0, 0]
    center = joints[[0, 5, 9, 13, 17]].mean(axis=0)
    joints -= center
    return (transform[:3, :3] @ joints.T).T + transform[:3, 3]


def test_load_h2o_oracle_and_extract_grasp(tmp_path: Path) -> None:
    index = tmp_path / "action_train.txt"
    index.write_text(
        "id path action_label start_act end_act start_frame end_frame\n"
        "1 subject1/h1/1 13 0 9 0 9\n",
        encoding="utf-8",
    )
    interval = read_action_index(index)[0]
    root = tmp_path / "dataset"
    camera = root / interval.sequence / "cam4"
    object_to_hand = np.eye(4)
    object_to_hand[:3, 3] = [0, 0, 0.08]
    for frame_id in interval.frame_ids:
        object_to_world = np.eye(4)
        object_to_world[0, 3] = frame_id * 0.01
        hand_to_world = object_to_world @ object_to_hand
        hand_values = np.zeros(128)
        hand_values[64] = 1
        hand_values[65:] = _right_hand_joints(hand_to_world).ravel()
        stem = f"{frame_id:06d}.txt"
        _write(camera / "cam_pose" / stem, np.eye(4))
        _write(camera / "obj_pose_rt" / stem, np.r_[5, object_to_world.ravel()])
        _write(camera / "hand_pose" / stem, hand_values)

    trajectory = load_oracle_trajectory(root, interval)
    grasp = estimate_human_grasp(trajectory)
    motion = relative_object_motion(trajectory)

    assert trajectory.object_name == "milk"
    assert grasp.hand == "right"
    assert grasp.stable_mask.all()
    assert grasp.translation_rms_m < 1e-10
    assert grasp.rotation_p90_deg < 1e-5
    assert np.allclose(grasp.object_to_hand, object_to_hand)
    assert np.allclose(motion[-1, :3, 3], [0.09, 0, 0])


def test_action_index_rejects_malformed_row(tmp_path: Path) -> None:
    index = tmp_path / "bad.txt"
    index.write_text("id path action_label start_act end_act start_frame end_frame\n1 bad\n")
    try:
        read_action_index(index)
    except ValueError as error:
        assert "Malformed H2O action row" in str(error)
    else:
        raise AssertionError("Malformed action row should fail")
