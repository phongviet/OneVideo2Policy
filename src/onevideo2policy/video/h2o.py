"""Readers and geometry helpers for the official H2O pose release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

OBJECT_NAMES = {
    0: "background",
    1: "book",
    2: "espresso",
    3: "lotion",
    4: "spray",
    5: "milk",
    6: "cocoa",
    7: "chips",
    8: "cappuccino",
}
ACTION_NAMES = {
    0: "background",
    1: "grab book",
    2: "grab espresso",
    3: "grab lotion",
    4: "grab spray",
    5: "grab milk",
    6: "grab cocoa",
    7: "grab chips",
    8: "grab cappuccino",
    9: "place book",
    10: "place espresso",
    11: "place lotion",
    12: "place spray",
    13: "place milk",
    14: "place cocoa",
    15: "place chips",
    16: "place cappuccino",
    17: "open lotion",
    18: "open milk",
    19: "open chips",
    20: "close lotion",
    21: "close milk",
    22: "close chips",
    23: "pour milk",
    24: "take out espresso",
    25: "take out cocoa",
    26: "take out chips",
    27: "take out cappuccino",
    28: "put in espresso",
    29: "put in cocoa",
    30: "put in cappuccino",
    31: "apply lotion",
    32: "apply spray",
    33: "read book",
    34: "read espresso",
    35: "spray spray",
    36: "squeeze lotion",
}


@dataclass(frozen=True)
class H2OActionInterval:
    sample_id: int
    sequence: str
    action_id: int
    start_frame: int
    end_frame: int
    sequence_start: int
    sequence_end: int

    @property
    def action(self) -> str:
        return ACTION_NAMES[self.action_id]

    @property
    def frame_ids(self) -> NDArray[np.int64]:
        return np.arange(self.start_frame, self.end_frame + 1, dtype=np.int64)


@dataclass(frozen=True)
class H2OOracleTrajectory:
    interval: H2OActionInterval
    object_id: int
    object_to_world: FloatArray
    left_hand_to_world: FloatArray
    right_hand_to_world: FloatArray
    left_valid: NDArray[np.bool_]
    right_valid: NDArray[np.bool_]

    @property
    def object_name(self) -> str:
        return OBJECT_NAMES[self.object_id]


@dataclass(frozen=True)
class H2OGraspEstimate:
    hand: str
    object_to_hand: FloatArray
    stable_mask: NDArray[np.bool_]
    translation_rms_m: float
    rotation_p90_deg: float


def read_action_index(path: str | Path) -> list[H2OActionInterval]:
    """Read an official H2O action split file."""
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("id "):
            continue
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(f"Malformed H2O action row {line_number + 1}: {line!r}")
        rows.append(
            H2OActionInterval(
                sample_id=int(fields[0]),
                sequence=fields[1],
                action_id=int(fields[2]),
                start_frame=int(fields[3]),
                end_frame=int(fields[4]),
                sequence_start=int(fields[5]),
                sequence_end=int(fields[6]),
            )
        )
    return rows


def _read_numbers(path: Path) -> FloatArray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite H2O annotation: {path}")
    return values


def _read_transform(path: Path) -> FloatArray:
    values = _read_numbers(path)
    if values.size != 16:
        raise ValueError(f"Expected 16 transform values in {path}, found {values.size}")
    transform = values.reshape(4, 4)
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"Invalid homogeneous transform in {path}")
    return transform


def _read_object_pose(path: Path) -> tuple[int, FloatArray]:
    values = _read_numbers(path)
    if values.size != 17:
        raise ValueError(f"Expected object id plus 16 transform values in {path}")
    object_id = int(values[0])
    if object_id not in OBJECT_NAMES or object_id == 0:
        raise ValueError(f"Frame has no supported foreground object: {path}")
    return object_id, values[1:].reshape(4, 4)


def _palm_frame(joints: FloatArray, *, hand: str) -> FloatArray:
    """Build a palm frame from wrist and MCP joints in the official 21-joint order."""
    wrist = joints[0]
    index_mcp = joints[5]
    middle_mcp = joints[9]
    pinky_mcp = joints[17]
    x_axis = index_mcp - pinky_mcp
    x_axis /= np.linalg.norm(x_axis)
    y_hint = middle_mcp - wrist
    z_axis = np.cross(x_axis, y_hint)
    z_axis /= np.linalg.norm(z_axis)
    if hand == "left":
        z_axis *= -1
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = np.mean(joints[[0, 5, 9, 13, 17]], axis=0)
    return transform


def _read_hand_poses(path: Path) -> tuple[bool, FloatArray, bool, FloatArray]:
    values = _read_numbers(path)
    if values.size != 128:
        raise ValueError(f"Expected two 64-value hand records in {path}")
    result: list[bool | FloatArray] = []
    for hand_index, hand in enumerate(("left", "right")):
        record = values[hand_index * 64 : (hand_index + 1) * 64]
        valid = bool(round(record[0]))
        joints = record[1:].reshape(21, 3)
        result.extend((valid, _palm_frame(joints, hand=hand) if valid else np.eye(4)))
    return result[0], result[1], result[2], result[3]  # type: ignore[return-value]


def load_oracle_trajectory(
    dataset_root: str | Path,
    interval: H2OActionInterval,
    *,
    camera: str = "cam4",
) -> H2OOracleTrajectory:
    """Load world-frame object and palm SE(3) trajectories for one H2O action."""
    sequence = Path(dataset_root) / interval.sequence / camera
    object_poses = []
    left_poses = []
    right_poses = []
    left_valid = []
    right_valid = []
    object_ids = []
    for frame_id in interval.frame_ids:
        stem = f"{frame_id:06d}.txt"
        camera_to_world = _read_transform(sequence / "cam_pose" / stem)
        object_id, object_to_camera = _read_object_pose(sequence / "obj_pose_rt" / stem)
        left_ok, left_to_camera, right_ok, right_to_camera = _read_hand_poses(
            sequence / "hand_pose" / stem
        )
        object_ids.append(object_id)
        object_poses.append(camera_to_world @ object_to_camera)
        left_poses.append(camera_to_world @ left_to_camera)
        right_poses.append(camera_to_world @ right_to_camera)
        left_valid.append(left_ok)
        right_valid.append(right_ok)
    if len(set(object_ids)) != 1:
        raise ValueError(f"Object identity changed inside {interval.action}: {set(object_ids)}")
    return H2OOracleTrajectory(
        interval=interval,
        object_id=object_ids[0],
        object_to_world=np.stack(object_poses),
        left_hand_to_world=np.stack(left_poses),
        right_hand_to_world=np.stack(right_poses),
        left_valid=np.asarray(left_valid),
        right_valid=np.asarray(right_valid),
    )


def _rotation_error_deg(rotations: FloatArray, center: FloatArray) -> FloatArray:
    relative = np.einsum("ij,njk->nik", center.T, rotations)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def _average_transform(transforms: FloatArray) -> FloatArray:
    mean_rotation = transforms[:, :3, :3].mean(axis=0)
    u, _, vt = np.linalg.svd(mean_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.median(transforms[:, :3, 3], axis=0)
    return result


def estimate_human_grasp(
    trajectory: H2OOracleTrajectory,
    *,
    translation_threshold_m: float = 0.02,
    rotation_threshold_deg: float = 20.0,
    minimum_stable_frames: int = 8,
) -> H2OGraspEstimate:
    """Choose the hand most rigidly coupled to the object and average its grasp frame."""
    candidates = []
    world_to_object = np.linalg.inv(trajectory.object_to_world)
    for hand, poses, valid in (
        ("left", trajectory.left_hand_to_world, trajectory.left_valid),
        ("right", trajectory.right_hand_to_world, trajectory.right_valid),
    ):
        relative = world_to_object @ poses
        valid_relative = relative[valid]
        if len(valid_relative) < minimum_stable_frames:
            continue
        center = _average_transform(valid_relative)
        translation_error = np.linalg.norm(valid_relative[:, :3, 3] - center[:3, 3], axis=1)
        rotation_error = _rotation_error_deg(valid_relative[:, :3, :3], center[:3, :3])
        stable_valid = (translation_error <= translation_threshold_m) & (
            rotation_error <= rotation_threshold_deg
        )
        if stable_valid.sum() < minimum_stable_frames:
            continue
        stable = np.zeros(len(relative), dtype=bool)
        stable[np.flatnonzero(valid)[stable_valid]] = True
        grasp = _average_transform(relative[stable])
        translation_residuals = np.linalg.norm(
            relative[stable, :3, 3] - grasp[:3, 3], axis=1
        )
        translation_rms = float(
            np.sqrt(np.mean(np.square(translation_residuals)))
        )
        rotation_p90 = float(
            np.quantile(_rotation_error_deg(relative[stable, :3, :3], grasp[:3, :3]), 0.9)
        )
        candidates.append(
            (
                translation_rms + rotation_p90 / 1000,
                hand,
                grasp,
                stable,
                translation_rms,
                rotation_p90,
            )
        )
    if not candidates:
        raise ValueError("No hand has a stable object-relative grasp interval")
    _, hand, grasp, stable, translation_rms, rotation_p90 = min(candidates, key=lambda row: row[0])
    return H2OGraspEstimate(hand, grasp, stable, translation_rms, rotation_p90)


def relative_object_motion(trajectory: H2OOracleTrajectory) -> FloatArray:
    """Express the object trajectory relative to its first action frame."""
    return np.linalg.inv(trajectory.object_to_world[0]) @ trajectory.object_to_world


def align_vectors(source: FloatArray, target: FloatArray) -> FloatArray:
    """Return the minimum rotation taking one nonzero 3D vector onto another."""
    source = np.array(source, dtype=float, copy=True)
    target = np.array(target, dtype=float, copy=True)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(source @ target, -1, 1))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-9:
        if cosine > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source @ axis) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis -= (axis @ source) * source
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)


def retarget_object_path(
    motion: FloatArray,
    stable_mask: NDArray[np.bool_],
    start_position: FloatArray,
    end_position: FloatArray,
    start_orientation: FloatArray,
    waypoint_count: int = 12,
) -> tuple[FloatArray, FloatArray, float]:
    """Align and uniformly scale a stable H2O path to a robot start and goal."""
    stable_ids = np.flatnonzero(stable_mask)
    if len(stable_ids) < 2:
        raise ValueError("H2O retargeting needs at least two stable grasp frames")
    selected = motion[stable_ids[0] : stable_ids[-1] + 1]
    selected = selected[
        np.unique(np.linspace(0, len(selected) - 1, waypoint_count).round().astype(int))
    ]
    human_xyz = selected[:, :3, 3] - selected[0, :3, 3]
    human_displacement = human_xyz[-1]
    robot_displacement = end_position - start_position
    alignment = align_vectors(human_displacement, robot_displacement)
    scale = float(np.linalg.norm(robot_displacement) / np.linalg.norm(human_displacement))
    positions = start_position + (scale * (alignment @ human_xyz.T)).T
    first_rotation = selected[0, :3, :3]
    orientations = []
    for pose in selected:
        relative_rotation = first_rotation.T @ pose[:3, :3]
        simulator_delta = alignment @ relative_rotation @ alignment.T
        orientations.append(simulator_delta @ start_orientation)
    return positions, np.stack(orientations), scale
