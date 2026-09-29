"""Compare XYZ-only and full-SE(3) H2O Pour retargeting on a Panda."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils as transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_h2o_oracle_retarget import osc_action  # noqa: E402

from onevideo2policy.video.h2o import retarget_object_path


def rotation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    relative = target.T @ actual
    cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def set_can_pose(env: Any, position: np.ndarray, yaw: float) -> dict[str, np.ndarray]:
    joint = env.objects[env.object_id].joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = position
    qpos[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()
    return env._get_observations(force_update=True)


def run_episode(
    env: Any,
    motion: np.ndarray,
    stable_mask: np.ndarray,
    initial_position: np.ndarray,
    yaw: float,
    displacement: np.ndarray,
    representation: str,
    max_steps: int,
) -> dict:
    observation = set_can_pose(env, initial_position, yaw)
    grasp_xy = initial_position[:2]
    phase = hold = waypoint = 0
    positions = orientations = None
    initial_object_rotation = target_object_rotation = None
    target_object_position = None
    target_eef_position = target_eef_orientation = None
    for _step in range(max_steps):
        eef = observation["robot0_eef_pos"]
        current_orientation = transform.quat2mat(observation["robot0_eef_quat"])
        if phase == 0:
            goal = np.r_[grasp_xy, 0.98]
            goal_orientation, gripper = current_orientation, -1.0
        elif phase == 1:
            goal = np.r_[grasp_xy, 0.88]
            goal_orientation, gripper = current_orientation, -1.0
        elif phase == 2:
            goal, goal_orientation, gripper = eef, current_orientation, 1.0
        elif phase == 3:
            goal = np.r_[grasp_xy, 1.02]
            goal_orientation, gripper = current_orientation, 1.0
        else:
            if positions is None:
                initial_object_position = observation["Can_pos"].copy()
                initial_object_rotation = transform.quat2mat(observation["Can_quat"])
                positions, orientations, _ = retarget_object_path(
                    motion,
                    stable_mask,
                    eef,
                    eef + displacement,
                    current_orientation,
                    waypoint_count=20,
                )
                target_object_position = initial_object_position + displacement
                eef_rotation_delta = orientations[-1] @ current_orientation.T
                target_object_rotation = eef_rotation_delta @ initial_object_rotation
                target_eef_position = positions[-1].copy()
                target_eef_orientation = orientations[-1].copy()
                if representation == "xyz":
                    orientations[:] = current_orientation
            goal = positions[waypoint]
            goal_orientation, gripper = orientations[waypoint], 1.0
        observation, _, done, _ = env.step(
            osc_action(observation, goal, goal_orientation, gripper)
        )
        error = float(np.linalg.norm(goal - observation["robot0_eef_pos"]))
        orientation_tracking_error = rotation_error_deg(
            transform.quat2mat(observation["robot0_eef_quat"]), goal_orientation
        )
        if phase in (0, 1, 3) and error < 0.012:
            phase += 1
            hold = 0
        elif phase == 2:
            hold += 1
            if hold >= 25:
                phase = 3
                hold = 0
        elif phase == 4:
            hold += 1
            if (error < 0.014 and orientation_tracking_error < 3) or hold >= 30:
                waypoint += 1
                hold = 0
                if waypoint >= len(positions):
                    break
        if done:
            break
    final_position = observation["Can_pos"].copy()
    final_rotation = transform.quat2mat(observation["Can_quat"])
    position_error = (
        float(np.linalg.norm(final_position - target_object_position))
        if target_object_position is not None
        else float("inf")
    )
    orientation_error = (
        rotation_error_deg(final_rotation, target_object_rotation)
        if target_object_rotation is not None
        else float("inf")
    )
    achieved_rotation = (
        rotation_error_deg(final_rotation, initial_object_rotation)
        if initial_object_rotation is not None
        else 0.0
    )
    final_eef_position = observation["robot0_eef_pos"]
    final_eef_rotation = transform.quat2mat(observation["robot0_eef_quat"])
    eef_position_error = float(np.linalg.norm(final_eef_position - target_eef_position))
    eef_orientation_error = rotation_error_deg(
        final_eef_rotation, target_eef_orientation
    )
    grasped = bool(
        env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.objects[env.object_id].contact_geoms,
        )
    )
    success = eef_position_error <= 0.03 and eef_orientation_error <= 10 and grasped
    return {
        "success": success,
        "steps": _step + 1,
        "completed_waypoints": waypoint,
        "position_error_m": position_error,
        "orientation_error_deg": orientation_error,
        "achieved_rotation_deg": achieved_rotation,
        "eef_position_error_m": eef_position_error,
        "eef_orientation_error_deg": eef_orientation_error,
        "grasped": grasped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1618)
    args = parser.parse_args()
    with np.load(args.trajectory) as trajectory:
        original_motion = trajectory["object_relative_motion"]
        original_stable_mask = trajectory["stable_grasp_mask"]
    source_angles = np.array(
        [rotation_error_deg(pose[:3, :3], np.eye(3)) for pose in original_motion]
    )
    peak_id = int(np.argmax(source_angles))
    evaluation_end = int(np.flatnonzero(source_angles >= 40)[0])
    motion = original_motion[: evaluation_end + 1]
    stable_mask = original_stable_mask[: evaluation_end + 1]
    rng = np.random.default_rng(args.seed)
    initial_positions = np.c_[
        rng.uniform(-0.02, 0.05, args.episodes),
        rng.uniform(-0.12, -0.08, args.episodes),
        np.full(args.episodes, 0.86),
    ]
    angles = rng.uniform(-0.1, 0.1, args.episodes)
    distances = rng.uniform(0.10, 0.12, args.episodes)
    displacements = np.c_[
        distances * np.cos(angles),
        distances * np.sin(angles),
        np.zeros(args.episodes),
    ]
    yaws = rng.uniform(-np.pi, np.pi, args.episodes)
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        single_object_mode=2,
        object_type="can",
        control_freq=20,
        horizon=args.max_steps,
        ignore_done=False,
        hard_reset=False,
    )
    results: dict[str, list[dict]] = {"xyz": [], "se3": []}
    started = time.perf_counter()
    try:
        for representation in results:
            for episode in range(args.episodes):
                env.reset()
                result = run_episode(
                    env,
                    motion,
                    stable_mask,
                    initial_positions[episode],
                    float(yaws[episode]),
                    displacements[episode],
                    representation,
                    args.max_steps,
                )
                results[representation].append(result)
                print(
                    f"{representation} {episode + 1}/{args.episodes}: "
                    f"success={result['success']} rotation_error="
                    f"{result['eef_orientation_error_deg']:.1f}deg",
                    flush=True,
                )
    finally:
        env.close()
    summary = {}
    for representation, rows in results.items():
        summary[representation] = {
            "successes": sum(row["success"] for row in rows),
            "episodes": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_position_error_m": float(np.mean([row["position_error_m"] for row in rows])),
            "mean_orientation_error_deg": float(
                np.mean([row["orientation_error_deg"] for row in rows])
            ),
            "mean_achieved_rotation_deg": float(
                np.mean([row["achieved_rotation_deg"] for row in rows])
            ),
            "mean_eef_position_error_m": float(
                np.mean([row["eef_position_error_m"] for row in rows])
            ),
            "mean_eef_orientation_error_deg": float(
                np.mean([row["eef_orientation_error_deg"] for row in rows])
            ),
        }
    payload = {
        "schema_version": 1,
        "experiment": "H2O Pour XYZ versus SE(3) representation",
        "source": "subject1/h1/1 pour milk frames 304-442",
        "source_motion": {
            "frames": len(original_motion),
            "stable_grasp_frames": int(original_stable_mask.sum()),
            "net_translation_m": float(np.linalg.norm(original_motion[-1, :3, 3])),
            "net_rotation_deg": rotation_error_deg(
                original_motion[-1, :3, :3], np.eye(3)
            ),
            "peak_tilt_frame_offset": peak_id,
            "peak_tilt_deg": float(source_angles[peak_id]),
            "evaluated_tilt_deg": float(source_angles[evaluation_end]),
            "evaluated_segment": (
                f"action start through first 40deg tilt, frames 0-{evaluation_end}"
            ),
        },
        "success_definition": (
            "gripper position error <=3cm, rotation error <=10deg, grasp retained"
        ),
        "results": summary,
        "gate": {
            "minimum_se3_successes": 16,
            "maximum_xyz_successes": 4,
            "episodes": 20,
            "pass": args.episodes == 20
            and summary["se3"]["successes"] >= 16
            and summary["xyz"]["successes"] <= 4,
        },
        "episodes_detail": results,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "the simulator uses a grasped can and pose goal rather than liquid dynamics",
            "success scores the retargeted gripper pose because the proxy can can slip in-hand",
            "the result tests orientation-sensitive motion transfer, not pouring volume",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
