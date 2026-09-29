"""Retarget an H2O oracle Place trajectory to 20 randomized Panda rollouts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils as transform

from onevideo2policy.video.h2o import retarget_object_path


def osc_action(
    observation: dict[str, np.ndarray],
    position: np.ndarray,
    orientation: np.ndarray,
    gripper: float,
) -> np.ndarray:
    position_error = np.clip((position - observation["robot0_eef_pos"]) / 0.05, -0.35, 0.35)
    target_quaternion = transform.mat2quat(orientation)
    rotation_error = np.clip(
        transform.get_orientation_error(observation["robot0_eef_quat"], target_quaternion),
        -0.1,
        0.1,
    )
    return np.r_[position_error, rotation_error, gripper]


def place_can_near_target(env: Any, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Create a reachable Place perturbation with the H2O motion's metric scale."""
    target = env.target_bin_placements[env.object_id]
    angle = np.random.uniform(-0.22, 0.22)
    # Keep the can clear of the bin wall (its collision radius is about 5 cm).
    distance = np.random.uniform(0.19, 0.22)
    position = np.array(
        [
            target[0] + distance * np.sin(angle),
            target[1] - distance * np.cos(angle),
            observation["Can_pos"][2],
        ]
    )
    joint = env.objects[env.object_id].joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = position
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()
    return env._get_observations(force_update=True)


def run_episode(env: Any, motion: np.ndarray, stable_mask: np.ndarray, max_steps: int) -> dict:
    observation = env.reset()
    observation = place_can_near_target(env, observation)
    target = env.target_bin_placements[env.object_id].copy()
    initial_can = observation["Can_pos"].copy()
    grasp_xy = observation["Can_pos"][:2].copy()
    phase = 0
    hold = 0
    waypoint = 0
    positions = orientations = None
    path_scale = None
    maximum_path_error = 0.0

    def result(success: bool, steps: int) -> dict:
        can_position = observation["Can_pos"].copy()
        grasped = env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.objects[env.object_id].contact_geoms,
        )
        return {
            "success": success,
            "steps": steps,
            "final_phase": phase,
            "completed_waypoints": waypoint,
            "path_scale": path_scale,
            "maximum_commanded_path_error_m": maximum_path_error,
            "initial_can_position": initial_can.tolist(),
            "final_can_position": can_position.tolist(),
            "target_position": target.tolist(),
            "final_target_xy_error_m": float(np.linalg.norm(can_position[:2] - target[:2])),
            "final_in_bin": not env.not_in_bin(can_position, env.object_id),
            "final_grasped": bool(grasped),
        }

    for step in range(max_steps):
        eef = observation["robot0_eef_pos"]
        current_orientation = transform.quat2mat(observation["robot0_eef_quat"])
        if phase == 0:
            goal = np.r_[grasp_xy, observation["Can_pos"][2] + 0.12]
            goal_orientation, gripper = current_orientation, -1.0
        elif phase == 1:
            goal = np.r_[grasp_xy, 0.88]
            goal_orientation, gripper = current_orientation, -1.0
        elif phase == 2:
            goal, goal_orientation, gripper = eef, current_orientation, 1.0
        elif phase == 3:
            goal = np.r_[grasp_xy, 1.02]
            goal_orientation, gripper = current_orientation, 1.0
        elif phase == 4:
            if positions is None:
                object_to_eef = eef - observation["Can_pos"]
                end_object = np.r_[target[:2], 0.86]
                positions, orientations, path_scale = retarget_object_path(
                    motion,
                    stable_mask,
                    observation["Can_pos"],
                    end_object,
                    current_orientation,
                )
                positions += object_to_eef
            goal = positions[waypoint]
            goal_orientation, gripper = orientations[waypoint], 1.0
            maximum_path_error = max(maximum_path_error, float(np.linalg.norm(goal - eef)))
        elif phase == 5:
            goal, goal_orientation, gripper = eef, current_orientation, -1.0
        else:
            goal = np.r_[target[:2], 1.05]
            goal_orientation, gripper = current_orientation, -1.0
        action = osc_action(observation, goal, goal_orientation, gripper)
        observation, _, done, _ = env.step(action)
        position_error = float(np.linalg.norm(goal - observation["robot0_eef_pos"]))
        if phase in (0, 1, 3, 6) and position_error < 0.012:
            phase += 1
            hold = 0
        elif phase == 2:
            hold += 1
            if hold >= 25:
                phase = 3
                hold = 0
        elif phase == 4:
            hold += 1
            if position_error < 0.014 or hold >= 20:
                waypoint += 1
                hold = 0
                if waypoint >= len(positions):
                    phase = 5
        elif phase == 5:
            hold += 1
            if hold >= 25:
                phase = 6
        success = bool(env._check_success())
        if success or done:
            return result(success, step + 1)
    return result(False, max_steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    oracle = json.loads(args.oracle_report.read_text(encoding="utf-8"))
    with np.load(args.trajectory) as trajectory:
        motion = trajectory["object_relative_motion"]
        stable_mask = trajectory["stable_grasp_mask"]
    np.random.seed(args.seed)
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
    started = time.perf_counter()
    episodes = []
    try:
        for episode in range(args.episodes):
            result = run_episode(env, motion, stable_mask, args.max_steps)
            episodes.append(result)
            print(f"episode {episode + 1}/{args.episodes}: {result}", flush=True)
    finally:
        env.close()
    successes = sum(result["success"] for result in episodes)
    payload = {
        "schema_version": 1,
        "experiment": "H2O oracle Place to Panda retargeting",
        "source_sequence": oracle["sequence"],
        "source_action": oracle["action"],
        "source_object": oracle["object_name"],
        "source_grasp": {
            "hand": oracle["selected_hand"],
            "stable_frames": oracle["stable_grasp_frames"],
            "translation_rms_m": oracle["grasp_translation_rms_m"],
            "rotation_p90_deg": oracle["grasp_rotation_p90_deg"],
        },
        "simulator": "robosuite PickPlace can proxy",
        "robot": "Panda",
        "controller": "OSC_POSE",
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "gate": {
            "required_successes": 10,
            "required_episodes": 20,
            "pass": args.episodes == 20 and successes >= 10,
        },
        "trajectory": {
            "stable_h2o_frames": int(stable_mask.sum()),
            "retargeted_waypoints": 12,
            "maximum_control_steps_per_waypoint": 20,
            "full_se3_commands": True,
            "endpoint_alignment": "uniform scale plus minimum 3D rotation",
            "simulator_start_distance_m": "uniform [0.19, 0.22] from target",
        },
        "episodes_detail": episodes,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "uses robosuite's can proxy because the H2O object mesh was not downloaded",
            "endpoint alignment adapts one human trajectory to randomized simulator layouts",
            "this evaluates oracle retargeting, not a learned visual policy",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
