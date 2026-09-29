"""Generate validated Panda demonstrations from an H2O Place trajectory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils as transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_h2o_oracle_retarget import (  # noqa: E402
    osc_action,
    place_can_near_target,
)

from onevideo2policy.video.h2o import retarget_object_path


def pose_6d(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Return XYZ plus the first two object rotation columns."""
    rotation = transform.quat2mat(observation["Can_quat"])
    return np.r_[observation["Can_pos"], rotation[:, :2].T.ravel()]


def collect_episode(
    env,
    motion: np.ndarray,
    stable_mask: np.ndarray,
    max_steps: int,
) -> tuple[dict[str, np.ndarray], bool, dict]:
    observation = place_can_near_target(env, env.reset())
    target = env.target_bin_placements[env.object_id].copy()
    initial_can = observation["Can_pos"].copy()
    grasp_xy = initial_can[:2].copy()
    keys = (
        "proprio",
        "objects",
        "actions",
        "eef_positions",
        "can_positions",
        "can_poses_6d",
        "target_positions",
        "phases",
    )
    records: dict[str, list[np.ndarray | int]] = {key: [] for key in keys}
    pose_images, pose_images_front, pose_targets = [], [], []
    phase = hold = waypoint = 0
    positions = orientations = None
    path_scale = None
    success = False
    for _step in range(max_steps):
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
        elif phase == 5:
            goal, goal_orientation, gripper = eef, current_orientation, -1.0
        else:
            goal = np.r_[target[:2], 1.05]
            goal_orientation, gripper = current_orientation, -1.0
        action = osc_action(observation, goal, goal_orientation, gripper)
        if phase in (0, 1) and _step % 5 == 0:
            pose_images.append(
                env.sim.render(camera_name="agentview", height=84, width=84)[::-1].copy()
            )
            pose_images_front.append(
                env.sim.render(camera_name="frontview", height=84, width=84)[::-1].copy()
            )
            pose_targets.append(pose_6d(observation))
        records["proprio"].append(observation["robot0_proprio-state"].copy())
        records["objects"].append(observation["object-state"].copy())
        records["actions"].append(action.copy())
        records["eef_positions"].append(eef.copy())
        records["can_positions"].append(observation["Can_pos"].copy())
        records["can_poses_6d"].append(pose_6d(observation))
        records["target_positions"].append(target.copy())
        records["phases"].append(phase)
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
            break
    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["pose_images"] = np.asarray(pose_images, dtype=np.uint8)
    arrays["pose_images_front"] = np.asarray(pose_images_front, dtype=np.uint8)
    arrays["pose_targets_6d"] = np.asarray(pose_targets, dtype=np.float32)
    metadata = {
        "success": success,
        "steps": _step + 1,
        "initial_can_position": initial_can.tolist(),
        "final_can_position": observation["Can_pos"].tolist(),
        "path_scale": path_scale,
    }
    return arrays, success, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=130)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    with np.load(args.trajectory) as trajectory:
        motion = trajectory["object_relative_motion"]
        stable_mask = trajectory["stable_grasp_mask"]
    np.random.seed(args.seed)
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        reward_shaping=True,
        single_object_mode=2,
        object_type="can",
        control_freq=20,
        horizon=args.max_steps,
        ignore_done=False,
        hard_reset=False,
    )
    episodes, details = [], []
    attempts = 0
    started = time.perf_counter()
    try:
        while len(episodes) < args.episodes and attempts < args.max_attempts:
            attempts += 1
            episode, success, metadata = collect_episode(
                env, motion, stable_mask, args.max_steps
            )
            if success:
                episodes.append(episode)
                details.append(metadata)
                print(
                    f"successful episode {len(episodes)}/{args.episodes} "
                    f"after {attempts} attempts ({metadata['steps']} steps)",
                    flush=True,
                )
    finally:
        env.close()
    if len(episodes) != args.episodes:
        raise RuntimeError(f"Generated {len(episodes)}/{args.episodes} successful episodes")
    keys = episodes[0]
    concatenated = {key: np.concatenate([episode[key] for episode in episodes]) for key in keys}
    episode_ends = np.cumsum([len(episode["actions"]) for episode in episodes])
    pose_episode_ends = np.cumsum([len(episode["pose_images"]) for episode in episodes])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "demonstrations.npz",
        **concatenated,
        episode_ends=episode_ends,
        pose_episode_ends=pose_episode_ends,
    )
    report = {
        "schema_version": 1,
        "status": "pass",
        "source": "H2O subject1/h1/1 place milk frames 0-48",
        "environment": "robosuite PickPlace can proxy",
        "controller": "H2O-derived 12-waypoint SE(3) trajectory with OSC_POSE",
        "requested_episodes": args.episodes,
        "successful_episodes": len(episodes),
        "attempts": attempts,
        "samples": int(episode_ends[-1]),
        "pose_training_samples": int(pose_episode_ends[-1]),
        "mean_episode_steps": float(np.mean([row["steps"] for row in details])),
        "mean_path_scale": float(np.mean([row["path_scale"] for row in details])),
        "cameras": ["agentview 84x84 RGB", "frontview 84x84 RGB"],
        "pose_target": "XYZ plus continuous 6D rotation representation",
        "elapsed_seconds": time.perf_counter() - started,
        "episodes_detail": details,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
