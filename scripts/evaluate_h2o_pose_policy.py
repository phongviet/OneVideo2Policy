"""Compare oracle and learned-pose execution of an H2O-derived Place skill."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import robosuite as suite
import torch
from robosuite.utils import transform_utils as transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_h2o_oracle_retarget import osc_action  # noqa: E402
from robosuite_policy_models import VisualPosePolicy  # noqa: E402
from train_h2o_pose_policy import rotation_errors_deg, rotation_from_6d  # noqa: E402

from onevideo2policy.video.h2o import retarget_object_path


def set_can_pose(env: Any, position: np.ndarray, yaw: float) -> dict[str, np.ndarray]:
    joint = env.objects[env.object_id].joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = position
    qpos[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()
    return env._get_observations(force_update=True)


def image_tensor(observation: dict[str, np.ndarray], device: str) -> torch.Tensor:
    image = np.concatenate(
        (observation["agentview_image"], observation["frontview_image"]), axis=-1
    )
    tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    return tensor.permute(2, 0, 1).unsqueeze(0) / 255


def predict_pose(
    model: VisualPosePolicy,
    observation: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> np.ndarray:
    with torch.inference_mode():
        normalized = model(image_tensor(observation, device))[0].cpu().numpy()
    return normalized * std + mean


def run_episode(
    env: Any,
    motion: np.ndarray,
    stable_mask: np.ndarray,
    initial_position: np.ndarray,
    yaw: float,
    mode: str,
    model: VisualPosePolicy,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    max_steps: int,
) -> dict:
    observation = set_can_pose(env, initial_position, yaw)
    observation["agentview_image"] = env.sim.render(
        camera_name="agentview", height=84, width=84
    )[::-1].copy()
    observation["frontview_image"] = env.sim.render(
        camera_name="frontview", height=84, width=84
    )[::-1].copy()
    target = env.target_bin_placements[env.object_id].copy()
    estimated_pose = predict_pose(model, observation, mean, std, device)
    actual_rotation = transform.quat2mat(observation["Can_quat"])
    estimated_rotation = rotation_from_6d(estimated_pose[None, 3:])[0]
    translation_error = float(np.linalg.norm(estimated_pose[:3] - observation["Can_pos"]))
    rotation_error = float(
        rotation_errors_deg(estimated_rotation[None], actual_rotation[None])[0]
    )
    grasp_xy = (
        observation["Can_pos"][:2].copy()
        if mode == "oracle"
        else estimated_pose[:2].copy()
    )
    phase = hold = waypoint = 0
    positions = orientations = None
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
        elif phase == 4:
            if positions is None:
                positions, orientations, _ = retarget_object_path(
                    motion,
                    stable_mask,
                    eef,
                    np.r_[target[:2], 0.885],
                    current_orientation,
                )
            goal = positions[waypoint]
            goal_orientation, gripper = orientations[waypoint], 1.0
        elif phase == 5:
            goal, goal_orientation, gripper = eef, current_orientation, -1.0
        else:
            goal = np.r_[target[:2], 1.05]
            goal_orientation, gripper = current_orientation, -1.0
        observation, _, done, _ = env.step(
            osc_action(observation, goal, goal_orientation, gripper)
        )
        error = float(np.linalg.norm(goal - observation["robot0_eef_pos"]))
        if phase in (0, 1, 3, 6) and error < 0.012:
            phase += 1
            hold = 0
        elif phase == 2:
            hold += 1
            if hold >= 25:
                phase = 3
                hold = 0
        elif phase == 4:
            hold += 1
            if error < 0.014 or hold >= 20:
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
    return {
        "success": success,
        "steps": _step + 1,
        "terminal_phase": phase,
        "initial_position": initial_position.tolist(),
        "yaw_rad": yaw,
        "estimated_position": estimated_pose[:3].tolist(),
        "translation_error_m": translation_error,
        "rotation_error_deg": rotation_error,
        "final_can_position": observation["Can_pos"].tolist(),
        "final_target_xy_error_m": float(
            np.linalg.norm(observation["Can_pos"][:2] - target[:2])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args()
    with np.load(args.trajectory) as trajectory:
        motion = trajectory["object_relative_motion"]
        stable_mask = trajectory["stable_grasp_mask"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = VisualPosePolicy().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = np.asarray(checkpoint["pose_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["pose_std"], dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    angles = rng.uniform(-0.22, 0.22, args.episodes)
    distances = rng.uniform(0.19, 0.22, args.episodes)
    yaws = rng.uniform(-np.pi, np.pi, args.episodes)
    target = np.array([0.1975, 0.4025, 0.8])
    positions = np.c_[
        target[0] + distances * np.sin(angles),
        target[1] - distances * np.cos(angles),
        np.full(args.episodes, 0.86),
    ]
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
    results: dict[str, list[dict]] = {"oracle": [], "estimated": []}
    started = time.perf_counter()
    try:
        for mode in results:
            for episode in range(args.episodes):
                env.reset()
                result = run_episode(
                    env,
                    motion,
                    stable_mask,
                    positions[episode],
                    float(yaws[episode]),
                    mode,
                    model,
                    mean,
                    std,
                    device,
                    args.max_steps,
                )
                results[mode].append(result)
                print(
                    f"{mode} {episode + 1}/{args.episodes}: "
                    f"success={result['success']} pose_error={result['translation_error_m']:.4f}m",
                    flush=True,
                )
    finally:
        env.close()
    summary = {}
    for mode, rows in results.items():
        summary[mode] = {
            "successes": sum(row["success"] for row in rows),
            "episodes": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_final_target_xy_error_m": float(
                np.mean([row["final_target_xy_error_m"] for row in rows])
            ),
        }
    pose_rows = results["estimated"]
    payload = {
        "schema_version": 1,
        "experiment": "H2O Place oracle versus learned visual 6D pose",
        "source": "subject1/h1/1 place milk",
        "policy": "dual-view 2.62M-parameter pose estimator plus H2O SE(3) skill",
        "pose_metrics": {
            "translation_error_mean_m": float(
                np.mean([row["translation_error_m"] for row in pose_rows])
            ),
            "translation_error_p90_m": float(
                np.quantile([row["translation_error_m"] for row in pose_rows], 0.9)
            ),
            "rotation_error_mean_deg": float(
                np.mean([row["rotation_error_deg"] for row in pose_rows])
            ),
            "rotation_error_p90_deg": float(
                np.quantile([row["rotation_error_deg"] for row in pose_rows], 0.9)
            ),
        },
        "execution": summary,
        "gate": {
            "minimum_estimated_successes": 16,
            "episodes": 20,
            "pass": args.episodes == 20 and summary["estimated"]["successes"] >= 16,
        },
        "episodes_detail": results,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "the milk container is represented by robosuite's axisymmetric can proxy",
            "control uses estimated translation and demonstrated relative rotations",
            "absolute estimated yaw is scored but is not needed for the top grasp",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
