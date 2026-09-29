"""Train a compact dual-view 6D object-pose policy on H2O-derived demonstrations."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robosuite_policy_models import VisualPosePolicy  # noqa: E402


def image_tensor(
    images: np.ndarray,
    front: np.ndarray,
    ids: np.ndarray,
    device: str,
    *,
    augment: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.concatenate((images[ids], front[ids]), axis=-1))
    tensor = tensor.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255
    if augment:
        scale = torch.empty((len(ids), 1, 1, 1), device=device).uniform_(0.75, 1.25)
        shift = torch.empty((len(ids), 1, 1, 1), device=device).uniform_(-0.12, 0.12)
        tensor = (tensor * scale + shift + torch.randn_like(tensor) * 0.015).clamp(0, 1)
    return tensor


def rotation_from_6d(values: np.ndarray) -> np.ndarray:
    first = values[..., :3]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = values[..., 3:]
    second = second - (first * second).sum(-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def rotation_errors_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(target, -1, -2) @ prediction
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(args.data) as data:
        images = data["pose_images"]
        front = data["pose_images_front"]
        targets = data["pose_targets_6d"].astype(np.float32)
        ends = data["pose_episode_ends"].astype(int)
    starts = np.r_[0, ends[:-1]]
    episode_ids = np.arange(len(ends))
    np.random.shuffle(episode_ids)
    split = min(max(1, round(0.8 * len(ends))), len(ends) - 1)
    train_episodes, validation_episodes = episode_ids[:split], episode_ids[split:]
    train_ids = np.concatenate([np.arange(starts[i], ends[i]) for i in train_episodes])
    validation_ids = np.concatenate(
        [np.arange(starts[i], ends[i]) for i in validation_episodes]
    )
    mean = targets[train_ids].mean(0)
    std = np.maximum(targets[train_ids].std(0), 1e-4)
    model = VisualPosePolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_mae, best_state, history = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        shuffled = np.random.permutation(train_ids)
        for ids in np.array_split(shuffled, int(np.ceil(len(shuffled) / args.batch_size))):
            prediction = model(image_tensor(images, front, ids, device, augment=True))
            target = torch.from_numpy((targets[ids] - mean) / std).to(device)
            loss = torch.nn.functional.huber_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        predictions = []
        with torch.inference_mode():
            for ids in np.array_split(
                validation_ids, int(np.ceil(len(validation_ids) / args.batch_size))
            ):
                predictions.append(
                    model(image_tensor(images, front, ids, device, augment=False)).cpu().numpy()
                )
        normalized = np.concatenate(predictions)
        validation_mae = float(np.abs(normalized - (targets[validation_ids] - mean) / std).mean())
        history.append(
            {
                "epoch": epoch + 1,
                "train_huber": float(np.mean(losses)),
                "validation_normalized_mae": validation_mae,
            }
        )
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for ids in np.array_split(
            validation_ids, int(np.ceil(len(validation_ids) / args.batch_size))
        ):
            predictions.append(
                model(image_tensor(images, front, ids, device, augment=False)).cpu().numpy()
            )
    predicted = np.concatenate(predictions) * std + mean
    target = targets[validation_ids]
    translation_errors = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=1)
    rotation_errors = rotation_errors_deg(
        rotation_from_6d(predicted[:, 3:]), rotation_from_6d(target[:, 3:])
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "h2o_visual_pose",
            "model_state": best_state,
            "pose_mean": mean,
            "pose_std": std,
            "dual_camera": True,
        },
        args.output / "visual_pose.pt",
    )
    report = {
        "schema_version": 1,
        "status": "trained",
        "device": device,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "episodes": len(ends),
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "train_samples": len(train_ids),
        "validation_samples": len(validation_ids),
        "best_epoch": min(history, key=lambda row: row["validation_normalized_mae"]),
        "translation_error_mean_m": float(translation_errors.mean()),
        "translation_error_p90_m": float(np.quantile(translation_errors, 0.9)),
        "rotation_error_mean_deg": float(rotation_errors.mean()),
        "rotation_error_p90_deg": float(np.quantile(rotation_errors, 0.9)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
