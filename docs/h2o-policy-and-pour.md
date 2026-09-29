# H2O policy and Pour experiments

## Question

After the H2O oracle trajectory passes, two experiments test the remaining claims:

1. Can 100 H2O-derived robot demonstrations train a small visual pose policy without
   reducing Place success?
2. Does an orientation-sensitive Pour motion require SE(3), rather than XYZ alone?

All experiments are hardware-free. They use the real H2O object and hand annotations
as the skill source and a robosuite can as the milk-container proxy.

## Place demonstrations and policy

The corrected H2O controller generated every requested demonstration successfully:

| Measurement | Result |
|---|---:|
| Successful demonstrations | **100/100** |
| Control samples | 23,972 |
| Dual-view pose frames | 1,768 |
| Mean trajectory length | 239.72 steps |
| Mean retarget scale | 2.245× |
| Compressed local dataset | 18 MB |

The pose frames are sampled during approach while the full state and action trajectory
is retained for every demonstration. The model uses two 84×84 RGB views and predicts
XYZ plus the continuous two-column rotation representation. It has 2,610,793 parameters
and trained on the local GPU in 25.8 seconds.

| Held-out validation metric | Result |
|---|---:|
| Translation error, mean | **3.54 mm** |
| Translation error, p90 | **7.88 mm** |
| Rotation error, mean | 57.77° |
| Rotation error, p90 | 146.82° |

The high absolute rotation error is expected for the axisymmetric can proxy: yaw is
not visually identifiable and is irrelevant to the top grasp. The paired closed-loop
test therefore scores all pose components but uses estimated translation with the H2O
relative rotations.

| Paired Place evaluation | Oracle | Estimated |
|---|---:|---:|
| Success | **20/20** | **20/20** |
| Mean final target XY error | 2.65 cm | 2.65 cm |
| Estimated translation error | — | 4.62 mm mean, 7.39 mm p90 |
| Estimated absolute rotation error | — | 73.22° mean |

This is a compact visual pose policy plus a demonstrated SE(3) skill and phase
controller. It is not an end-to-end low-level action policy. The pose model is trained
and evaluated on synthetic robosuite imagery; the separate gated H2O RGB-D archive is
not available locally, so no real-pixel pose claim is made.

## Pour representation ablation

The real H2O `pour milk` interval has 139 stable grasp frames, 5.00 mm object-frame
translation RMS, a 29.53° final rotation, and an 80.90° peak tilt. The controlled
ablation evaluates the action prefix through its first 41.14° tilt. Both variants use
the same translated path and retain the grasp. XYZ holds orientation fixed; SE(3)
replays the demonstrated relative rotations.

Success requires gripper position error at most 3 cm, orientation error at most 10°,
and a retained grasp.

| Representation | Success | Mean position error | Mean orientation error |
|---|---:|---:|---:|
| XYZ-only | **0/20** | 1.13 cm | 59.17° |
| Full SE(3) | **20/20** | 0.09 cm | 2.84° |

This experiment tests orientation-sensitive motion transfer. The proxy has no liquid
simulation, so it does not measure poured volume or spillage.

## Reproduce

```bash
MUJOCO_GL=egl PYTHONPATH=src:scripts .venv/bin/python \
  scripts/generate_h2o_place_demos.py \
  --trajectory results/h2o_oracle_place/oracle-place-trajectory.npz \
  --output results/h2o_place_demos_100 --episodes 100

PYTHONPATH=scripts .venv/bin/python scripts/train_h2o_pose_policy.py \
  --data results/h2o_place_demos_100/demonstrations.npz \
  --output results/h2o_pose_policy

MUJOCO_GL=egl PYTHONPATH=src:scripts .venv/bin/python \
  scripts/evaluate_h2o_pose_policy.py \
  --trajectory results/h2o_oracle_place/oracle-place-trajectory.npz \
  --checkpoint results/h2o_pose_policy/visual_pose.pt \
  --episodes 20 \
  --output docs/experiments/h2o-place-oracle-vs-estimated.json

MUJOCO_GL=egl PYTHONPATH=src:scripts .venv/bin/python \
  scripts/evaluate_h2o_pour_ablation.py \
  --trajectory results/h2o_oracle_pour/oracle-pour-trajectory.npz \
  --episodes 20 \
  --output docs/experiments/h2o-pour-representation-ablation.json
```

Machine-readable summaries: [demonstration generation](experiments/h2o-place-demonstrations.json),
[policy training](experiments/h2o-pose-policy-training.json),
[oracle versus estimated execution](experiments/h2o-place-oracle-vs-estimated.json),
and [Pour ablation](experiments/h2o-pour-representation-ablation.json).
