# H2O oracle Place experiment

## Goal

The next gate uses one public H2O demonstration and requires no new recording or manual
video annotation:

> Can an H2O Place interval automatically produce a stable object-frame human grasp
> and a robot-ready SE(3) trajectory?

This replaces continued HOI4D action expansion. The completed HOI4D experiment remains
as a boundary study showing where the current rigid, center-only representation fails.

## Selected sequence

The official training action index contains 149 Place intervals and 15 `pour milk`
intervals. The first oracle target is:

| Field | Value |
|---|---|
| Sequence | `subject1/h1/1` |
| Action | `place milk` |
| Frames | 0–48, 49 frames |
| Follow-up action in the same take | `pour milk`, frames 304–442 |

The Place interval is used first. Pour is reserved for the XYZ-versus-SE(3) ablation
after the Place retargeting gate passes.

## Implemented oracle path

The loader consumes the official `cam4` annotation directories:

```text
cam_pose/*.txt       camera → world
obj_pose_rt/*.txt    object → camera
hand_pose/*.txt      left/right 21-joint hand poses in camera coordinates
```

It performs the following operations without per-sequence clicks:

1. Compose camera and object annotations into world-frame object SE(3).
2. Construct left and right palm frames from the wrist and MCP joints.
3. Express each palm in the moving object frame.
4. Select the hand with the most stable object-relative transform.
5. Reject frames over 2 cm translation or 20° rotation from the robust mean.
6. Save the averaged object-to-hand grasp and full object-relative SE(3) trajectory.

The generic human-hand-to-Panda-gripper mapping remains a single global calibration;
it is not a new annotation for each demonstration.

## Measured result

Both oracle gates pass on the real H2O annotations:

| Measurement | Result |
|---|---:|
| Frames | 49 |
| Automatically selected hand | right |
| Stable grasp frames | 34 |
| Object-frame grasp translation RMS | 5.14 mm |
| Object-frame grasp rotation p90 | 6.73° |
| Object motion | 10.57 cm, 7.45° |
| Perturbed Panda Place rollouts | **20/20 (100%)** |
| Final target XY error | 2.71 cm mean, 1.44–3.72 cm range |
| Retarget scale | 2.28× mean, 2.15–2.36× range |

The controller downsamples the 34-frame stable interval to 12 SE(3) waypoints, applies
one minimum 3D rotation and uniform scale to match each simulator start and goal, and
uses the H2O relative rotations for gripper orientation. The simulator randomizes the
can 19–22 cm from the target. This clearance keeps the can outside the bin collision
wall; the initial 13.5–15.5 cm setup overlapped the wall and invalidated the grasp point.

This result validates oracle grasp extraction and trajectory retargeting. It does not
measure visual pose estimation or a learned policy. The simulator also uses a can as a
milk-container proxy because the H2O object mesh is not included in the pose archive.

The compact extraction record is
[`h2o-oracle-place.json`](experiments/h2o-oracle-place.json), and all 20 rollout records
are in
[`h2o-oracle-place-retarget.json`](experiments/h2o-oracle-place-retarget.json).

## Data access

H2O requires accepting its academic, non-commercial terms. The official server returns
HTTP 401 without the temporary username and password issued after registration. Only
the following pose archive is required for the oracle gate:

```text
subject1_pose_v1_1.tar.gz    approximately 78 MB
```

RGB-D, the MANO-only archive, and the full multiview release are unnecessary at this
stage. Register at [the official H2O download page](https://h2odataset.ethz.ch/), then
extract the pose archive under:

```text
data/raw/sources/h2o/
```

Do not commit the archive or credentials. Both remain covered by the repository's
ignored `data/raw/` directory.

## Run

```bash
curl -L \
  https://raw.githubusercontent.com/taeinkwon/h2odataset/main/action_labels/action_train.txt \
  -o data/raw/sources/h2o/action_train.txt

PYTHONPATH=src .venv/bin/python scripts/evaluate_h2o_oracle_place.py \
  --dataset-root data/raw/sources/h2o \
  --action-index data/raw/sources/h2o/action_train.txt \
  --sequence subject1/h1/1 \
  --action-id 13 \
  --output results/h2o_oracle_place

MUJOCO_GL=egl PYTHONPATH=src:scripts .venv/bin/python \
  scripts/evaluate_h2o_oracle_retarget.py \
  --trajectory results/h2o_oracle_place/oracle-place-trajectory.npz \
  --oracle-report results/h2o_oracle_place/report.json \
  --episodes 20 \
  --output docs/experiments/h2o-oracle-place-retarget.json
```

The report rejects a trivial action unless object motion exceeds 3 cm translation or
15° rotation. It also requires at least eight stable grasp frames.

## Remaining experiments

1. Generate 100 validated demonstrations from the passing oracle trajectory and train
   the visual policy.
2. Replace oracle object poses with estimated poses and report Oracle versus Estimated.
3. Run the same comparison on `pour milk`, including XYZ-only versus full SE(3).

The [action-index audit](experiments/h2o-action-index-audit.json) records the public
sequence selection. H2O's official format is documented by the
[dataset repository](https://github.com/taeinkwon/h2odataset) and the
[ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kwon_H2O_Two_Hands_Manipulating_Objects_for_First_Person_Interaction_Recognition_ICCV_2021_paper.pdf).
