<div align="center">

# OneVideo2Policy

### One recorded manipulation sequence → metric scene → synthetic robot data → closed-loop policy

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-77%20passing-2EA44F)](#reproduce-locally)
[![Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Scope](https://img.shields.io/badge/scope-hardware--free-orange)](#scope-and-limitations)

<img src="docs/assets/metric-registered-demo.gif" width="760" alt="Metric registered Gaussian scene with a Panda robot performing the ball-to-bowl task">

<sub>Real project output: a 353-step Panda trajectory depth-composited into two registered views of the reconstructed HOI4D scene.</sub>

</div>

## Problem definition

Robot policies need variation in object pose, viewpoint, lighting, and scene appearance.
A single human video provides only one instance of each, while collecting more robot
trajectories requires hardware and operator time.

**Research question:** can one recorded manipulation be converted into geometrically
consistent robot training data, on a small local machine, and improve closed-loop
robustness under controlled shifts?

The policy study uses one **HOI4D RGB-D ball-to-bowl sequence**. A separate 61-sequence
audit tests whether the input and skill representation extend to mug, kettle, trash-can,
and storage-furniture actions. The final policy claim remains limited to hardware-free
robosuite evaluation.

## Contributions

1. **A small-device reproduction pipeline.** SAM2.1 Small, CoTracker3, TripoSR-128,
   RGB-D geometry, and a 2.61M-parameter policy replace the larger TRELLIS/VGGT path.
2. **A hybrid metric scene.** Gaussian splats model appearance; depth-fitted primitives
   provide scale and collisions after learned meshes fail the metric geometry gate.
3. **A controlled robustness study.** Paired data-coverage and camera ablations plus
   five training seeds distinguish a strong checkpoint from repeatable performance.

## Results at a glance

| Measurement | Result |
|---|---:|
| SAM2.1 Small mask IoU | source **0.777**, target **0.920** |
| CoTracker3 survival / identity swaps | **0.914 / 0.951 / 0** |
| Tracking ablation error | **34.89 px → 0.70 px** |
| Metric scene | **7,007 Gaussians**, **+0.49 dB** held-out PSNR |
| Robot-scene registration | **2.7 mm** plane residual, **353 steps** |
| Policy training | **4,000 frames**, **55.4 s** |
| Selected checkpoint | **97/100** across five conditions |
| Five-seed combined shift | **76.0 ± 5.8%**, 190/250 pooled |
| Cross-action pipeline | **4 actions × 8 gates**, **0/4** strict passes |
| H2O oracle SE(3) retargeting | **20/20** perturbed Panda Place rollouts |
| Repository checks | **77 tests**, Ruff clean |

Machine-readable results are under [`docs/experiments`](docs/experiments).

## Pipeline

<img src="docs/assets/project-overview.png" width="100%" alt="Measured five-stage pipeline using actual project outputs: HOI4D RGB-D input, SAM2 and CoTracker perception, a Gaussian metric scene, registered robot synthesis, and closed-loop policy evaluation">

**Pipeline.** One RGB-D demonstration is segmented and tracked, converted into a
metric Gaussian scene and object-relative skill, transferred to a Panda, and randomized
across pose, camera, lighting, and background. The resulting 4,000 dual-view frames train
a compact waypoint model evaluated with paired shifts and five training seeds.

## Real generated outputs

<table>
<tr>
<td width="50%" align="center"><img src="docs/assets/metric-registered-demo.gif" width="100%" alt="Dual-view metric registered Gaussian robot demonstration"></td>
<td width="50%" align="center"><img src="docs/assets/gaussian-object-trajectory.gif" width="100%" alt="Recovered ball trajectory animated in the fused Gaussian scene"></td>
</tr>
<tr>
<td><b>Metric robot demonstration.</b> Two robosuite cameras registered to HOI4D task anchors. Robot pixels are depth-composited with the Gaussian scene; foreground visibility is 99.2–99.4%.</td>
<td><b>Recovered object motion.</b> The ball trajectory is replayed through a fused 7,007-Gaussian RGB-D scene. This output drives background and motion-conditioned data generation.</td>
</tr>
</table>

## Experiments

### 1. Synthetic data coverage matters

<img src="docs/assets/policy-experiment-summary.png" width="100%" alt="Coverage ablation and camera displacement sensitivity charts">

All rows use the same 50 paired combined-shift episodes. Counts are successes / episodes.

| Training coverage | Data size | Combined-shift result |
|---|---:|---:|
| One recorded demonstration | 1 trajectory | **3/50 (6%)** |
| Clean simulator frames | 500 | **1/50 (2%)** |
| Appearance + object pose | 2,000 | **10/50 (20%)** |
| + camera variation | 3,000 | **25/50 (50%)** |
| + combined perturbations | 4,000 | **41/50 (82%)** |
| Privileged oracle ceiling | — | **50/50 (100%)** |

The clean 500-frame model is worse than the one-demo fit on this shifted test. The gain
appears only as the training distribution gains pose, appearance, camera, and combined
coverage; sample count alone does not explain it.

### 2. The selected checkpoint is strong but seed-sensitive

| Test condition | Selected checkpoint |
|---|---:|
| Nominal | **20/20** |
| Camera shift (2 cm) | **19/20** |
| Half lighting | **20/20** |
| Gaussian background | **20/20** |
| All shifts combined | **18/20** |
| **Total** | **97/100** |

Five independent training seeds on a larger, fixed combined-shift test produced
**82%, 68%, 78%, 72%, and 80%** success. The mean is **76.0%**, the sample standard
deviation is **5.8 percentage points**, and the pooled count is **190/250**. This is
the primary reliability result; 97/100 describes one selected checkpoint.

### 3. Camera displacement is the clearest failure axis

| Camera displacement | 0 cm | 1 cm | 2 cm | 3 cm | 4 cm |
|---|---:|---:|---:|---:|---:|
| Success | **49/50** | **47/50** | **38/50** | **30/50** | **25/50** |
| Rate | 98% | 94% | 76% | 60% | 50% |

Performance falls monotonically beyond 1 cm, showing that viewpoint coverage remains
the largest measured weakness of the compact visual policy.

### 4. Tracked correspondences recover motion that masks miss

<table>
<tr>
<td width="50%"><img src="docs/assets/em1-0406-rotation-ablation.png" width="100%" alt="Rotation tracking ablation"></td>
<td width="50%"><img src="docs/assets/em1-0406-reconstruction-proportions.png" width="100%" alt="Single-image reconstruction proportion comparison"></td>
</tr>
<tr>
<td>Across 13 frames and 82 held-out correspondences, tracked fitting cuts median error by <b>98.0%</b> and recovers a median <b>−41.7°</b> image-plane rotation.</td>
<td>Stable Fast 3D is proportionally closer than TripoSR, but neither learned mesh passes the 15% metric extent gate. RGB-D geometry is retained for physics.</td>
</tr>
</table>

| Reconstruction | Scale-aligned extents | Peak GPU | Watertight | Max secondary error |
|---|---:|---:|:---:|---:|
| RGB-D reference | 81.9 × 60.6 × 35.4 mm | — | — | — |
| TripoSR-128 | 81.9 × 69.7 × 14.5 mm | **1.87 GB** | Yes | 58.9% |
| Stable Fast 3D | 81.9 × 68.3 × 23.7 mm | **6.17 GB** | No | 33.1% |

### 5. Other HOI4D actions expose the full pipeline boundary

<img src="docs/assets/hoi4d-cross-action-results.png" width="100%" alt="HOI4D cross-action audit with five real sequence frames, action coverage, object visibility, and representation requirements">

Across 61 local sequences, all have 300 object-pose records and 300 motion masks. The
five-sequence audit covers rigid pick/place, pouring, and articulated open/close motion.
SAM2.1 Small reaches **0.946 mug IoU** and **0.971 kettle IoU**, but a one-point prompt
fails on the trash-can lid and storage door. Part-aware negatives recover the storage
door to **0.802 IoU** and leave the trash-can lid unresolved at 0.212.

<img src="docs/assets/hoi4d-whole-pipeline-results.png" width="100%" alt="Pass and fail matrix for eight pipeline stages on four HOI4D action classes">

The follow-up runs all eight pipeline gates on the four new RGB-D actions. RGB-D
odometry passes every row at **0.72–0.76 px** reprojection and **2.5–3.3 mm** depth
residual. The mug reaches metric motion; kettle, trash-can, and storage sequences fail
one or more perception or geometry gates. All rows then fail action-aware skill or task
generation, and none has a matching trained visual policy. The strict result is
**0/4 end-to-end passes**, which localizes the remaining work beyond SAM2. See the
[whole-pipeline protocol and analysis](docs/hoi4d-whole-pipeline-study.md), the
[input audit](docs/hoi4d-cross-action-study.md), and the
[machine-readable pipeline results](docs/experiments/hoi4d-whole-pipeline-results.json).

## Reproduce locally

### Lightweight package and checks

```bash
git clone https://github.com/phongviet/OneVideo2Policy-From-one-human-video-to-robust-robot-manipulation-data.git
cd OneVideo2Policy-From-one-human-video-to-robust-robot-manipulation-data
uv sync --extra dev --extra video
uv run ov2p validate-config configs/place.yaml
uv run ruff check .
uv run pytest
```

### Prepare a video or RGB-D sequence

```bash
uv run ov2p prepare-video path/to/demo.mp4 \
  --output data/interim/demo --fps 30 --max-width 960
uv run ov2p validate-manifest data/interim/demo/manifest.json
```

For RGB-D input, add `--depth-dir path/to/depth`. The output manifest records original
frame IDs, timestamps, image paths, depth alignment, and both source and inference
resolutions.

### Run the CPU-safe end-to-end baseline

```bash
uv run ov2p run-local-e2e \
  --config configs/two_paths.yaml \
  --manifest data/interim/hoi4d_ball_to_bowl_gate/manifest.json \
  --masks data/interim/hoi4d_ball_to_bowl_gate/masks \
  --output results/local_e2e/hoi4d_ball_to_bowl
```

### Re-run the policy study

The full study needs robosuite, MuJoCo EGL, PyTorch, and the already generated local
training data. It runs 5 training seeds and 800 evaluation episodes:

```bash
MUJOCO_GL=egl PYTHONPATH=scripts .venv/bin/python scripts/run_report_experiments.py
```

Re-run the four-action pipeline matrix after preparing the corresponding HOI4D RGB-D
sequences and SAM predictions:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_hoi4d_cross_action_tracking.py
PYTHONPATH=src .venv/bin/python scripts/evaluate_hoi4d_whole_pipeline.py
```

Generated datasets, checkpoints, rollouts, and reports stay local and are ignored by
Git. Compact, reviewable measurements are committed under [`docs/experiments`](docs/experiments).

## Repository map

```text
configs/                 frozen task, model, metric, and gate settings
docs/assets/             real GIFs, figures, and diagrams used in this README
docs/experiments/        compact JSON records for reported measurements
scripts/                 data generation, model training, rendering, and evaluation
src/onevideo2policy/     reusable video, geometry, tracking, generation, and policy code
tests/                   fast CPU-only unit and integration tests
data/, results/, weights/ local inputs and generated artifacts (ignored)
reports/                 local paper sources and builds (ignored)
```

More detail: [two-path execution](docs/two-path-execution.md),
[Gaussian splatting](docs/gaussian-splatting.md),
[local model benchmarks](docs/local-model-benchmarks.md), and
[Video2Robo gap audit](docs/video2robo-gap-audit.md). The
[H2O oracle Place](docs/h2o-oracle-place.md) gate now passes 20/20 perturbed rollouts;
the next experiment trains a policy from those validated trajectories and compares
oracle with estimated object poses.

## Scope and limitations

- The evidence covers one rigid ball-to-bowl **Place** task in simulation.
- Four additional HOI4D actions were evaluated through all available pipeline stages;
  none has a matching task-specific visual policy, so all fail the final policy gate.
- The selected metric path uses one RGB-D sequence, so the final system is not a
  strictly monocular reconstruction pipeline.
- Gaussian splats provide appearance; fitted metric primitives provide collision
  geometry because both learned mesh alternatives failed the dimension gate.
- The learned network estimates waypoints; grasp phase logic and low-level control are
  scripted.
- Camera variation remains a major failure mode, and five-seed performance is below the
  frozen 80% robustness target.
- Physical robot execution, sim-to-real success, deformable objects, bimanual control,
  and VLA training were not evaluated.

## References

- [Video2Robo: 3DGS-based Synthetic Data from One Video Enables Scalable Robot Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Deng_Video2Robo_3DGS-based_Synthetic_Data_from_One_Video_Enables_Scalable_Robot_CVPR_2026_paper.html)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [CoTracker3](https://github.com/facebookresearch/co-tracker)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d)
- [HOI4D](https://hoi4d.github.io/)
- [robosuite](https://robosuite.ai/)
- [MuJoCo](https://mujoco.org/)

This is an independent reproduction and is not affiliated with the referenced authors
or projects. Third-party models and datasets are not vendored; follow their licenses
and access terms.

## License

Released under the [MIT License](LICENSE).
