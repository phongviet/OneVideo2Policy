# Roadmap and gates

## Current milestone

The local pipeline now runs segmentation, tracking, TripoSR reconstruction proposals,
Depth Anything V2 Small, measured collision geometry, robosuite demonstration
generation, and policy evaluation on the 6 GB machine.

Completed evidence:

1. SAM2.1 Hiera Small and CoTracker3 pass the HOI4D perception gate.
2. HOI4D RGB-D provides metric dimensions and depth reference.
3. TripoSR and Depth Anything V2 Small fit local VRAM and have measured reports.
4. Primitive collision assets and the point-robot systems path run end to end.
5. One hundred successful Panda demonstrations were generated in robosuite.
6. The temporal visual waypoint policy passes 19/20 held-out rollouts.
7. Metric RGB-D odometry and the target-relative ball trajectory cover all 72 frames.
8. A fused Gaussian scene renders the animated ball with 0.28 px median centroid error;
   held-out footprint tuning improves three independent views by 0.49 dB mean PSNR.
9. The measured ball-to-bowl simulator passes 3/3 exact-position controller rollouts.
10. The compact learned visual waypoint policy passes 5/5 randomized measured-task
    rollouts with 4.9 mm median initial XY error.
11. Gaussian appearance composition produces synchronized robot demonstrations; the
    mixed model passes 5/5 paired clean and 5/5 composited rollouts.
12. The selected robustness checkpoint passes 97/100 across nominal, camera,
    lighting, Gaussian-background, and combined 20-episode conditions. A later
    five-seed replication obtains 76.0±5.8% mean combined-shift success, showing
    that the original checkpoint does not establish seed-stable robustness.
13. Semantic metric anchors register robosuite into HOI4D with a 2.7 mm table-plane
    residual and produce a 353-sample depth-ordered dual-camera Gaussian demo.
14. The physical deployment preflight validates calibration, freezes artifact hashes,
    smoke-tests checkpoint inference to 3.84 mm, and safety-checks a 149-command
    Cartesian dry run. A paired-trial validator enforces the final hardware gate.
15. Printable watertight source and target fixtures match the frozen policy geometry
    to STL float precision and include reproducible hash and manifold audits.
16. A print-ready ChArUco target and automatic detector produce native intrinsics,
    distortion, robot-frame correspondences, and held-out camera calibration inputs.

## Evaluation gates

### Gate A — reconstruction quality (evaluated; learned meshes rejected for metrics)

The retained Record3D `EM1-0406` sequence provides a 282×282 asymmetric action-camera
crop and 609 aligned metric-depth samples. Its robust visible-surface extents are
81.9 × 60.6 × 35.4 mm. After isotropic longest-axis alignment, TripoSR predicts
81.9 × 69.7 × 14.5 mm and Stable Fast 3D predicts 81.9 × 68.3 × 23.7 mm. Neither
passes the frozen requirement that both secondary extents fall within 15% of the
RGB-D reference. Stable Fast 3D is closer, but is non-watertight and peaks at
6,169 MiB; TripoSR is watertight and peaks at 1,867 MiB. The project therefore uses
Record3D/HOI4D metric depth or fitted primitives for geometry and keeps TripoSR only
as a fast visual proposal. The comparison requirement is complete without promoting
an inaccurate learned mesh into physics.

### Gate B — motion quality (passed for observable image-plane rotation)

Camera and spherical-object translation pass their metric artifact checks. The retained
Record3D `EM1-0406` sequence supplies a nonsymmetric action camera with a rectangular
body and offset lens. Across 13 fully visible post-placement frames, a correspondence
constrained similarity fit reduces median error on 82 held-out points from 34.89 px
for mask-only pose fitting to 0.70 px, a 98.0% reduction. The recovered median image
plane rotation is −41.7°. This closes the tracked-point rotation ablation; full SE(3)
object-pose accuracy remains outside this 2D test.

### Gate C — task transfer (passed locally)

The measured HOI4D ball-and-bowl geometry is retargeted into robosuite. Successful
oracle and learned-policy rollouts verify grasp, transfer, release, and placement.

### Gate D — policy evidence (evaluated locally; seed-stable gate not passed)

The selected waypoint policy passes 97/100 total held-out simulated rollouts across
object pose, ±2 cm camera translation, half lighting, Gaussian background, and their
combination. The combined condition passes 18/20. Five independently trained models
subsequently average 76.0±5.8% over 50 paired combined-shift episodes each (range
68–82%), below the frozen 80% gate. The result supports the value of matched training
coverage while rejecting a seed-stable robustness claim. This completes the planned
hardware-free evaluation scope.

### Optional Gate E — physical validation (out of scope)

The local deployment package and software preflight are complete. The preflight uses
the frozen policy checkpoint, transforms policy outputs into the robot-base frame,
enforces 1 cm steps, stationary gripper transitions,
8 cm/s speed, a 15 N force limit, workspace bounds, and the exact dual-camera training
geometry. It refuses to arm from the simulation calibration fixture.
The physical object gate also requires the 3.832 cm source and 9.912 × 5.693 cm target
geometry used to train and evaluate the selected policy; the separate 3 cm filmed
object is outside this tolerance.
No physical devices are available for this project, so physical calibration and robot
trials are explicitly excluded from completion. The repository retains the deployment
package and runbook as an optional future extension. If hardware becomes available,
five paired physical trials per controller would be required; both scripted and learned
controllers would need at least 4/5 successes with zero safety aborts or force
violations. Follow [`physical-evaluation-runbook.md`](physical-evaluation-runbook.md).

## Completion status

Gates A through D are complete. Gate E is outside the declared hardware-free scope.
The learned policy is evaluated in simulation under controlled camera, lighting,
Gaussian-background, and combined shifts. Its selected checkpoint is strong, but the
combined-shift result is seed-sensitive. No physical-policy or sim-to-real claim is made.

## Next research milestone — H2O oracle Place

The four-action HOI4D expansion is complete as a boundary study. The next experiment
uses one public H2O `place milk` interval to isolate downstream transfer from perception:

1. Load provided object SE(3), camera pose, and two-hand pose annotations.
2. Infer the object-frame grasp automatically from stable hand-object coupling.
3. Retarget the relative SE(3) motion to a Panda in MuJoCo.
4. Require at least 10/20 successful perturbed Place rollouts before generating 100
   demonstrations or training another policy.
5. Compare oracle and estimated object poses only after the oracle path passes.

See [`h2o-oracle-place.md`](h2o-oracle-place.md). The official pose archive is gated by
the H2O academic-use registration and is the only external input still required.
