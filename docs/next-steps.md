# Next steps

## Current state

The 72-frame HOI4D ball-to-bowl sequence now supplies RGB, aligned sensor depth,
official motion masks, measured geometry, and a passing SAM2/CoTracker3 perception
run. The selected 6 GB model stack is:

- SAM2.1 Hiera Small for segmentation;
- CoTracker3 offline for point tracking;
- TripoSR at mesh resolution 128 for visual object proposals;
- Depth Anything V2 Metric Hypersim Small at input size 518 for depth structure;
- HOI4D RGB-D and measured primitives for scale and collision geometry;
- robosuite with the temporal visual waypoint policy for robot experiments.

TripoSR and Depth Anything have both run on the local machine. The learned HOI4D
meshes are too flat for collision use, and monocular depth has a large scale bias.
Those failures are contained by using measured geometry for physics.

The measured ball-to-bowl robosuite task is now implemented. Its scripted controller
passes 3/3 oracle-position rollouts, and the 2.61-million-parameter visual waypoint
model passes 5/5 randomized rollouts after training on 500 targeted localization
frames. Median held-out localization error is 5.7 mm and median closed-loop initial
error is 4.9 mm.

Gaussian appearance composition is also integrated. A mixed 2,000-frame locator
passes 5/5 paired clean and 5/5 Gaussian-composite rollouts. One complete composited
robot trajectory contains 350 synchronized dual-camera, state, and action samples.
That policy augmentation remains appearance-only. A separate semantic-anchor path now
registers robosuite to HOI4D metric cameras and produces a 353-sample dual-camera
demonstration with Gaussian/simulator depth ordering.

The selected 4,000-frame robustness checkpoint includes rendered ±2 cm camera jitter
and half-light data. It passes 97/100 total rollouts: 20/20 nominal, 19/20 camera,
20/20 lighting, 20/20 Gaussian appearance, and 18/20 combined. A five-seed paired
replication averages 76.0±5.8% on the combined shift, so the selected checkpoint does
not establish seed-stable robustness.

## Completion status

No work remains in the declared hardware-free project scope. All local data processing,
model comparisons, reconstruction and motion gates, Gaussian composition, simulation,
policy training, robustness evaluation, and deployment-software checks are complete.
The evaluation now records the failed seed-stability gate rather than treating the
selected checkpoint as a general robustness result.

Physical calibration and robot trials are excluded because no real devices are
available. They remain a future extension and are not a condition of project completion.

The correspondence-sensitive rotation ablation is complete on the asymmetric
`EM1-0406` action camera: 0.70 px median held-out error versus 34.89 px without
tracked correspondences. Its calibrated depth also closes the reconstruction
comparison: neither TripoSR nor Stable Fast 3D passes the 15% proportion gate, so
metric RGB-D geometry remains selected. Metric Gaussian insertion is implemented from the
shared table normal, bowl center, and source direction, including a 353-sample
depth-ordered dual-camera demonstration. The local physical software preflight now
passes: frozen-checkpoint smoke inference is within 3.84 mm, and 149 Cartesian
commands complete under 1 cm step and 8 cm/s limits. Gripper changes happen only at
stationary source and target waypoints. Policy task coordinates are explicitly
transformed into robot-base coordinates before safety checks. Hardware arming remains
disabled by the simulation fixture. The optional physical extension would need a robot,
two calibrated cameras, workspace measurements, an operator-approved safety envelope,
and objects matching the frozen geometry. See the
[`physical evaluation runbook`](physical-evaluation-runbook.md).

## Reproduce the final local preflight

Reproduce the hardware-neutral software preflight:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_physical_evaluation.py \
  --calibration configs/physical_calibration_simulation_fixture.json \
  --safety configs/physical_safety.yaml \
  --camera-reference configs/physical_camera_reference.json \
  --checkpoint results/policy/ball_localization_robust_4000/visual_waypoint.pt \
  --smoke-data results/simulation/ball_localization_robust_4000/demonstrations.npz \
  --output results/deployment/physical_preflight_sim \
  --allow-simulation-fixture
```

The detailed measurements are in
[`local-model-benchmarks.md`](local-model-benchmarks.md), and the gate sequence is in
[`roadmap.md`](roadmap.md).
