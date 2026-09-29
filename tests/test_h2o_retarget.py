from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from robosuite.utils import transform_utils as transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_h2o_oracle_retarget import osc_action  # noqa: E402


def test_osc_orientation_command_uses_convergent_sign() -> None:
    observation = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    target = transform.quat2mat(transform.axisangle2quat(np.array([0.1, 0.0, 0.0])))

    action = osc_action(observation, np.zeros(3), target, -1.0)

    # robosuite's OSC action convention is opposite the world-frame axis-angle sign.
    assert action[3] < 0
    assert np.allclose(action[4:6], 0, atol=1e-8)
    assert action[-1] == -1
