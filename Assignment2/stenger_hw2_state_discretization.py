"""HW2 Part 2: state discretization and boundary tests.

The supplied environment is used exactly as given.  Discretization only uses
its ``discretize_state`` helper (np.digitize on internal bin edges).

Run:  python stenger_hw2_state_discretization.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "futura_env"))
from furuta_env import discretize_state  # noqa: E402

RESULTS = HERE / "results"

# State order: [phi, phi_dot, theta, theta_dot]
# phi   = horizontal arm angle (rad),   phi_dot   = arm angular velocity (rad/s)
# theta = pendulum angle (rad, 0 = upright, +-pi = downward), theta_dot = pendulum velocity (rad/s)
#
# Internal edges only: k edges -> k + 1 bins.  The two outer bins are open-ended,
# so any value below the first edge maps to bin 0 and any value above the last
# edge maps to the last bin (out-of-range values are clamped, never an error).
BINS = (
    np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),                       # phi: 6 bins
    np.array([-3.0, -1.5, 0.0, 1.5, 3.0]),                       # phi_dot: 6 bins
    np.deg2rad([-20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0]),  # theta: 10 bins
    np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]),            # theta_dot: 8 bins
)
SHAPE = tuple(len(edges) + 1 for edges in BINS)
N_STATES = int(np.prod(SHAPE))
N_ACTIONS = 3


def wrap_to_pi(angle):
    """wrapToPi: map an angle to [-pi, pi]; angles already in range are unchanged."""
    angle = float(angle)
    if -np.pi <= angle <= np.pi:
        return angle
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def discretize(state):
    """Return the 4-tuple of bin indices for a continuous state (theta is wrapped first)."""
    s = np.array(state, dtype=float)
    s[2] = wrap_to_pi(s[2])
    return discretize_state(s, BINS)


def state_index(state):
    """Flat Q-table row index in [0, N_STATES)."""
    return int(np.ravel_multi_index(discretize(state), SHAPE))


# (description, state) pairs; the test list required by the assignment.
TEST_STATES = [
    ("upright, at rest", [0.0, 0.0, 0.0, 0.0]),
    ("downward, at rest", [0.0, 0.0, np.pi, 0.0]),
    ("theta near -pi", [0.0, 0.0, -np.pi + 0.01, 0.0]),
    ("theta near +pi", [0.0, 0.0, np.pi - 0.01, 0.0]),
    ("theta = 3*pi/2 (wraps to -pi/2)", [0.0, 0.0, 1.5 * np.pi, 0.0]),
    ("phi_dot exactly on edge 1.5", [0.0, 1.5, 0.0, 0.0]),
    ("theta exactly on edge +5 deg", [0.0, 0.0, np.deg2rad(5.0), 0.0]),
    ("both velocities below range", [0.0, -10.0, 0.0, -14.0]),
    ("both velocities above range", [0.0, 10.0, 0.0, 14.0]),
    ("typical balancing state", [0.2, 0.8, -0.05, -0.7]),
]


def main():
    print("Bins per variable [phi, phi_dot, theta, theta_dot]:", SHAPE)
    print("Discrete states |S| =", N_STATES)
    print("Q-table entries |Q| = |S| * n_a =", N_STATES * N_ACTIONS)
    rows = []
    for name, state in TEST_STATES:
        idx = discretize(state)
        flat = state_index(state)
        print(f"{name:34s} state={np.round(state, 4).tolist()} -> bins={idx} flat={flat}")
        rows.append({"name": name, "state": [float(v) for v in state],
                     "theta_wrapped": wrap_to_pi(state[2]), "bins": list(idx), "flat": flat})
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "part2_discretization.json").write_text(json.dumps(
        {"shape": SHAPE, "n_states": N_STATES, "n_q": N_STATES * N_ACTIONS,
         "edges": [e.tolist() for e in BINS], "tests": rows}, indent=2))


if __name__ == "__main__":
    main()
