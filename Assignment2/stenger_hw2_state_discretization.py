import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "futura_env"))
from furuta_env import discretize_state

RESULTS = HERE / "results"

BINS = (
    np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
    np.array([-3.0, -1.5, 0.0, 1.5, 3.0]),
    np.deg2rad([-20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0]),
    np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]),
)
SHAPE = tuple(len(edges) + 1 for edges in BINS)
N_STATES = int(np.prod(SHAPE))
N_ACTIONS = 3


def wrap_to_pi(angle):
    angle = float(angle)
    if -np.pi <= angle <= np.pi:
        return angle
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def discretize(state):
    s = np.array(state, dtype=float)
    s[2] = wrap_to_pi(s[2])
    return discretize_state(s, BINS)


def state_index(state):
    return int(np.ravel_multi_index(discretize(state), SHAPE))


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
    for name, state in TEST_STATES:
        idx = discretize(state)
        flat = state_index(state)
        print(f"{name:34s} state={np.round(state, 4).tolist()} -> bins={idx} flat={flat}")


if __name__ == "__main__":
    with io.StringIO() as buffer:
        with contextlib.redirect_stdout(buffer):
            main()
        report = buffer.getvalue()
    print(report, end="")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "part2_discretization.txt").write_text(report, encoding="utf-8")
