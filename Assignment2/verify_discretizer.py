"""Part 2: report bin edges, state/Q counts, and boundary tests."""

import numpy as np

from io_utils import save_json
from tabular import Discretizer

N_ACTIONS = 3


def main():
    disc = Discretizer()
    edge20 = float(disc.bins[2][6])  # +20 deg boundary
    cases = [
        ("upright", [0, 0, 0, 0]),
        ("downward (+pi)", [0, 0, np.pi, 0]),
        ("downward (-pi)", [0, 0, -np.pi, 0]),
        ("theta near +pi", [0, 0, np.pi - 0.01, 0]),
        ("theta near -pi", [0, 0, -np.pi + 0.01, 0]),
        ("theta = pi+0.01 (wraps)", [0, 0, np.pi + 0.01, 0]),
        ("theta on +20 deg boundary", [0, 0, edge20, 0]),
        ("theta just below boundary", [0, 0, edge20 - 1e-9, 0]),
        ("phi_dot below range", [0, -100.0, 0, 0]),
        ("phi_dot above range", [0, 100.0, 0, 0]),
        ("theta_dot below range", [0, 0, 0, -100.0]),
        ("theta_dot above range", [0, 0, 0, 100.0]),
        ("phi above range", [4.0, 0, 0, 0]),
    ]
    tests = [
        {"name": n, "state": np.array(s, dtype=float),
         "tuple": disc.indices(s), "index": disc.index(s)}
        for n, s in cases
    ]
    out = {
        "edges": {k: b for k, b in zip(("phi", "phi_dot", "theta", "theta_dot"), disc.bins)},
        "shape": disc.shape,
        "n_states": disc.n_states,
        "n_actions": N_ACTIONS,
        "n_q": disc.n_states * N_ACTIONS,
        "tests": tests,
    }
    save_json("verify_discretizer", out)
    return out


if __name__ == "__main__":
    result = main()
    print("shape", result["shape"], "|S| =", result["n_states"], "|Q| =", result["n_q"])
    for t in result["tests"]:
        print(f'{t["name"]:30s} {t["state"]} -> {t["tuple"]} idx {t["index"]}')
