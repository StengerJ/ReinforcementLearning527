import contextlib
import io

import numpy as np

from stenger_midterm_agent import BALANCE, HERE, N_ACTIONS, SWINGUP, Discretizer

RESULTS = HERE / "results"
NAMES = ["phi", "phi_dot", "theta~", "theta_dot"]


def probe_values(edges, is_angle):
    if is_angle:
        return [-np.pi, -np.pi + 1e-3, edges[0], 0.5 * (edges[0] + edges[-1]), edges[-1], np.pi - 1e-3, np.pi,
                1.5 * np.pi, -1.5 * np.pi]
    span = edges[-1] - edges[0]
    return [edges[0] - span, edges[0] - 1e-6, edges[0], 0.5 * (edges[0] + edges[-1]), edges[-1] - 1e-6,
            edges[-1], edges[-1] + span, 1e6]


def report(task):
    disc = Discretizer(task.bins)
    print(f"=== {task.name} discretization ===")
    for name, edges, n in zip(NAMES, disc.bins, disc.shape):
        shown = np.rad2deg(edges) if name == "theta~" else edges
        unit = "deg" if name == "theta~" else ""
        print(f"  {name:9s} {n:2d} bins, interior edges {unit} {np.round(shown, 3).tolist()}")
    print(f"  states |S| = {' x '.join(map(str, disc.shape))} = {disc.n_states}")
    print(f"  Q-table entries |S| x |A| = {disc.n_states} x {N_ACTIONS} = {disc.n_states * N_ACTIONS}")
    print("  boundary tests (other components held at 0, theta~ test uses raw theta):")
    for i, (name, edges) in enumerate(zip(NAMES, disc.bins)):
        for v in probe_values(edges, name == "theta~"):
            state = np.zeros(4)
            state[i] = v
            b = disc.bins_of(state)
            flat = disc.index(state)
            assert 0 <= b[i] < disc.shape[i] and 0 <= flat < disc.n_states
            print(f"    {name:9s} = {v:+14.6f} -> bin {b[i]:2d} of {disc.shape[i]}, flat {flat}")
    up = disc.bins_of([0, 0, 0, 0])[2]
    down = disc.bins_of([0, 0, np.pi, 0])[2]
    print(f"  upright theta~ bin = {up}, downward theta~ bin = {down}\n")


def main():
    report(BALANCE)
    report(SWINGUP)


if __name__ == "__main__":
    with io.StringIO() as buffer:
        with contextlib.redirect_stdout(buffer):
            main()
        text = buffer.getvalue()
    print(text, end="")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "part2_discretization.txt").write_text(text, encoding="utf-8")
