import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "futura_env"))
from furuta_env import FurutaEnv

from stenger_hw2_audit import td_target, td_update
from stenger_hw2_epsilon_greedy import action_probabilities, epsilon_greedy
from stenger_hw2_state_discretization import (
    BINS, N_ACTIONS, N_STATES, SHAPE, discretize, state_index, wrap_to_pi)


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def test_environment():
    env = FurutaEnv(seed=0)
    state, _ = env.reset()
    check("reset returns a 4-element state", state.shape == (4,))
    check("reset starts within 0.02 rad of upright", abs(state[2]) <= 0.02 and abs(state[0]) <= 0.02)
    torques = []
    for a in range(env.number_of_actions):
        env.reset(initial_state=state)
        torques.append(env.step(a)[4]["torque"])
    check("actions 0,1,2 map to -0.02, 0, +0.02 N m", np.allclose(torques, [-0.02, 0.0, 0.02]))
    try:
        env.step(3)
        check("illegal action is rejected", False)
    except ValueError:
        check("illegal action is rejected", True)
    env.reset(initial_state=[0, 0, np.pi, 0])
    _, r, term, trunc, _ = env.step(1)
    check("downward state terminates with reward -10 (env is a balancing task)", term and r == -10.0)
    short = FurutaEnv(max_steps=3)
    short.reset(initial_state=[0, 0, 0, 0])
    flags = [short.step(1)[2:4] for _ in range(3)]
    check("time limit sets truncated (not terminated)", flags[-1] == (False, True))


def test_discretizer():
    check("number of states = product of bin counts", N_STATES == int(np.prod([len(b) + 1 for b in BINS])))
    check("wrap_to_pi leaves +-pi unchanged", wrap_to_pi(np.pi) == np.pi and wrap_to_pi(-np.pi) == -np.pi)
    check("wrap_to_pi(3*pi/2) = -pi/2", np.isclose(wrap_to_pi(1.5 * np.pi), -0.5 * np.pi))
    rng = np.random.default_rng(1)
    extreme = rng.uniform(-100, 100, size=(2000, 4))
    flat = [state_index(s) for s in extreme]
    check("extreme states map to valid indices", min(flat) >= 0 and max(flat) < N_STATES)
    check("value on an edge goes to the upper bin", discretize([0, 1.5, 0, 0])[1] == 4)
    check("value just below an edge stays in the lower bin", discretize([0, 1.4999, 0, 0])[1] == 3)
    check("upright and downward map to different theta bins",
          discretize([0, 0, 0, 0])[2] != discretize([0, 0, np.pi, 0])[2])
    check("flat index round-trips", np.unravel_index(state_index([0.2, 0.8, -0.05, -0.7]), SHAPE)
          == discretize([0.2, 0.8, -0.05, -0.7]))


def test_epsilon_greedy():
    q = np.array([1.0, 2.0, 2.0])
    check("exact probabilities for eps=0.2", np.allclose(action_probabilities(q, 0.2), [0.2 / 3, 0.4 + 0.2 / 3, 0.4 + 0.2 / 3]))
    rng = np.random.default_rng(0)
    draws = [epsilon_greedy(q, 0.0, rng) for _ in range(2000)]
    check("eps=0 never picks a non-greedy action", 0 not in draws)
    check("eps=0 breaks the tie between both greedy actions", {1, 2} <= set(draws))
    rng = np.random.default_rng(0)
    counts = np.bincount([epsilon_greedy(q, 1.0, rng) for _ in range(30000)], minlength=3) / 30000
    check("eps=1 is uniform over all actions", np.allclose(counts, 1 / 3, atol=0.015))


def test_td():
    q_next = np.array([1.0, 5.0, 3.0])
    check("SARSA target uses Q(S',A')", td_target("sarsa", 1.0, 0.9, q_next, 2, False) == 1.0 + 0.9 * 3.0)
    check("Q-learning target uses max Q(S',.)", td_target("qlearning", 1.0, 0.9, q_next, 2, False) == 1.0 + 0.9 * 5.0)
    check("targets agree when A' is greedy", td_target("sarsa", 1.0, 0.9, q_next, 1, False)
          == td_target("qlearning", 1.0, 0.9, q_next, 1, False))
    check("terminal continuation is zero for both",
          td_target("sarsa", -10.0, 0.9, q_next * 1e6, 1, True) == -10.0
          and td_target("qlearning", -10.0, 0.9, q_next * 1e6, 1, True) == -10.0)
    q = np.zeros((N_STATES, N_ACTIONS))
    q[7] = q_next
    before = q.copy()
    td_update(q, 3, 1, "qlearning", 1.0, 0.9, 0.5, 7, None, False)
    check("update changes exactly one entry by alpha*(Y-Q)", np.isclose(q[3, 1], 0.5 * (1.0 + 4.5))
          and np.count_nonzero(q != before) == 1)


if __name__ == "__main__":
    test_environment()
    test_discretizer()
    test_epsilon_greedy()
    test_td()
    print("All checks passed.")
