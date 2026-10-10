"""Episode, training and evaluation loops for tabular SARSA / Q-learning."""

from __future__ import annotations

import numpy as np

from furuta_env import FurutaEnv, wrap_to_pi
from tabular import Discretizer, epsilon_greedy, td_update


def run_episode(
    env, disc, Q, rng, epsilon, alpha, gamma,
    kind="qlearning", learn=True, visited=None, state_visits=None, seed=None,
):
    """Run one episode. Returns (undiscounted return, steps, outcome)."""
    state, _ = env.reset(seed=seed)
    s = disc.index(state)
    a = epsilon_greedy(Q[s], epsilon, rng)
    total, steps = 0.0, 0
    while True:
        next_state, reward, terminated, truncated, info = env.step(a)
        total += reward
        steps += 1
        s2 = disc.index(next_state)
        if visited is not None:
            visited[s, a] = True
        if state_visits is not None:
            state_visits[s] += 1
        a2 = None if terminated else epsilon_greedy(Q[s2], epsilon, rng)
        if learn:
            # Only `terminated` zeroes the continuation; truncation bootstraps.
            td_update(Q, s, a, kind, reward, gamma, alpha, s2, a2, terminated)
        if terminated or truncated:
            return total, steps, info["outcome"]
        s, a = s2, a2


def evaluate(env, disc, Q, seed):
    """Greedy rollout with frozen Q (eps = 0, ties -> lowest index, no updates)."""
    state, _ = env.reset(seed=seed)
    trace = [wrap_to_pi(state[2])]
    steps, outcome = 0, None
    while True:
        a = int(np.argmax(Q[disc.index(state)]))
        state, _, terminated, truncated, info = env.step(a)
        steps += 1
        trace.append(info["theta_wrapped"])
        if terminated or truncated:
            outcome = info["outcome"]
            break
    return {"theta_wrapped": np.array(trace), "outcome": outcome, "steps": steps}


def classify_evaluation(theta_wrapped, outcome, near_downward_rad=0.5):
    """Name what the greedy policy did (thresholds: 20 deg upright, 0.5 rad from downward)."""
    if outcome == "success":
        return "completed_5s_balance"
    th = np.abs(np.asarray(theta_wrapped, dtype=float))
    if th.min() <= np.deg2rad(20.0):
        return "reached_upright_region"
    if (np.pi - th).max() >= near_downward_rad:
        return "growing_swing"
    return "near_downward"


def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def train(
    seed, n_episodes, max_steps, alpha, gamma, eps_start, eps_end,
    kind="qlearning", disc=None,
):
    disc = disc or Discretizer()
    env = FurutaEnv(max_steps=max_steps, seed=seed)
    rng = np.random.default_rng(seed)
    Q = np.zeros((disc.n_states, env.number_of_actions))
    visited = np.zeros(Q.shape, dtype=bool)
    state_visits = np.zeros(disc.n_states, dtype=np.int64)
    returns = np.zeros(n_episodes)
    durations = np.zeros(n_episodes, dtype=np.int64)
    visited_frac = np.zeros(n_episodes)
    outcomes = []
    for ep in range(n_episodes):
        eps = eps_start + (eps_end - eps_start) * ep / max(n_episodes - 1, 1)
        ret, steps, outcome = run_episode(
            env, disc, Q, rng, eps, alpha, gamma, kind=kind,
            visited=visited, state_visits=state_visits,
        )
        returns[ep], durations[ep], visited_frac[ep] = ret, steps, visited.mean()
        outcomes.append(outcome)
    eval_env = FurutaEnv(max_steps=max_steps, seed=seed)
    ev = evaluate(eval_env, disc, Q, seed=10_000 + seed)
    return {
        "Q": Q, "returns": returns, "durations": durations,
        "visited_frac": visited_frac, "state_visits": state_visits,
        "outcomes": outcomes, "eval": ev,
        "eval_class": classify_evaluation(ev["theta_wrapped"], ev["outcome"]),
    }
