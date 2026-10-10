"""Tabular RL building blocks: discretizer, epsilon-greedy policy, TD updates."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from furuta_env import discretize_state, wrap_to_pi


def default_bins() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal bin edges for [phi, phi_dot, theta_wrapped, theta_dot]."""
    phi = np.linspace(-np.pi, np.pi, 6)[1:-1]
    phi_dot = np.linspace(-10.0, 10.0, 6)[1:-1]
    theta = np.deg2rad([-165, -135, -105, -75, -45, -20, 20, 45, 75, 105, 135, 165])
    theta_dot = np.linspace(-8.0, 8.0, 8)[1:-1]
    return phi, phi_dot, theta, theta_dot


class Discretizer:
    """Maps a continuous state to a flat Q-table row index.

    ``np.digitize`` with *internal* edges gives open-ended outer bins, so any
    out-of-range value is clamped into the first or last bin.  theta is wrapped
    to [-pi, pi] first.  A value exactly on an edge belongs to the upper bin.
    """

    def __init__(self, bins=None) -> None:
        if bins is None:
            bins = default_bins()
        self.bins = tuple(np.asarray(b, dtype=float) for b in bins)
        self.shape = tuple(len(b) + 1 for b in self.bins)
        self.n_states = int(np.prod(self.shape))

    def indices(self, state) -> Tuple[int, int, int, int]:
        s = np.array(state, dtype=float)
        if s.shape != (4,):
            raise ValueError("state must have shape (4,)")
        if not np.all(np.isfinite(s)):
            raise ValueError("state must be finite")
        s[2] = wrap_to_pi(s[2])
        return discretize_state(s, self.bins)

    def index(self, state) -> int:
        return int(np.ravel_multi_index(self.indices(state), self.shape))

    def unravel(self, flat: int) -> Tuple[int, int, int, int]:
        return tuple(int(i) for i in np.unravel_index(flat, self.shape))


def greedy_actions(q_row: np.ndarray) -> np.ndarray:
    """All actions whose value equals the maximum (exact ties)."""
    return np.flatnonzero(q_row == np.max(q_row))


def action_probabilities(q_row: np.ndarray, epsilon: float) -> np.ndarray:
    """Exact action probabilities implied by ``epsilon_greedy``."""
    n = len(q_row)
    greedy = greedy_actions(q_row)
    p = np.full(n, epsilon / n)
    p[greedy] += (1.0 - epsilon) / len(greedy)
    return p


def epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """With prob. epsilon pick uniformly among ALL actions (greedy ones included);
    otherwise pick uniformly among the tied greedy actions."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    return int(rng.choice(greedy_actions(q_row)))


def td_target(
    kind: str,
    reward: float,
    gamma: float,
    q_next_row: np.ndarray,
    a_next: Optional[int],
    terminated: bool,
) -> float:
    """Y = R + gamma * continuation; continuation is 0 iff the transition terminated.

    ``kind`` is "sarsa" (uses Q(S', A')) or "qlearning" (uses max_a Q(S', a)).
    Time-limit truncation is NOT terminal and still bootstraps.
    """
    if kind not in ("sarsa", "qlearning"):
        raise ValueError(f"unknown kind {kind!r}")
    if terminated:
        return float(reward)
    if kind == "sarsa":
        continuation = q_next_row[a_next]
    else:
        continuation = np.max(q_next_row)
    return float(reward + gamma * continuation)


def td_update(
    Q: np.ndarray,
    s: int,
    a: int,
    kind: str,
    reward: float,
    gamma: float,
    alpha: float,
    s_next: int,
    a_next: Optional[int],
    terminated: bool,
) -> float:
    """In-place update Q[s,a] += alpha * (Y - Q[s,a]); returns the target Y."""
    target = td_target(kind, reward, gamma, Q[s_next], a_next, terminated)
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return target
