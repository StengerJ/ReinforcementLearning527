"""Swing-up task wrapped around the unmodified, professor-supplied ``FurutaEnv``.

Why this wrapper exists: the supplied environment is a *balancing* task (it
starts near upright and terminates as soon as |theta| > 20 degrees).  HW2 needs
*swing-up* episodes (downward start; leaving the upright region must not end
the episode; success = 5 s inside the +-20 degree region; unsafe = arm limit;
time-limit truncation).  The supplied file is left untouched, so this wrapper
uses it only for the dynamics (RK4 step, torque table, arm limit) and
recomputes termination, truncation, outcome and reward from the returned state.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import hw2  # noqa: F401  (puts starter/ on sys.path)
from furuta_env import FurutaEnv


def wrap_to_pi(angle):
    """Wrap angle(s) to [-pi, pi]; values already in range are returned untouched."""
    a = np.asarray(angle, dtype=float)
    wrapped = np.where(np.abs(a) <= np.pi, a, (a + np.pi) % (2.0 * np.pi) - np.pi)
    return float(wrapped) if wrapped.ndim == 0 else wrapped


class SwingUpTask:
    """Gym-like swing-up episodes on top of an unmodified ``FurutaEnv``."""

    ANGLE_LIMIT = np.deg2rad(20.0)

    def __init__(self, dt: float = 0.01, max_steps: int = 2000, seed: Optional[int] = None,
                 success_hold_seconds: float = 5.0) -> None:
        self.env = FurutaEnv(dt=dt, max_steps=max_steps, seed=seed)
        self.dt = self.env.dt
        self.max_steps = int(max_steps)
        self.torque_values = self.env.torque_values
        self.success_steps = int(round(success_hold_seconds / self.dt))
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.upright_steps = 0

    @property
    def number_of_actions(self) -> int:
        return self.env.number_of_actions

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    @state.setter
    def state(self, value) -> None:
        self.env.state = np.asarray(value, dtype=float)

    def reset(self, seed: Optional[int] = None,
              initial_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, object]]:
        """Begin a swing-up episode near the downward equilibrium (theta = pi)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if initial_state is None:
            initial_state = np.array(
                [
                    self.rng.uniform(-0.05, 0.05),
                    0.0,
                    np.pi + self.rng.uniform(-0.05, 0.05),
                    0.0,
                ],
                dtype=float,
            )
        state, _ = self.env.reset(initial_state=initial_state)
        self.steps = 0
        self.upright_steps = 0
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, object]]:
        """Advance the dynamics; ignore the env's termination/truncation/reward."""
        state, env_reward, env_terminated, _, env_info = self.env.step(action)
        self.steps += 1

        torque = float(env_info["torque"])
        theta = float(state[2])
        phi = float(state[0])
        theta_wrapped = wrap_to_pi(theta)

        # Leaving the upright region is NOT a termination condition: a swing-up
        # episode must pass through every angle.  Only the arm limit is unsafe.
        if abs(theta_wrapped) <= self.ANGLE_LIMIT:
            self.upright_steps += 1
        else:
            self.upright_steps = 0

        unsafe = bool(abs(phi) > self.env.arm_angle_limit)
        success = (not unsafe) and self.upright_steps >= self.success_steps
        terminated = unsafe or success
        truncated = bool(self.steps >= self.max_steps) and not terminated

        if unsafe:
            outcome = "unsafe"
        elif success:
            outcome = "success"
        elif truncated:
            outcome = "timeout"
        else:
            outcome = None

        # Non-negative living reward: terminating is never better than surviving.
        reward = float(0.5 * (1.0 + np.cos(theta_wrapped)) - 0.001 * torque**2)
        if unsafe:
            reward = -10.0

        info = {
            "torque": torque,
            "theta": theta,
            "theta_wrapped": theta_wrapped,
            "phi": phi,
            "outcome": outcome,
            "env_terminated": bool(env_terminated),
            "env_reward": float(env_reward),
        }
        return state.copy(), reward, terminated, truncated, info
