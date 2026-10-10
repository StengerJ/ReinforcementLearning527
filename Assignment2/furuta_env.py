"""Minimal Furuta-pendulum environment for tabular reinforcement learning.

This is a Python restructuring of the supplied MATLAB model.  It uses the
lumped parameters reported in Gaefvert (1998); it is not a calibrated model
of the Quanser QUBE-Servo 3.

State convention
----------------
x = [phi, phi_dot, theta, theta_dot]

The upright equilibrium is theta = 0 and the downward equilibrium is
|theta| = pi.  Episodes start near the downward equilibrium (swing-up task).
The only control input is the torque applied to the horizontal arm.  For a
tabular agent, an integer action is mapped to one of a small set of torque
values.

HW2 modification: the supplied starter was a balancing task (upright start,
termination at 20 degrees).  It was edited so that leaving the upright region
does NOT terminate an episode, success means holding |wrapToPi(theta)| <= 20
degrees for five seconds, and ``info["outcome"]``
reports how an episode ended.  The per-step reward is
(1 + cos(theta_wrapped)) / 2 (0 at the bottom, 1 upright) minus a tiny torque
penalty, and -10 for an unsafe termination.  The living reward is non-negative
so ending an episode early is never better than continuing (a negative living
reward made crashing into the arm limit look attractive).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def wrap_to_pi(angle):
    """Wrap angle(s) to [-pi, pi]; values already in range are returned untouched."""
    a = np.asarray(angle, dtype=float)
    wrapped = np.where(np.abs(a) <= np.pi, a, (a + np.pi) % (2.0 * np.pi) - np.pi)
    return float(wrapped) if wrapped.ndim == 0 else wrapped


@dataclass(frozen=True)
class FurutaParameters:
    """Lumped parameters from the supplied MATLAB simulation."""

    alpha: float = 0.0033472
    beta: float = 0.0038852
    gamma: float = 0.0024879
    delta: float = 0.097625


class FurutaEnv:
    """Small Gym-like environment with ``reset`` and ``step`` methods.

    The environment intentionally has no dependency on Gymnasium.  This keeps
    the starter code transparent and allows students to see the complete
    interaction loop.
    """

    def __init__(
        self,
        dt: float = 0.01,
        max_steps: int = 2000,
        torque_values: Tuple[float, ...] = (-0.02, 0.0, 0.02),
        angle_limit: float = np.deg2rad(20.0),
        arm_angle_limit: float = np.pi,
        process_noise_std: float = 0.0,
        seed: Optional[int] = None,
        success_hold_seconds: float = 5.0,
    ) -> None:
        self.p = FurutaParameters()
        self.dt = float(dt)
        self.success_steps = int(round(success_hold_seconds / self.dt))
        self.upright_steps = 0
        self.max_steps = int(max_steps)
        self.torque_values = np.asarray(torque_values, dtype=float)
        self.angle_limit = float(angle_limit)
        self.arm_angle_limit = float(arm_angle_limit)
        self.process_noise_std = float(process_noise_std)
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=float)
        self.steps = 0

    @property
    def number_of_actions(self) -> int:
        return int(self.torque_values.size)

    def reset(
        self,
        seed: Optional[int] = None,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Begin a swing-up episode near the downward equilibrium."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if initial_state is None:
            # Near the downward equilibrium; small noise makes episodes differ.
            self.state = np.array(
                [
                    self.rng.uniform(-0.05, 0.05),
                    0.0,
                    np.pi + self.rng.uniform(-0.05, 0.05),
                    0.0,
                ],
                dtype=float,
            )
        else:
            initial_state = np.asarray(initial_state, dtype=float)
            if initial_state.shape != (4,):
                raise ValueError("initial_state must have shape (4,)")
            self.state = initial_state.copy()

        self.steps = 0
        self.upright_steps = 0
        return self.state.copy(), {}

    def dynamics(self, state: np.ndarray, torque: float) -> np.ndarray:
        """Evaluate the continuous-time nonlinear equations of motion."""
        phi, phi_dot, theta, theta_dot = np.asarray(state, dtype=float)
        del phi  # The ideal model is rotationally symmetric in phi.

        alpha = self.p.alpha
        beta = self.p.beta
        gamma = self.p.gamma
        delta = self.p.delta

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        scale = 1.0 / (
            alpha * beta
            - gamma**2
            + (beta**2 + gamma**2) * sin_theta**2
        )

        phi_ddot = scale * (
            beta * gamma * (sin_theta**2 - 1.0) * sin_theta * phi_dot**2
            - 2.0 * beta**2 * cos_theta * sin_theta * phi_dot * theta_dot
            + beta * gamma * sin_theta * theta_dot**2
            - gamma * delta * cos_theta * sin_theta
            + beta * torque
        )

        theta_ddot = scale * (
            beta
            * (alpha + beta * sin_theta**2)
            * cos_theta
            * sin_theta
            * phi_dot**2
            - 2.0
            * beta
            * gamma
            * (1.0 - sin_theta**2)
            * sin_theta
            * phi_dot
            * theta_dot
            - gamma**2 * cos_theta * sin_theta * theta_dot**2
            + delta * (alpha + beta * sin_theta**2) * sin_theta
            - gamma * cos_theta * torque
        )

        return np.array([phi_dot, phi_ddot, theta_dot, theta_ddot])

    def _rk4(self, state: np.ndarray, torque: float) -> np.ndarray:
        """Advance the continuous model by one sample using RK4."""
        h = self.dt
        k1 = self.dynamics(state, torque)
        k2 = self.dynamics(state + 0.5 * h * k1, torque)
        k3 = self.dynamics(state + 0.5 * h * k2, torque)
        k4 = self.dynamics(state + h * k3, torque)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, object]]:
        """Apply one discrete action and return a Gym-like transition tuple."""
        if not 0 <= int(action) < self.number_of_actions:
            raise ValueError(f"action must be in [0, {self.number_of_actions - 1}]")

        torque = float(self.torque_values[int(action)])
        next_state = self._rk4(self.state, torque)

        if self.process_noise_std > 0.0:
            next_state += self.rng.normal(0.0, self.process_noise_std, size=4)

        self.state = next_state
        self.steps += 1

        theta = float(next_state[2])
        phi = float(next_state[0])
        theta_wrapped = wrap_to_pi(theta)

        # Leaving the upright region is NOT a termination condition: a swing-up
        # episode must pass through every angle.  Only the arm limit is unsafe.
        if abs(theta_wrapped) <= self.angle_limit:
            self.upright_steps += 1
        else:
            self.upright_steps = 0

        unsafe = bool(abs(phi) > self.arm_angle_limit)
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
        }
        return self.state.copy(), reward, terminated, truncated, info


def discretize_state(
    state: np.ndarray,
    bins: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[int, int, int, int]:
    """Convert a continuous state into a tuple suitable for a Q table.

    Each array in ``bins`` contains internal bin boundaries.  For example,
    four boundaries produce five discrete intervals.
    """
    if len(bins) != 4:
        raise ValueError("bins must contain one boundary array per state variable")
    return tuple(
        int(np.digitize(value, boundaries))
        for value, boundaries in zip(np.asarray(state, dtype=float), bins)
    )

