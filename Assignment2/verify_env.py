"""Part 1: verify reset()/step() behavior and episode outcomes."""

import numpy as np

from furuta_env import FurutaEnv, wrap_to_pi
from io_utils import save_json


def run_until_done(env, action, limit):
    for _ in range(limit):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {"steps": env.steps, "terminated": terminated,
                    "truncated": truncated, "outcome": info["outcome"]}
    return {"steps": env.steps, "terminated": False, "truncated": False, "outcome": None}


def counter_reset_scenario(inside_steps=300):
    """Upright counter must reset on leaving the region; success needs 500 steps after re-entry."""
    e = FurutaEnv()
    e.reset(initial_state=np.zeros(4))
    for _ in range(inside_steps):
        e.step(1)
    before = e.upright_steps
    e.state = np.array([0.0, 0.0, 0.5, 0.0])      # force |theta| = 0.5 rad > 20 deg
    _, _, exit_term, _, info = e.step(1)
    after_exit = e.upright_steps
    e.state = np.zeros(4)                          # re-enter the region
    steps_after_reentry, outcome = 0, None
    for _ in range(2000):
        _, _, term, trunc, info = e.step(1)
        steps_after_reentry += 1
        if term or trunc:
            outcome = info["outcome"]
            break
    return {"inside_steps_before_exit": inside_steps, "upright_steps_before_exit": before,
            "upright_steps_after_exit": after_exit, "terminated_on_exit": exit_term,
            "steps_after_reentry_until_done": steps_after_reentry, "outcome": outcome}


def main():
    env = FurutaEnv(seed=0)
    s0, _ = env.reset(seed=0)
    out = {
        "initial_state": s0,
        "initial_theta_wrapped": wrap_to_pi(s0[2]),
        "dt": env.dt,
        "max_steps": env.max_steps,
        "success_steps": env.success_steps,
        "actions": [{"index": i, "torque": float(t)} for i, t in enumerate(env.torque_values)],
    }
    s1, r, terminated, truncated, info = env.step(2)
    out["transition"] = {"S_t": s0, "A_t": 2, "R_t1": r, "S_t1": s1}
    out["outside_upright"] = {
        "theta_wrapped_deg": float(np.rad2deg(info["theta_wrapped"])),
        "terminated": terminated,
        "truncated": truncated,
    }

    outcomes = {}
    e = FurutaEnv()
    e.reset(initial_state=np.zeros(4))
    outcomes["success"] = run_until_done(e, 1, 2000)
    e = FurutaEnv()
    e.reset(initial_state=np.array([3.1, 5.0, np.pi, 0.0]))
    outcomes["unsafe"] = run_until_done(e, 1, 2000)
    e = FurutaEnv(max_steps=50, seed=0)
    e.reset()
    outcomes["timeout"] = run_until_done(e, 1, 2000)
    e = FurutaEnv(seed=0)
    e.reset(seed=0)
    outcomes["zero_torque_downward_full_episode"] = run_until_done(e, 1, 10_000)
    out["outcomes"] = outcomes
    out["upright_counter_reset"] = counter_reset_scenario()

    save_json("verify_env", out)
    return out


if __name__ == "__main__":
    result = main()
    for k, v in result.items():
        print(f"{k}: {v}")
