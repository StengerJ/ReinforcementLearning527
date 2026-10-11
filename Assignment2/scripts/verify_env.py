"""Part 1: verify reset()/step() behavior and episode outcomes.

Reports BOTH the unmodified supplied ``FurutaEnv`` (a balancing task: upright
start, terminates at |theta| > 20 deg) and our ``SwingUpTask`` wrapper (swing-up
episodes), which documents why the wrapper is needed.
Run from the project root:  python -m scripts.verify_env
"""

import numpy as np

import hw2  # noqa: F401  (puts starter/ on sys.path)
from furuta_env import FurutaEnv
from hw2.io_utils import save_json
from hw2.task import SwingUpTask, wrap_to_pi


def run_until_done(env, action, limit):
    for _ in range(limit):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {"steps": env.steps, "terminated": terminated,
                    "truncated": truncated, "outcome": info["outcome"]}
    return {"steps": env.steps, "terminated": False, "truncated": False, "outcome": None}


def counter_reset_scenario(inside_steps=300):
    """Upright counter must reset on leaving the region; success needs 500 steps after re-entry."""
    e = SwingUpTask()
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


def raw_env_report():
    """What the unmodified supplied FurutaEnv does (balancing task)."""
    env = FurutaEnv(seed=0)
    s0, _ = env.reset(seed=0)
    s1, r, terminated, truncated, info = env.step(2)
    raw = {
        "initial_state": s0,
        "actions": [{"index": i, "torque": float(t)} for i, t in enumerate(env.torque_values)],
        "transition": {"S_t": s0, "A_t": 2, "R_t1": r, "S_t1": s1,
                       "terminated": terminated, "truncated": truncated,
                       "torque": info["torque"]},
        "angle_limit_deg": float(np.rad2deg(env.angle_limit)),
    }
    # Downward start: |theta| ~ pi > 20 deg, so the supplied env terminates on step 1.
    env.reset(initial_state=np.array([0.0, 0.0, np.pi, 0.0]))
    _, r_dn, term_dn, trunc_dn, info_dn = env.step(1)
    raw["downward_start"] = {"steps": env.steps, "terminated": term_dn, "truncated": trunc_dn,
                             "reward": r_dn, "theta_deg": float(np.rad2deg(info_dn["theta"]))}
    return raw


def main():
    env = SwingUpTask(seed=0)
    s0, _ = env.reset(seed=0)
    out = {
        "raw_env": raw_env_report(),
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
        "env_terminated": info["env_terminated"],
    }

    outcomes = {}
    e = SwingUpTask()
    e.reset(initial_state=np.zeros(4))
    outcomes["success"] = run_until_done(e, 1, 2000)
    e = SwingUpTask()
    e.reset(initial_state=np.array([3.1, 5.0, np.pi, 0.0]))
    outcomes["unsafe"] = run_until_done(e, 1, 2000)
    e = SwingUpTask(max_steps=50, seed=0)
    e.reset()
    outcomes["timeout"] = run_until_done(e, 1, 2000)
    e = SwingUpTask(seed=0)
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
