"""HW2 Part 1: verify the supplied environment and its episode logic.

FurutaEnv is imported and used exactly as supplied (no wrapper, no edits).

Run:  python stenger_hw2_test_environment.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "futura_env"))
from furuta_env import FurutaEnv  # noqa: E402

from stenger_hw2_state_discretization import wrap_to_pi  # noqa: E402

RESULTS = HERE / "results"
SEED = 7


def run_until_done(env, policy):
    """Step until terminated or truncated; return (steps, terminated, truncated, last state, last reward)."""
    while True:
        state, reward, terminated, truncated, _ = env.step(policy())
        if terminated or truncated:
            return env.steps, terminated, truncated, state, reward


def main():
    out = {"seed": SEED}

    # 1) One initial state from reset().
    env = FurutaEnv(seed=SEED)
    state, info = env.reset()
    print("Initial state [phi, phi_dot, theta, theta_dot]:", state.tolist())
    print(f"  theta = {np.rad2deg(state[2]):.3f} deg (0 = upright): reset starts NEAR UPRIGHT, at rest")
    print("  dt =", env.dt, "s, max_steps =", env.max_steps, "->", env.dt * env.max_steps, "s")
    out.update(initial_state=state.tolist(), dt=env.dt, max_steps=env.max_steps)

    # 2) Every legal action and its torque (same start state each time).
    out["actions"] = []
    for action in range(env.number_of_actions):
        env.reset(initial_state=state)
        _, _, _, _, info = env.step(action)
        print(f"Action {action}: torque = {info['torque']:+.2f} N m")
        out["actions"].append({"action": action, "torque": info["torque"]})

    # 3) One sampled transition (S_t, A_t, R_{t+1}, S_{t+1}).
    env.reset(initial_state=state)
    action = 2
    next_state, reward, terminated, truncated, info = env.step(action)
    print("Transition: S_t =", state.tolist())
    print("            A_t =", action, f"(torque {info['torque']:+.2f} N m)")
    print("        R_{t+1} =", reward)
    print("        S_{t+1} =", next_state.tolist())
    print("     terminated =", terminated, " truncated =", truncated)
    out["transition"] = {"S": state.tolist(), "A": action, "R": reward, "S_next": next_state.tolist(),
                         "terminated": terminated, "truncated": truncated}

    # 4a) Unsafe/angle termination: zero torque from reset() falls past 20 degrees.
    env = FurutaEnv(seed=SEED)
    env.reset()
    steps, term, trunc, last, rew = run_until_done(env, lambda: 1)
    print(f"Zero torque from reset: ended after {steps} steps ({steps * env.dt:.2f} s), "
          f"terminated={term}, truncated={trunc}, theta={np.rad2deg(last[2]):.2f} deg, reward={rew}")
    out["zero_torque"] = {"steps": steps, "terminated": term, "truncated": trunc,
                          "theta_deg": float(np.rad2deg(last[2])), "reward": rew}

    # 4b) Arm-limit termination: |phi| > pi.
    env.reset(initial_state=[np.pi - 0.001, 1.0, 0.0, 0.0])
    _, rew, term, trunc, info = env.step(1)
    print(f"Arm past pi: phi={info['phi']:.4f}, terminated={term}, truncated={trunc}, reward={rew}")
    out["arm_limit"] = {"phi": info["phi"], "terminated": term, "truncated": trunc, "reward": rew}

    # 4c) Time-limit truncation: the exact upright equilibrium never falls.
    env = FurutaEnv(seed=SEED, max_steps=50)
    env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
    steps, term, trunc, last, rew = run_until_done(env, lambda: 1)
    print(f"Upright equilibrium, max_steps=50: ended after {steps} steps, "
          f"terminated={term}, truncated={trunc}")
    out["truncation"] = {"steps": steps, "terminated": term, "truncated": trunc}

    # 4d) Success: the environment has no success flag; check what info contains.
    print("info keys returned by step():", sorted(info.keys()), "-> no success signal")
    out["info_keys"] = sorted(info.keys())

    # 5) Is being OUTSIDE the upright region terminal?  Start hanging downward.
    env = FurutaEnv(seed=SEED)
    down = np.array([0.0, 0.0, np.pi, 0.0])
    env.reset(initial_state=down)
    nxt, rew, term, trunc, _ = env.step(1)
    print(f"Downward start (theta_wrapped={np.rad2deg(wrap_to_pi(down[2])):.1f} deg): "
          f"terminated={term} after 1 step, reward={rew}")
    # Also: the termination test uses RAW theta, not wrapToPi(theta).
    env.reset(initial_state=[0.0, 0.0, 2 * np.pi, 0.0])
    _, rew2, term2, _, _ = env.step(1)
    print(f"theta = 2*pi (physically upright, wrapped = 0 deg): terminated={term2}, reward={rew2}")
    out["outside_upright"] = {"downward_terminated": term, "downward_reward": rew,
                              "two_pi_terminated": term2, "two_pi_reward": rew2}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "part1_environment.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
