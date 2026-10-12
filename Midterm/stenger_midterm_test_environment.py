import contextlib
import io
import subprocess
import sys

import numpy as np

from stenger_midterm_agent import BALANCE, HERE, HOLD_STEPS, SWINGUP, TS, UPRIGHT_RAD, Episode, wrap_to_pi
from furuta_env import FurutaEnv

RESULTS = HERE / "results"
SEED = 7


def main():
    print("=== example_run.py (supplied, unmodified) ===")
    out = subprocess.run([sys.executable, "example_run.py"], cwd=HERE / "futura_env",
                         capture_output=True, text=True, check=True)
    print(out.stdout.strip())

    env = FurutaEnv(seed=SEED)
    print("\n=== supplied defaults read from FurutaEnv ===")
    print(f"dt = {env.dt} s, max_steps = {env.max_steps} ({env.max_steps * env.dt:.0f} s)")
    print(f"torque_values = {env.torque_values.tolist()} N m")
    print(f"angle_limit = {np.rad2deg(env.angle_limit):.1f} deg (raw theta), arm_angle_limit = {env.arm_angle_limit:.4f} rad")
    print("built-in reward = 1 - (theta/angle_limit)^2 - 0.001 u^2, and -10 on termination")

    state, info = env.reset()
    print("\n=== state components after reset() ===")
    for name, value in zip(["phi (arm angle, rad)", "phi_dot (arm rate, rad/s)",
                            "theta (pendulum angle, rad)", "theta_dot (pendulum rate, rad/s)"], state):
        print(f"  {name:34s} = {value:+.5f}")
    print(f"  info from reset() = {info}")

    print("\n=== every legal action, from the same state ===")
    for a in range(env.number_of_actions):
        env.reset(initial_state=state)
        s2, r, term, trunc, info = env.step(a)
        print(f"  action {a}: torque reported = {info['torque']:+.3f} N m, reward = {r:.6f}, "
              f"terminated = {term}, truncated = {trunc}")
    try:
        env.step(3)
    except ValueError as err:
        print(f"  action 3 rejected: {err}")

    print("\n=== swing-up configuration (constructor arguments only) ===")
    sw_env = SWINGUP.make_env(SEED)
    rng = np.random.default_rng(SEED)
    ep = Episode(SWINGUP, sw_env, rng)
    s0 = ep.state.copy()
    print(f"angle_limit = {sw_env.angle_limit}, arm_angle_limit = {sw_env.arm_angle_limit:.4f}")
    print(f"downward start: state = {np.round(s0, 5).tolist()}, wrapped theta = {np.rad2deg(wrap_to_pi(s0[2])):+.2f} deg")
    s1, r, terminal, done = ep.sim_step(2)
    print(f"one continuing transition with A=2: S' = {np.round(s1, 5).tolist()}, "
          f"R = {r:.5f}, terminal = {terminal}, done = {done}")

    print("\n=== episode outcomes ===")
    print(f"success     : |wrapped theta| <= {np.rad2deg(UPRIGHT_RAD):.0f} deg for {HOLD_STEPS} consecutive steps "
          f"({HOLD_STEPS * TS:.0f} s); counted by the agent loop, demonstrated by the evaluation episodes")
    ep = Episode(SWINGUP, SWINGUP.make_env(SEED), np.random.default_rng(SEED))
    done = False
    while not done:
        _, r, terminal, done = ep.sim_step(2)
    print(f"unsafe      : constant +torque from downward -> outcome '{ep.log.outcome}' after {ep.log.steps} steps, "
          f"|phi| = {abs(ep.env.state[0]):.3f} > pi, R = {r}")
    short = FurutaEnv(max_steps=50, angle_limit=np.inf)
    short.reset(initial_state=[0.0, 0.0, np.pi, 0.0])
    flags = [short.step(1)[2:4] for _ in range(50)]
    print(f"time limit  : zero torque, max_steps=50 -> (terminated, truncated) at step 50 = {flags[-1]}")
    ep = Episode(BALANCE, BALANCE.make_env(SEED), np.random.default_rng(SEED))
    done = False
    while not done:
        _, r, terminal, done = ep.sim_step(1)
    print(f"balance task: zero torque from near upright -> outcome '{ep.log.outcome}' after {ep.log.steps} steps "
          f"({ep.log.steps * TS:.2f} s), R = {r}")

    print("\n=== who chooses what ===")
    print("agent      : action A_t (index 0/1/2), discretization, Q table, epsilon, alpha, gamma, swing-up reward")
    print("environment: next state S_{t+1}, applied torque, built-in reward, terminated (limits), truncated (max_steps)")


if __name__ == "__main__":
    with io.StringIO() as buffer:
        with contextlib.redirect_stdout(buffer):
            main()
        report = buffer.getvalue()
    print(report, end="")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "part1_environment.txt").write_text(report, encoding="utf-8")
