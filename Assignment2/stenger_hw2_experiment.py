import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "futura_env"))
from furuta_env import FurutaEnv

from stenger_hw2_audit import td_update
from stenger_hw2_epsilon_greedy import epsilon_greedy
from stenger_hw2_state_discretization import N_ACTIONS, N_STATES, SHAPE, state_index, wrap_to_pi

RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"

CONFIG = {
    "algorithm": "Q-learning",
    "seeds": [0, 1, 2],
    "episodes": 1500,
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 1000,
    "q_init": 0.0,
    "max_steps": 2000,
    "dt": 0.01,
    "eval_seed_offset": 10_000,
    "moving_average_window": 50,
    "upright_deg": 20.0,
    "success_seconds": 5.0,
}


def epsilon_at(episode):
    frac = min(episode / CONFIG["epsilon_decay_episodes"], 1.0)
    return CONFIG["epsilon_start"] + frac * (CONFIG["epsilon_end"] - CONFIG["epsilon_start"])


def longest_upright_run(theta_wrapped):
    best = run = 0
    for inside in np.abs(theta_wrapped) <= np.deg2rad(CONFIG["upright_deg"]):
        run = run + 1 if inside else 0
        best = max(best, run)
    return best


def train(seed):
    env = FurutaEnv(seed=seed, max_steps=CONFIG["max_steps"], dt=CONFIG["dt"])
    rng = np.random.default_rng(seed)
    q = np.full((N_STATES, N_ACTIONS), CONFIG["q_init"])
    visited = np.zeros_like(q, dtype=bool)
    state_visits = np.zeros(N_STATES, dtype=np.int64)
    n = CONFIG["episodes"]
    returns, steps_per_ep, coverage = np.zeros(n), np.zeros(n, dtype=int), np.zeros(n)
    terminated_flags = np.zeros(n, dtype=bool)

    for ep in range(n):
        eps = epsilon_at(ep)
        state, _ = env.reset()
        s = state_index(state)
        total = 0.0
        while True:
            a = epsilon_greedy(q[s], eps, rng)
            next_state, reward, terminated, truncated, _ = env.step(a)
            s2 = state_index(next_state)
            visited[s, a] = True
            state_visits[s] += 1
            td_update(q, s, a, "qlearning", reward, CONFIG["gamma"], CONFIG["alpha"],
                      s2, None, terminated)
            total += reward
            s = s2
            if terminated or truncated:
                break
        returns[ep], steps_per_ep[ep], coverage[ep] = total, env.steps, visited.mean()
        terminated_flags[ep] = terminated
    return q, returns, steps_per_ep, coverage, terminated_flags, state_visits


def evaluate(q, seed):
    q = q.copy()
    q.setflags(write=False)
    env = FurutaEnv(seed=seed, max_steps=CONFIG["max_steps"], dt=CONFIG["dt"])
    rng = np.random.default_rng(seed)
    state, _ = env.reset()
    theta, phi, actions = [wrap_to_pi(state[2])], [state[0]], []
    while True:
        a = epsilon_greedy(q[state_index(state)], 0.0, rng)
        state, _, terminated, truncated, _ = env.step(a)
        theta.append(wrap_to_pi(state[2]))
        phi.append(state[0])
        actions.append(a)
        if terminated or truncated:
            break
    return np.array(theta), np.array(phi), np.array(actions), terminated, truncated


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    lo = np.maximum(np.arange(1, len(x) + 1) - w, 0)
    return (c[1:] - c[lo]) / (np.arange(1, len(x) + 1) - lo)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    dt, w = CONFIG["dt"], CONFIG["moving_average_window"]
    success_steps = int(round(CONFIG["success_seconds"] / dt))
    summary = {"config": CONFIG, "shape": SHAPE, "n_states": N_STATES, "seeds": []}
    runs = {}

    for seed in CONFIG["seeds"]:
        t0 = time.time()
        q, returns, steps, coverage, term, state_visits = train(seed)
        theta, phi, actions, ev_term, ev_trunc = evaluate(q, CONFIG["eval_seed_offset"] + seed)
        run_len = longest_upright_run(theta)
        runs[seed] = (returns, steps, coverage, theta)
        np.savez(RESULTS / f"part5_seed{seed}.npz", q=q, returns=returns, steps=steps,
                 coverage=coverage, terminated=term, state_visits=state_visits,
                 eval_theta=theta, eval_phi=phi, eval_actions=actions)

        visits = state_visits.reshape(SHAPE)
        rec = {
            "seed": seed,
            "runtime_s": round(time.time() - t0, 1),
            "first_100_mean_return": float(returns[:100].mean()),
            "last_100_mean_return": float(returns[-100:].mean()),
            "first_100_mean_duration_s": float(steps[:100].mean() * dt),
            "last_100_mean_duration_s": float(steps[-100:].mean() * dt),
            "max_duration_s": float(steps.max() * dt),
            "episodes_reaching_5s": int((steps >= success_steps).sum()),
            "episodes_truncated_20s": int((~term).sum()),
            "final_visited_fraction": float(coverage[-1]),
            "visited_states": int((state_visits > 0).sum()),
            "eval_steps": int(len(actions)),
            "eval_duration_s": float(len(actions) * dt),
            "eval_terminated": bool(ev_term),
            "eval_truncated": bool(ev_trunc),
            "eval_longest_upright_s": float(run_len * dt),
            "eval_completed_5s": bool(run_len >= success_steps),
            "eval_max_abs_theta_deg": float(np.rad2deg(np.abs(theta).max())),
            "eval_final_phi": float(phi[-1]),
            "eval_action_counts": np.bincount(actions, minlength=N_ACTIONS).tolist(),
            "visits_by_phi_bin": visits.sum(axis=(1, 2, 3)).tolist(),
            "visits_by_phidot_bin": visits.sum(axis=(0, 2, 3)).tolist(),
            "visits_by_theta_bin": visits.sum(axis=(0, 1, 3)).tolist(),
            "visits_by_thetadot_bin": visits.sum(axis=(0, 1, 2)).tolist(),
        }
        summary["seeds"].append(rec)
        print(json.dumps(rec))

    (RESULTS / "part5_summary.json").write_text(json.dumps(summary, indent=2))

    episodes = np.arange(1, CONFIG["episodes"] + 1)
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for seed, (returns, _, _, _) in runs.items():
        ax.plot(episodes, returns, color=colors[seed], alpha=0.18, lw=0.6)
        ax.plot(episodes, moving_average(returns, w), color=colors[seed], lw=1.6,
                label=f"seed {seed} ({w}-ep. moving avg.)")
    ax.set(xlabel="Episode", ylabel="Episode return", title="Training return (faint: raw)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "returns.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for seed, (_, steps, _, _) in runs.items():
        ax.plot(episodes, steps * dt, color=colors[seed], alpha=0.18, lw=0.6)
        ax.plot(episodes, moving_average(steps * dt, w), color=colors[seed], lw=1.6, label=f"seed {seed}")
    ax.axhline(CONFIG["success_seconds"], color="gray", ls="--", lw=1, label="5 s objective")
    ax.set(xlabel="Episode", ylabel="Duration (s)", title="Episode duration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "durations.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for seed, (_, _, coverage, _) in runs.items():
        ax.plot(episodes, 100 * coverage, color=colors[seed], lw=1.6, label=f"seed {seed}")
    ax.set(xlabel="Episode", ylabel="Visited Q entries (%)", title="Fraction of state-action entries visited")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for seed, (_, _, _, theta) in runs.items():
        ax.plot(np.arange(len(theta)) * dt, np.rad2deg(theta), color=colors[seed], lw=1.2,
                label=f"seed {seed}")
    for y in (-CONFIG["upright_deg"], CONFIG["upright_deg"]):
        ax.axhline(y, color="gray", ls="--", lw=1)
    ax.set(xlabel="Time (s)", ylabel=r"$\tilde\theta$ (deg)", title="Greedy evaluation (Q frozen, $\\epsilon=0$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "evaluation_theta.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
