import argparse
import contextlib
import io
import time
from dataclasses import asdict, replace
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stenger_midterm_agent import (
    BALANCE, BALANCE_AGENT, HERE, HOLD_STEPS, SWINGUP, SWINGUP_AGENT, TS, Discretizer, Hyper, evaluate,
    evaluate_combined, qlearning_target, sarsa_target, train, wrap_to_pi)

RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"

METHODS = ("sarsa", "qlearning")
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = tuple(range(10_000, 10_020))
MA_WINDOW = 100

EXPERIMENTS = {
    "balance": (BALANCE, Hyper(episodes=2000, q_init=100.0)),
    "swingup": (SWINGUP, Hyper(episodes=6000, q_init=0.0)),
    "balance_agent": (BALANCE_AGENT, Hyper(episodes=4000, q_init=100.0)),
    "swingup_agent": (SWINGUP_AGENT, Hyper(episodes=4000, q_init=0.0)),
}
ALL_NAMES = ["balance", "swingup", "swingup_comparison", "balance_agent", "swingup_agent"]
COMPARISON = ("swingup_comparison", "swingup", {"upright_start_fraction": 0.5})

LABEL = {"sarsa": "SARSA", "qlearning": "Q-learning"}
COLOR = {"sarsa": "tab:blue", "qlearning": "tab:red"}


def experiment(name):
    if name == COMPARISON[0]:
        task, hyper = EXPERIMENTS[COMPARISON[1]]
        return replace(task, **COMPARISON[2]), hyper
    return EXPERIMENTS[name]


def run_job(job):
    name, method, seed, episodes = job
    task, hyper = experiment(name)
    hyper = replace(hyper, episodes=episodes or hyper.episodes)
    t0 = time.time()
    q, visits, hist = train(method, task, hyper, seed, progress=max(1, hyper.episodes // 10))
    np.savez_compressed(RESULTS / f"{name}_{method}_seed{seed}.npz", q=q, visits=visits, **hist)
    return name, method, seed, time.time() - t0


def load(name, method, seed):
    return np.load(RESULTS / f"{name}_{method}_seed{seed}.npz")


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - w, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def eval_stats(logs):
    returns = np.array([e.ret for e in logs])
    first = [e.first_upright_step * TS for e in logs if e.first_upright_step >= 0]
    return {
        "success_rate": float(np.mean([e.max_hold >= HOLD_STEPS for e in logs])),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "reached_upright": len(first),
        "never_upright": len(logs) - len(first),
        "time_to_upright_mean_s": float(np.mean(first)) if first else float("nan"),
        "duration_mean_s": float(np.mean([e.steps * TS for e in logs])),
        "longest_hold_mean_s": float(np.mean([e.max_hold * TS for e in logs])),
        "closest_approach_mean_deg": float(np.mean([e.closest_deg for e in logs])),
        "outcomes": {k: sum(e.outcome == k for e in logs) for k in ("success", "unsafe", "fell", "time-limit")},
    }


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def plot_training(name, task, hyper):
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True)
    for method in METHODS:
        for k, seed in enumerate(TRAIN_SEEDS):
            d = load(name, method, seed)
            keep = d["downward_start"] > 0.5 if task.swing_up else np.ones(len(d["returns"]), dtype=bool)
            episodes = np.flatnonzero(keep) + 1
            lab = f"{LABEL[method]}" if k == 0 else None
            axes[0].plot(episodes, d["returns"][keep], color=COLOR[method], alpha=0.05, lw=0.5)
            axes[0].plot(episodes, moving_average(d["returns"][keep], MA_WINDOW), color=COLOR[method], lw=1.2,
                         alpha=0.9, ls=("-", "--", ":")[k], label=lab)
            axes[1].plot(episodes, moving_average(d["durations"][keep], MA_WINDOW), color=COLOR[method], lw=1.2,
                         ls=("-", "--", ":")[k], label=lab)
            axes[2].plot(episodes, 100 * moving_average(d["success"][keep], MA_WINDOW), color=COLOR[method],
                         lw=1.2, ls=("-", "--", ":")[k], label=lab)
    which = "downward-start episodes only, " if task.swing_up else ""
    axes[0].set_title(f"{name}: training ({which}faint: raw, lines: {MA_WINDOW}-ep. moving avg.;\n"
                      f"seeds {TRAIN_SEEDS} = solid/dashed/dotted)", fontsize=9)
    axes[0].set(ylabel="Episode return")
    axes[1].set(ylabel="Duration (s)")
    axes[2].set(ylabel="5 s success (%)", xlabel="Episode", ylim=(-3, 103))
    for ax in axes:
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_training.png", dpi=150)
    plt.close(fig)


def plot_evaluation(name, task, evals):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, method in zip(axes, METHODS):
        logs = evals[(method, TRAIN_SEEDS[0])]
        for i, e in enumerate(logs[:5]):
            t = np.arange(len(e.theta)) * TS
            th = np.rad2deg(np.asarray(e.theta))
            th[1:][np.abs(np.diff(th)) > 180] = np.nan
            ax.plot(t, th, lw=0.9, label=f"eval seed {EVAL_SEEDS[i]}: {e.outcome}")
        for y in (-20, 20):
            ax.axhline(y, color="gray", ls="--", lw=0.8)
        ax.set(title=f"{LABEL[method]} (train seed {TRAIN_SEEDS[0]}), greedy, Q frozen", xlabel="Time (s)")
        ax.legend(fontsize=6.5, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"wrapped $\tilde\theta$ (deg)")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_eval_theta.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    width = 0.38
    x = np.arange(len(TRAIN_SEEDS))
    for k, method in enumerate(METHODS):
        rates = [100 * eval_stats(evals[(method, s)])["success_rate"] for s in TRAIN_SEEDS]
        bars = ax.bar(x + (k - 0.5) * width, rates, width, color=COLOR[method], label=LABEL[method])
        ax.bar_label(bars, fmt="%.0f", fontsize=8)
    ax.set(xticks=x, xticklabels=[f"train seed {s}" for s in TRAIN_SEEDS], ylim=(0, 110),
           ylabel="Eval. episodes with 5 s hold (%)", title=f"{name}: success over {len(EVAL_SEEDS)} evaluation episodes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_eval_success.png", dpi=150)
    plt.close(fig)


def plot_visits(name, task):
    disc = Discretizer(task.bins)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, method in zip(axes, METHODS):
        visits = sum(load(name, method, s)["visits"].sum(axis=1) for s in TRAIN_SEEDS).reshape(disc.shape)
        grid = visits.sum(axis=(0, 1))
        im = ax.imshow(np.log10(1 + grid.T), origin="lower", aspect="auto", cmap="viridis")
        th = np.round(np.rad2deg(disc.bins[2])).astype(int)
        ax.set_xticks(np.arange(len(th) + 1) - 0.5, [""] + [str(v) for v in th], fontsize=6, rotation=90)
        ax.set_yticks(np.arange(len(disc.bins[3]) + 1) - 0.5, [""] + [f"{v:g}" for v in disc.bins[3]], fontsize=6)
        ax.set(title=f"{LABEL[method]}: log10(1 + visits)", xlabel=r"$\tilde\theta$ bin edges (deg)",
               ylabel=r"$\dot\theta$ bin edges (rad/s)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_visits.png", dpi=150)
    plt.close(fig)


def target_example(name, task, hyper):
    q = load(name, "qlearning", TRAIN_SEEDS[0])["q"]
    disc = Discretizer(task.bins)
    env = task.make_env(123)
    rng = np.random.default_rng(123)
    state = task.reset(env, rng)
    for _ in range(40):
        state = env.step(int(np.argmax(q[disc.index(state)])))[0]
    s = disc.index(state)
    a = int(np.argmax(q[s]))
    next_state, env_r, term, trunc, info = env.step(a)
    r = task.reward(env_r, next_state, info["torque"], False, term)
    s2 = disc.index(next_state)
    a2 = int(np.argmin(q[s2]))
    print(f"  S bins {disc.bins_of(state)}, A = {a}, R = {r:.4f}, S' bins {disc.bins_of(next_state)}")
    print(f"  Q(S',.) = {np.round(q[s2], 3).tolist()}, exploratory A' = {a2}")
    print(f"  SARSA target      R + g*Q(S',A')      = {sarsa_target(r, hyper.gamma, q[s2], a2, term):.4f}")
    print(f"  Q-learning target R + g*max_a Q(S',a) = {qlearning_target(r, hyper.gamma, q[s2], term):.4f}")


def summarize(names):

    out = print
    all_evals = {}
    for name in names:
        task, hyper = experiment(name)
        disc = Discretizer(task.bins)
        out(f"\n==================== {name} ====================")
        out(f"task: {asdict(task) | {'bins': '(see part2_discretization.txt)'}}")
        out(f"hyperparameters: {asdict(hyper)}")
        out(f"states = {disc.n_states}, Q entries = {disc.n_states * 3}, train seeds = {TRAIN_SEEDS}, "
            f"eval seeds = {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}")
        evals = {}
        for method in METHODS:
            for seed in TRAIN_SEEDS:
                d = load(name, method, seed)
                evals[(method, seed)] = evaluate(d["q"], task, EVAL_SEEDS)
                st = eval_stats(evals[(method, seed)])
                n = len(d["returns"])
                tail = np.arange(n) >= n - n // 10
                down = d["downward_start"] > 0.5
                out(f"\n{LABEL[method]} train seed {seed}")
                out(f"  training: last-10% mean return {d['returns'][tail].mean():.1f}, "
                    f"last-10% success {100 * d['success'][tail].mean():.1f}%, "
                    f"final Q-entry coverage {100 * d['coverage'][-1]:.1f}%")
                if task.swing_up and task.upright_start_fraction > 0:
                    out(f"  training, last 10% split by start: downward-start success "
                        f"{100 * d['success'][tail & down].mean():.1f}% ({int((tail & down).sum())} ep.), "
                        f"upright-start success {100 * d['success'][tail & ~down].mean():.1f}% "
                        f"({int((tail & ~down).sum())} ep.)")
                for k, v in st.items():
                    out(f"  eval {k} = {fmt(v)}")
                out("  eval per episode (seed: outcome, first upright s, longest hold s, return):")
                out("    " + "; ".join(
                    f"{es}: {e.outcome}, {'never' if e.first_upright_step < 0 else f'{e.first_upright_step * TS:.2f}'}, "
                    f"{e.max_hold * TS:.2f}, {e.ret:.0f}" for es, e in zip(EVAL_SEEDS, evals[(method, seed)])))
            pooled = [e for s in TRAIN_SEEDS for e in evals[(method, s)]]
            st = eval_stats(pooled)
            out(f"\n{LABEL[method]} pooled over {len(TRAIN_SEEDS)} seeds x {len(EVAL_SEEDS)} eval episodes: "
                f"success {100 * st['success_rate']:.1f}%, return {st['return_mean']:.1f} +- {st['return_std']:.1f}, "
                f"reached upright {st['reached_upright']}/{len(pooled)}, "
                f"time to upright {st['time_to_upright_mean_s']:.2f} s, duration {st['duration_mean_s']:.2f} s")
        all_evals[name] = evals
        plot_training(name, task, hyper)
        plot_evaluation(name, task, evals)
        plot_visits(name, task)
        if task.swing_up and name == "swingup":
            out("\nSARSA vs Q-learning target for one sampled swing-up transition (trained Q-learning table):")
            target_example(name, task, hyper)
    if "swingup" in all_evals and COMPARISON[0] in all_evals:
        plot_comparison(all_evals["swingup"], all_evals[COMPARISON[0]])
    if "balance_agent" in names and "swingup_agent" in names:
        summarize_two_agent()
    return all_evals


def summarize_two_agent():
    out = print
    out("\n==================== two-agent controller (swingup_agent -> balance_agent) ====================")
    out(f"swing-up agent acts until |theta~| <= {SWINGUP_AGENT.handoff[0]} deg and |theta_dot| <= "
        f"{SWINGUP_AGENT.handoff[1]} rad/s, then the balance agent acts every 10 ms; if |theta~| > 20 deg the "
        f"swing-up agent takes over again. Same method and training seed for both agents.")
    combined = {}
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            q_swing = load("swingup_agent", method, seed)["q"]
            q_bal = load("balance_agent", method, seed)["q"]
            logs = evaluate_combined(q_swing, q_bal, SWINGUP_AGENT, BALANCE_AGENT, EVAL_SEEDS)
            combined[(method, seed)] = logs
            st = eval_stats(logs)
            out(f"\n{LABEL[method]} train seed {seed}: success {sum(e.max_hold >= HOLD_STEPS for e in logs)}/"
                f"{len(logs)}, handoffs {sum(e.handoffs for e in logs)}, reached upright {st['reached_upright']}/"
                f"{len(logs)}, longest hold {max(e.max_hold for e in logs) * TS:.2f} s, "
                f"return {st['return_mean']:.1f} +- {st['return_std']:.1f}, outcomes {st['outcomes']}")
            out("  per episode (seed: outcome, first upright s, longest hold s, handoffs): " + "; ".join(
                f"{es}: {e.outcome}, {'never' if e.first_upright_step < 0 else f'{e.first_upright_step * TS:.2f}'}, "
                f"{e.max_hold * TS:.2f}, {e.handoffs}" for es, e in zip(EVAL_SEEDS, logs)))
        pooled = [e for s in TRAIN_SEEDS for e in combined[(method, s)]]
        st = eval_stats(pooled)
        out(f"\n{LABEL[method]} two-agent pooled: success {100 * st['success_rate']:.1f}%, reached upright "
            f"{st['reached_upright']}/{len(pooled)}, handoffs {sum(e.handoffs for e in pooled)}, "
            f"time to upright {st['time_to_upright_mean_s']:.2f} s")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, method in zip(axes, METHODS):
        logs = sorted((e for s in TRAIN_SEEDS for e in combined[(method, s)]), key=lambda e: -e.max_hold)[:4]
        for e in logs:
            th = np.rad2deg(np.asarray(e.theta))
            th[1:][np.abs(np.diff(th)) > 180] = np.nan
            ax.plot(np.arange(len(th)) * TS, th, lw=0.9, label=f"{e.outcome}, hold {e.max_hold * TS:.2f} s")
        for y in (-20, 20):
            ax.axhline(y, color="gray", ls="--", lw=0.8)
        ax.set(title=f"{LABEL[method]}: two-agent controller (4 longest holds)", xlabel="Time (s)")
        ax.legend(fontsize=6.5, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"wrapped $\tilde\theta$ (deg)")
    fig.tight_layout()
    fig.savefig(FIGURES / "two_agent_eval_theta.png", dpi=150)
    plt.close(fig)


def plot_comparison(base, comp):
    metrics = [("success_rate", "5 s success (%)", 100), ("reached", "reached upright (%)", 100),
               ("closest", "closest approach (deg)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    labels = ["baseline\nall starts downward", "comparison\n50% starts from env.reset()"]
    for ax, (key, title, scale) in zip(axes, metrics):
        for k, method in enumerate(METHODS):
            vals = []
            for evals in (base, comp):
                logs = [e for s in TRAIN_SEEDS for e in evals[(method, s)]]
                if key == "success_rate":
                    vals.append(scale * np.mean([e.max_hold >= HOLD_STEPS for e in logs]))
                elif key == "reached":
                    vals.append(scale * np.mean([e.first_upright_step >= 0 for e in logs]))
                else:
                    vals.append(np.mean([e.closest_deg for e in logs]))
            bars = ax.bar(np.arange(2) + (k - 0.5) * 0.38, vals, 0.38, color=COLOR[method], label=LABEL[method])
            ax.bar_label(bars, fmt="%.0f", fontsize=7)
        ax.set_xticks(np.arange(2), labels, fontsize=7)
        ax.set_title(title, fontsize=9)
        if scale == 100:
            ax.set_ylim(0, 105)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Controlled comparison: greedy evaluation, {len(TRAIN_SEEDS)} seeds x {len(EVAL_SEEDS)} episodes",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="experiment names to run")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=None, help="override episode count (smoke tests)")
    args = parser.parse_args()
    names = args.only or ALL_NAMES
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        jobs = [(n, m, s, args.episodes) for n in names for m in METHODS for s in TRAIN_SEEDS]
        with Pool(min(args.workers, len(jobs))) as pool:
            for name, method, seed, sec in pool.imap_unordered(run_job, jobs):
                print(f"finished {name}/{method}/seed{seed} in {sec / 60:.1f} min", flush=True)



    with io.StringIO() as buffer:
        with contextlib.redirect_stdout(buffer):
            summarize(names)
        text = buffer.getvalue()
    print(text, end="")
    out_name = "summary.txt" if names == ALL_NAMES else f"summary_{'_'.join(names)}.txt"
    (RESULTS / out_name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
