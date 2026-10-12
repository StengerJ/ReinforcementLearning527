"""Build report/figures/hw2_explainer.mp4: a narrated visual walkthrough of HW2.

Visuals are drawn with matplotlib from the saved results/ files and from live
calls to the unmodified FurutaEnv.  The voiceover uses the offline Windows
speech synthesizer (System.Speech, voice "Microsoft Zira Desktop"); every
narration sentence is synthesized separately so on-screen reveals and
subtitles start exactly when their sentence starts.

Requires: Windows, numpy, matplotlib, imageio-ffmpeg; run
stenger_hw2_experiment.py and stenger_hw2_audit.py first.
Run:  python stenger_hw2_make_video.py      (several minutes)
"""

import json
import subprocess
import sys
import tempfile
import textwrap
import wave
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Wedge  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "futura_env"))
from furuta_env import FurutaEnv  # noqa: E402

from stenger_hw2_epsilon_greedy import SEED as EPS_SEED, action_probabilities, epsilon_greedy  # noqa: E402
from stenger_hw2_state_discretization import BINS, wrap_to_pi  # noqa: E402

RESULTS = HERE / "results"
OUT = HERE / "report" / "figures" / "hw2_explainer.mp4"
VOICE = "Microsoft Zira Desktop"
W, H, FPS = 1280, 720, 24
RATE = 22050
LEAD, GAP, TAIL = 0.5, 0.3, 0.9  # seconds of silence around sentences

BG, PANEL, FG, MUTED = "#0f172a", "#1e293b", "#e2e8f0", "#94a3b8"
ACC, OK, WARN, GOLD, VIOLET = "#38bdf8", "#4ade80", "#f87171", "#fbbf24", "#a78bfa"
SEED_COLORS = [ACC, GOLD, OK]

AUDIT = json.loads((RESULTS / "part4_audit.json").read_text())
SUMMARY = json.loads((RESULTS / "part5_summary.json").read_text())
RUNS = {s: np.load(RESULTS / f"part5_seed{s}.npz") for s in (0, 1, 2)}


# ----------------------------------------------------------------- helpers
def fade(t, t0, d=0.45):
    return float(np.clip((t - t0) / d, 0.0, 1.0))


def text(fig, x, y, s, t=None, t0=0.0, size=18, color=FG, **kw):
    a = 1.0 if t is None else fade(t, t0)
    if a > 0:
        fig.text(x, y, s, size=size, color=color, alpha=a, **kw)


def header(fig, part, title):
    fig.text(0.04, 0.925, part, size=15, color=ACC, weight="bold")
    fig.text(0.04, 0.865, title, size=27, color=FG, weight="bold")
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.845, 0.845], color=PANEL, lw=2))


def panel(fig, x, y, w, h, alpha=1.0):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                  transform=fig.transFigure, facecolor=PANEL, edgecolor="none",
                                  alpha=alpha))


def style_ax(ax):
    ax.set_facecolor(PANEL)
    for side in ax.spines.values():
        side.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def pendulum(ax, theta, color=ACC, region=True, swing_path=False, cross=False):
    """Side view; theta measured from upright, positive to the right."""
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    if region:
        ax.add_patch(Wedge((0, 0), 1.25, 70, 110, color=OK, alpha=0.22))
        ax.text(0, 1.31, r"$|\tilde\theta|\leq 20^\circ$", color=OK, ha="center", size=11)
    if swing_path:
        for sign in (1, -1):
            ang = np.deg2rad(np.linspace(20, 180, 60)) * sign
            ax.plot(1.12 * np.sin(ang), 1.12 * np.cos(ang), color=WARN, lw=2.5, ls=(0, (4, 3)))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ls=":", color=MUTED, lw=1))
    x, y = np.sin(theta), np.cos(theta)
    ax.plot([0, x], [0, y], color=FG, lw=5, solid_capstyle="round")
    ax.add_patch(Circle((x, y), 0.1, color=color, zorder=5))
    ax.add_patch(Circle((0, 0), 0.06, color=MUTED, zorder=6))
    if cross:
        ax.plot([-0.7, 0.7], [-0.7, 0.7], color=WARN, lw=8, alpha=0.85)
        ax.plot([-0.7, 0.7], [0.7, -0.7], color=WARN, lw=8, alpha=0.85)


def arm_dial(ax, phi):
    """Top view of the horizontal arm; red shading marks the |phi| > pi limit."""
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color=MUTED, lw=1))
    ax.add_patch(Wedge((0, 0), 1.0, 265, 275, color=WARN, alpha=0.6))
    ax.plot([0, np.sin(phi)], [0, np.cos(phi)], color=GOLD, lw=4)
    ax.add_patch(Circle((0, 0), 0.07, color=MUTED))
    ax.text(0, -1.28, rf"arm (top view)  $\phi={phi:+.2f}$ rad", color=MUTED, ha="center", size=10)


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    lo = np.maximum(np.arange(1, len(x) + 1) - w, 0)
    return (c[1:] - c[lo]) / (np.arange(1, len(x) + 1) - lo)


def progress(t, t0, t1):
    return float(np.clip((t - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))


# ----------------------------------------------------------- precomputed data
def zero_torque_rollout():
    env = FurutaEnv(seed=7)
    s, _ = env.reset()
    thetas = [s[2]]
    while True:
        s, r, term, trunc, _ = env.step(1)
        thetas.append(s[2])
        if term or trunc:
            return np.array(thetas), r


ZERO_THETA, ZERO_R = zero_torque_rollout()


def eps_samples(epsilon):
    rng = np.random.default_rng(EPS_SEED)  # identical to stenger_hw2_epsilon_greedy.py
    q = np.array([1.0, 2.0, 2.0])
    return np.array([epsilon_greedy(q, epsilon, rng) for _ in range(10_000)])


EPS_DRAWS = {0.2: eps_samples(0.2), 0.0: eps_samples(0.0)}


# ------------------------------------------------------------------ scenes
def sc_title(fig, t, c, e):
    text(fig, 0.5, 0.80, "Building & Verifying a Tabular RL Agent", t, 0, 34, FG,
         ha="center", weight="bold")
    text(fig, 0.5, 0.735, "COSC 527 Reinforcement Learning  ·  Homework 2  ·  Furuta pendulum",
         t, 0.3, 17, ACC, ha="center")
    text(fig, 0.5, 0.69, "Joshua Stenger", t, 0.6, 15, MUTED, ha="center")
    pendulum(fig.add_axes([0.40, 0.30, 0.20, 0.36]), 0.45 * np.sin(1.4 * t))
    parts = ["1  Environment", "2  Discretization", "3  ε-greedy", "4  Update audit",
             "5  Experiment", "6  Reflection"]
    for i, p in enumerate(parts):
        text(fig, 0.11 + i * 0.156, 0.22, p, t, c[1] + 0.9 * i, 15, FG, ha="center")


def sc_setup(fig, t, c, e):
    header(fig, "Problem setting", "The Furuta pendulum as an RL problem")
    pendulum(fig.add_axes([0.04, 0.14, 0.40, 0.66]), 0.6 * np.sin(1.1 * t))
    rows = [(r"$\phi$", "arm angle (rad)"), (r"$\dot\phi$", "arm angular velocity (rad/s)"),
            (r"$\theta$", "pendulum angle (rad)"), (r"$\dot\theta$", "pendulum angular velocity (rad/s)")]
    text(fig, 0.48, 0.76, r"State  $x = [\phi,\ \dot\phi,\ \theta,\ \dot\theta]^T$", t, c[1], 22, FG)
    for i, (sym, desc) in enumerate(rows):
        text(fig, 0.50, 0.69 - 0.055 * i, sym, t, c[1] + 0.6 * (i + 1), 20, ACC)
        text(fig, 0.56, 0.69 - 0.055 * i, desc, t, c[1] + 0.6 * (i + 1), 17, FG)
    text(fig, 0.48, 0.44, r"$\tilde\theta=\mathrm{wrapToPi}(\theta)$:  0 = upright,  $\pm\pi$ = downward",
         t, c[2], 17, OK)
    for i, (a, tau) in enumerate([(0, "−0.02"), (1, "0"), (2, "+0.02")]):
        a_ = fade(t, c[3] + 0.5 * i)
        if a_ > 0:
            panel(fig, 0.49 + i * 0.15, 0.22, 0.13, 0.13, alpha=a_)
            fig.text(0.555 + i * 0.15, 0.305, f"action {a}", ha="center", size=14, color=MUTED, alpha=a_)
            fig.text(0.555 + i * 0.15, 0.245, f"{tau} N·m", ha="center", size=19, color=GOLD,
                     alpha=a_, weight="bold")


def sc_env(fig, t, c, e):
    header(fig, "Part 1 · 15 pts", "Verify the environment — used exactly as supplied")
    k = int(progress(t, c[2], e[2]) * (len(ZERO_THETA) - 1))
    th = ZERO_THETA[: k + 1]
    pendulum(fig.add_axes([0.03, 0.17, 0.33, 0.62]), th[-1], color=WARN if k == len(ZERO_THETA) - 1 else ACC)
    ax = fig.add_axes([0.42, 0.50, 0.53, 0.30])
    style_ax(ax)
    ax.axhspan(-20, 20, color=OK, alpha=0.12)
    ax.plot(np.arange(len(th)) * 0.01, np.rad2deg(th), color=ACC, lw=2.5)
    ax.set(xlim=(0, 0.6), ylim=(-5, 25), xlabel="time (s)", ylabel=r"$\theta$ (deg)")
    ax.set_title("zero torque from reset()", color=MUTED, size=12)
    if k == len(ZERO_THETA) - 1:
        ax.text(0.30, 21.5, f"terminated = True,  R = {ZERO_R:.0f}", color=WARN, size=14, weight="bold")
    lines = [(c[1], ACC, f"reset():  θ₀ = {np.rad2deg(ZERO_THETA[0]):.2f}°  → starts near UPRIGHT, at rest"),
             (c[2], WARN, f"|θ| > 20° after {(len(ZERO_THETA) - 1) * 0.01:.2f} s  → terminated, reward −10"),
             (c[3], GOLD, "time limit (max_steps)  → truncated = True, terminated = False"),
             (c[3] + 2.0, MUTED, "info = {torque, theta, phi}  → no success flag")]
    for i, (t0, col, s) in enumerate(lines):
        text(fig, 0.42, 0.38 - 0.06 * i, s, t, t0, 16, col)


def sc_finding(fig, t, c, e):
    header(fig, "Part 1 · key finding", "Leaving the upright region DOES terminate")
    pendulum(fig.add_axes([0.03, 0.15, 0.36, 0.66]), np.pi, color=WARN,
             swing_path=t >= c[3], cross=t >= c[1])
    lines = [(c[0], FG, "Assignment: leaving the upright region\nmust NOT terminate"),
             (c[1], WARN, "Supplied env: start at θ = π\n→ terminated after 1 step, R = −10"),
             (c[2], WARN, "θ = 2π (physically upright) also terminates\n→ raw θ is checked, not wrapToPi(θ)"),
             (c[3], GOLD, "Swing-up must cross every angle 180° → 20°\n→ swing-up is impossible"),
             (c[4], OK, "So Part 5 studies the BALANCING task\nthe supplied env implements")]
    for i, (t0, col, s) in enumerate(lines):
        text(fig, 0.42, 0.70 - 0.125 * i, s, t, t0, 17, col, va="center", linespacing=1.3)


def sc_disc(fig, t, c, e):
    header(fig, "Part 2 · 20 pts", "Discretize the continuous state")
    ax = fig.add_axes([0.02, 0.14, 0.40, 0.68])
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    edges = np.concatenate([[-180.0], np.rad2deg(BINS[2]), [180.0]])
    a = fade(t, c[1])
    palette = [VIOLET, ACC, OK, GOLD]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        col = MUTED if i in (0, 9) else palette[i % 4]
        ax.add_patch(Wedge((0, 0), 1.2, 90 - hi, 90 - lo, width=0.45 if i not in (0, 9) else 0.25,
                           color=col, alpha=0.75 * a, ec=BG, lw=1.5))
    if a > 0:
        ax.text(0, 1.30, "bins 1–8: inside ±20°", color=FG, ha="center", size=12, alpha=a)
        ax.text(-1.05, -0.95, "bin 0\n(θ̃ < −20°)", color=MUTED, ha="center", size=11, alpha=a)
        ax.text(1.05, -0.95, "bin 9\n(θ̃ > +20°)", color=MUTED, ha="center", size=11, alpha=a)
    tests = [(0.0, "upright → bin 5"), (np.pi - 0.01, "near +π → bin 9"),
             (-np.pi + 0.01, "near −π → bin 0"), (np.deg2rad(5.0), "on edge +5° → bin 7")]
    if t >= c[4] + 2.5:
        j = min(int((t - c[4] - 2.5) / 1.6), len(tests) - 1)
        th, lab = tests[j]
        ax.plot([0, 0.95 * np.sin(th)], [0, 0.95 * np.cos(th)], color=WARN, lw=3)
        ax.add_patch(Circle((0.95 * np.sin(th), 0.95 * np.cos(th)), 0.07, color=WARN))
        ax.text(0, 0, lab, color=WARN, ha="center", va="center", size=13, weight="bold")
    rows = [("variable", "range", "bins"), ("θ̃", "±20° (+ 2 outer)", "10"),
            ("φ", "±1 rad", "6"), ("φ̇", "±3 rad/s", "6"), ("θ̇", "±3 rad/s", "8")]
    for i, (v, r, n) in enumerate(rows):
        t0 = c[1] if i <= 1 else c[2] + 0.5 * (i - 2)
        col = MUTED if i == 0 else FG
        text(fig, 0.47, 0.74 - 0.055 * i, v, t, t0, 17, col)
        text(fig, 0.58, 0.74 - 0.055 * i, r, t, t0, 17, col)
        text(fig, 0.80, 0.74 - 0.055 * i, n, t, t0, 17, GOLD if i else col)
    text(fig, 0.47, 0.41, r"$|\mathcal{S}| = 6\cdot6\cdot10\cdot8 = 2880$     $|Q| = 2880\cdot3 = 8640$",
         t, c[3], 20, ACC)
    text(fig, 0.47, 0.33, "out of range  → clamped into the end bin (always a valid index)", t, c[4], 15, FG)
    text(fig, 0.47, 0.28, "exactly on an edge  → upper bin   (φ̇ = 1.5 → bin 4)", t, c[4] + 1.0, 15, FG)
    text(fig, 0.47, 0.23, "10 deliberate test states, all indices in [0, 2879]", t, c[4] + 2.0, 15, OK)


def eps_bars(fig, rect, draws, t0, t1, t, epsilon, show_expected):
    ax = fig.add_axes(rect)
    style_ax(ax)
    n = int(progress(t, t0, t1) * len(draws))
    counts = np.bincount(draws[:n], minlength=3)
    freq = counts / max(n, 1)
    ax.bar([0, 1, 2], freq, color=[MUTED, ACC, ACC], width=0.6)
    if show_expected:
        exp = action_probabilities(np.array([1.0, 2.0, 2.0]), epsilon)
        for i, p in enumerate(exp):
            ax.plot([i - 0.38, i + 0.38], [p, p], color=GOLD, lw=2.5, ls="--")
            ax.text(i, p + 0.03, f"{p:.3f}", color=GOLD, ha="center", size=11)
    for i, f in enumerate(freq):
        if n and f > 0.03:
            ax.text(i, 0.02, f"{f:.4f}", color=BG, ha="center", size=11, weight="bold")
        elif n:
            ax.text(i, 0.12, "never chosen" if f == 0 else f"{f:.4f}", color=FG, ha="center", size=11)
    ax.set(ylim=(0, 0.62), xticks=[0, 1, 2], xticklabels=["a=0\nQ=1", "a=1\nQ=2", "a=2\nQ=2"])
    ax.set_title(f"ε = {epsilon}   samples: {n:,}", color=FG, size=13)


def sc_eps(fig, t, c, e):
    header(fig, "Part 3 · 15 pts", "Verify the ε-greedy behaviour policy")
    text(fig, 0.04, 0.77, "with prob. ε: uniform over ALL actions     otherwise: uniform over tied greedy actions",
         t, c[1], 16, FG)
    text(fig, 0.04, 0.72, r"$P(a)=\epsilon/3+(1-\epsilon)/|\mathcal{G}|\cdot\mathbb{1}[a\in\mathcal{G}]$"
         f"     seed = {EPS_SEED}", t, c[1] + 2.5, 17, ACC)
    eps_bars(fig, [0.07, 0.24, 0.38, 0.40], EPS_DRAWS[0.2], c[3], e[3], t, 0.2, t >= c[2])
    if t >= c[4]:
        eps_bars(fig, [0.56, 0.24, 0.38, 0.40], EPS_DRAWS[0.0], c[4], e[4], t, 0.0, True)


def sc_audit(fig, t, c, e):
    header(fig, "Part 4 · 20 pts", "Audit one SARSA and one Q-learning update")
    tr, term = AUDIT["transitions"]
    g, al = AUDIT["gamma"], AUDIT["alpha"]
    if t >= c[1]:
        ax = fig.add_axes([0.05, 0.25, 0.30, 0.44])
        style_ax(ax)
        vals = tr["Q_old_next"]
        cols = [MUTED, MUTED, MUTED]
        if t >= c[2]:
            cols[tr["A_next"]] = GOLD
        if t >= c[3]:
            cols[int(np.argmax(vals))] = ACC
        ax.bar([0, 1, 2], vals, color=cols, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.08, f"{v:.2f}", color=FG, ha="center", size=13)
        ax.set(ylim=(0, 5), xticks=[0, 1, 2], xticklabels=["a=0", "a=1", "a=2"])
        ax.set_title(r"$Q_{old}(S_{t+1},\cdot)$     $A_{t+1}=2$ (exploratory)", color=FG, size=13)
        if t >= c[2]:
            ax.text(2, 3.0, "SARSA\nuses A′", color=GOLD, ha="center", size=11)
        if t >= c[3]:
            ax.text(1, 4.55, "Q-learning\nuses max", color=ACC, ha="center", size=11)
    text(fig, 0.05, 0.75, f"R = {tr['R']:.7f}    γ = {g}    α = {al}    Q_old(S,A) = {tr['Q_old_sa']}",
         t, c[1], 15, MUTED)
    eqs = [(c[2], GOLD, rf"$Y^{{SARSA}} = 1.0 + {g}\times{tr['Q_old_next'][tr['A_next']]:.2f} = {tr['Y_sarsa']:.3f}$"),
           (c[3], ACC, rf"$Y^{{Q}} = 1.0 + {g}\times\max_a Q = 1.0 + {g}\times{max(tr['Q_old_next']):.2f} = {tr['Y_q']:.3f}$"),
           (c[4], GOLD, rf"$Q^{{SARSA}}_{{new}} = 4.57 + 0.1\,({tr['Y_sarsa']:.3f}-4.57) = {tr['new_sarsa']:.4f}$"),
           (c[4] + 3.5, ACC, rf"$Q^{{Q}}_{{new}} = 4.57 + 0.1\,({tr['Y_q']:.3f}-4.57) = {tr['new_q']:.4f}$"),
           (c[4] + 6.5, OK, "program output: identical ✓"),
           (c[5] + 2.5, WARN, rf"terminal: $Q(S',\cdot)=[100,200,300]$ ignored, $Y={term['Y_sarsa']:.0f}$ for both")]
    for i, (t0, col, s) in enumerate(eqs):
        text(fig, 0.40, 0.64 - 0.085 * i, s, t, t0, 18, col)


def sc_train(fig, t, c, e):
    header(fig, "Part 5 · 20 pts", "Preliminary Q-learning experiment (3 seeds)")
    cfg = SUMMARY["config"]
    rows = [f"episodes: {cfg['episodes']}", f"α = {cfg['alpha']},  γ = {cfg['gamma']}",
            f"ε: {cfg['epsilon_start']} → {cfg['epsilon_end']} (linear, {cfg['epsilon_decay_episodes']} ep.)",
            "Q initialized to 0", f"max episode: {cfg['max_steps'] * cfg['dt']:.0f} s", "seeds: 0, 1, 2"]
    a = fade(t, c[1])
    if a:
        panel(fig, 0.03, 0.18, 0.27, 0.58, alpha=a)
    for i, r in enumerate(rows):
        text(fig, 0.05, 0.69 - 0.07 * i, r, t, c[1] + 0.8 * i, 15, FG)
    ax = fig.add_axes([0.38, 0.20, 0.57, 0.56])
    style_ax(ax)
    n = cfg["episodes"]
    k = max(int(progress(t, c[2], e[3]) * n), 1)
    for s in (0, 1, 2):
        d = RUNS[s]["steps"] * cfg["dt"]
        ax.plot(np.arange(1, k + 1), d[:k], color=SEED_COLORS[s], alpha=0.15, lw=0.6)
        ax.plot(np.arange(1, k + 1), moving_average(d, 50)[:k], color=SEED_COLORS[s], lw=2.2,
                label=f"seed {s}")
    ax.axhline(5, color=MUTED, ls="--", lw=1)
    ax.set(xlim=(0, n), ylim=(0, 21), xlabel="episode", ylabel="episode duration (s)")
    ax.legend(loc="upper left", fontsize=10, facecolor=PANEL, labelcolor=FG, edgecolor=MUTED)
    ax.set_title("duration (50-episode moving average; faint = raw)", color=MUTED, size=12)
    if t >= c[4]:
        cov = ", ".join(f"{100 * r['final_visited_fraction']:.1f}%" for r in SUMMARY["seeds"])
        text(fig, 0.38, 0.105, f"visited Q entries at the end: {cov}", t, c[4], 16, GOLD)


def sc_eval(fig, t, c, e):
    header(fig, "Part 5 · greedy evaluation", "Q frozen, ε = 0, no updates — seed 2")
    run = RUNS[2]
    th, phi = run["eval_theta"], run["eval_phi"]
    speed = 0.5
    k = int(np.clip((t - c[1]) * speed / 0.01, 0, len(th) - 1))
    fell = k == len(th) - 1
    pendulum(fig.add_axes([0.02, 0.36, 0.30, 0.46]), th[k], color=WARN if fell else ACC)
    arm_dial(fig.add_axes([0.08, 0.15, 0.18, 0.20]), phi[k])
    ax = fig.add_axes([0.40, 0.42, 0.55, 0.36])
    style_ax(ax)
    ax.axhspan(-20, 20, color=OK, alpha=0.12)
    ax.plot(np.arange(k + 1) * 0.01, np.rad2deg(th[: k + 1]), color=ACC, lw=2.2)
    ax.axvline(5, color=GOLD, ls="--", lw=1)
    ax.text(5.05, 15, "5 s goal", color=GOLD, size=11)
    ax.set(xlim=(0, 6), ylim=(-25, 25), xlabel="time (s)", ylabel=r"$\tilde\theta$ (deg)")
    ax.set_title(f"t = {k * 0.01:.2f} s  (half speed)", color=FG, size=13)
    if fell:
        ax.text(0.2, -22, f"fell past 20° at {k * 0.01:.2f} s", color=WARN, size=14, weight="bold")
    for i, r in enumerate(SUMMARY["seeds"]):
        s = (f"seed {r['seed']}:  balanced {r['eval_longest_upright_s']:.2f} s  →  "
             f"{'completed' if r['eval_completed_5s'] else 'did NOT complete'} 5 s")
        text(fig, 0.40, 0.29 - 0.06 * i, s, t, c[3] + 0.8 * i, 16, SEED_COLORS[r["seed"]])


def sc_reflect(fig, t, c, e):
    header(fig, "Part 6 · 10 pts", "Diagnostic reflection")
    r = SUMMARY["seeds"][2]
    v = np.array(r["visits_by_theta_bin"], dtype=float)
    pct = 100 * v / v.sum()
    ax = fig.add_axes([0.05, 0.18, 0.38, 0.52])
    style_ax(ax)
    cols = [GOLD if i in (4, 5) and t >= c[1] else ACC for i in range(10)]
    ax.bar(range(10), pct, color=cols)
    ax.set(xticks=range(10), xlabel=r"$\tilde\theta$ bin", ylabel="% of visits (seed 2)", ylim=(0, 50))
    ax.set_title("where training actually spends its time", color=FG, size=13)
    if t >= c[1]:
        ax.text(4.5, 47, f"bins 4–5 (|θ̃| < 2°): {pct[4] + pct[5]:.0f}%", color=GOLD, ha="center", size=12)
    items = [(c[1], GOLD, "(a) Limiting factor: the discretization",
              "~¾ of visits share two θ̃ bins → different states, one action"),
             (c[2], ACC, "(b) Rarely visited: |θ̇| > 3 rad/s",
              f"only {r['visits_by_thetadot_bin'][0] + r['visits_by_thetadot_bin'][7]} visits (seed 2); refine near upright instead"),
             (c[3], OK, "(c) Tests catch what plots hide",
              "terminal audit: Q(S′,·)=[100,200,300] must be ignored")]
    for i, (t0, col, h1, h2) in enumerate(items):
        text(fig, 0.49, 0.66 - 0.17 * i, h1, t, t0, 18, col, weight="bold")
        text(fig, 0.49, 0.615 - 0.17 * i, h2, t, t0 + 0.6, 15, FG)


def sc_close(fig, t, c, e):
    text(fig, 0.5, 0.80, "Summary", t, 0, 32, FG, ha="center", weight="bold")
    items = ["Env verified as supplied (and its termination rule documented)",
             "2880-state discretizer with 10 boundary tests",
             "ε-greedy frequencies match theory (seed 527)",
             "SARSA & Q-learning updates match by hand",
             "Q-learning, 3 seeds: learns to balance, 5 s goal not yet met",
             "Reflection: discretization is the main bottleneck"]
    for i, s in enumerate(items):
        text(fig, 0.22, 0.68 - 0.07 * i, "✓  " + s, t, 0.4 + 0.5 * i, 18, OK if i < 4 else FG)
    text(fig, 0.5, 0.16, "Thanks for watching!", t, c[1], 26, ACC, ha="center", weight="bold")


SCENES = [
    ("title", sc_title, [
        "This video walks through Homework 2 for COSC 527: building and verifying a tabular reinforcement learning agent for the Furuta pendulum.",
        "We cover six parts: the environment, the state discretization, the epsilon greedy policy, an update audit, a training experiment, and a reflection."]),
    ("setup", sc_setup, [
        "The Furuta pendulum has a motor driven horizontal arm with a free pendulum on its end.",
        "The state has four numbers: the arm angle phi, the arm velocity, the pendulum angle theta, and the pendulum velocity.",
        "Theta equal to zero means upright, and plus or minus pi means hanging straight down.",
        "The agent chooses one of three motor torques: minus zero point zero two, zero, or plus zero point zero two newton meters."]),
    ("env", sc_env, [
        "Part one tests the environment exactly as the professor supplied it, with no changes.",
        "Reset starts the pendulum near upright, less than one degree off vertical, and at rest.",
        "With zero torque, the pendulum falls past twenty degrees after zero point five five seconds, and the episode terminates with a reward of minus ten.",
        "Reaching the time limit sets truncated, not terminated, and the environment has no success flag, so the five second goal is measured by our own analysis code."]),
    ("finding", sc_finding, [
        "The assignment says that leaving the upright region should not end an episode.",
        "In the supplied environment, it does. Starting from the hanging position, the very first step terminates with minus ten.",
        "It even terminates at theta equal to two pi, which is physically upright, because the check uses the raw angle instead of the wrapped one.",
        "A swing up has to pass through every angle between down and up, so this rule makes swing up impossible.",
        "That is why the experiment studies the balancing task that the code actually implements."]),
    ("disc", sc_disc, [
        "Part two turns the continuous state into a Q table index.",
        "Theta gets ten bins: eight inside plus or minus twenty degrees, finest near upright, plus two open outer bins for everything else, since the environment terminates out there anyway.",
        "The arm angle and arm velocity get six bins each, and the pendulum velocity gets eight.",
        "That gives two thousand eight hundred eighty states, and eight thousand six hundred forty Q table entries.",
        "Values outside a range are clamped into the end bins, and a value exactly on an edge goes to the upper bin. Ten test states, including upright, downward, and angles near plus and minus pi, all map to valid indices."]),
    ("eps", sc_eps, [
        "Part three verifies the epsilon greedy policy.",
        "With probability epsilon we pick uniformly among all three actions. Otherwise we pick uniformly among the actions tied for the best value.",
        "For Q values of one, two, and two, with epsilon equal to zero point two, that predicts six point seven percent for action zero, and forty six point seven percent for each tied action.",
        "Ten thousand samples with random seed five twenty seven gave six point six, forty six point six, and forty six point eight percent, all within one standard error.",
        "With epsilon equal to zero, action zero never appears, and the tie still splits about fifty fifty, because exploitation breaks ties at random."]),
    ("audit", sc_audit, [
        "Part four audits one SARSA and one Q learning update on a real transition from the environment.",
        "The reward is about one, the next action is the exploratory action two, and the next state values are three point six six, four point zero eight, and two point three seven.",
        "SARSA uses the action actually taken: one plus zero point nine times two point three seven gives three point one three three.",
        "Q learning uses the maximum: one plus zero point nine times four point zero eight gives four point six seven two.",
        "With alpha equal to zero point one and an old value of four point five seven, the new values are four point four two six three, and four point five eight zero two, exactly what the program printed.",
        "The two targets agree whenever the next action is greedy. For a terminal transition, the continuation is zero, so both targets equal minus ten, even with next state values of one hundred, two hundred, and three hundred."]),
    ("train", sc_train, [
        "Part five trains Q learning on three random seeds.",
        "Each run uses fifteen hundred episodes, alpha zero point one, gamma zero point nine nine, epsilon decaying from one to zero point zero five over the first thousand episodes, a zero initialized table, and episodes of up to twenty seconds.",
        "For the first five hundred episodes, behavior is nearly random, and episodes last about half a second.",
        "Then learning takes off. The average episode length grows to between two and four and a half seconds, and every seed has some episodes that balance for the full twenty seconds.",
        "Even so, only five to eight percent of the Q table entries are ever visited."]),
    ("eval", sc_eval, [
        "For evaluation, the Q table is frozen, epsilon is zero, and no updates are made.",
        "Here is the greedy policy from seed two, shown at half speed.",
        "It holds the pendulum near upright for about three and a half seconds while the arm slowly drifts, and then the pendulum falls past twenty degrees.",
        "Seed zero lasted three point three seconds, and seed one only zero point six, so none of the three completed the five second balancing objective."]),
    ("reflect", sc_reflect, [
        "Finally, the reflection.",
        "The part most likely to limit learning is the discretization. About three quarters of all visits land in the two theta bins within two degrees of upright, so very different states have to share a single action.",
        "Pendulum speeds above three radians per second were visited only a handful of times. For balancing, those states almost always fall anyway, so resolution is better spent near upright.",
        "And tests like the terminal transition audit catch errors that a rising return plot would hide."]),
    ("close", sc_close, [
        "Every number in this video comes from the seeded scripts in the submission, and all of the details are in the report.",
        "Thanks for watching."]),
]


# ------------------------------------------------------------------- audio
def synthesize(tmp):
    items = [{"key": f"{name}_{i}", "text": s}
             for name, _, sents in SCENES for i, s in enumerate(sents)]
    (tmp / "lines.json").write_text(json.dumps(items), encoding="utf-8")
    ps1 = tmp / "tts.ps1"
    ps1.write_text(
        "param([string]$json, [string]$outdir)\n"
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        f"$s.SelectVoice('{VOICE}')\n"
        "$s.Rate = 1\n"
        f"$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({RATE}, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono)\n"
        "$items = Get-Content -Raw -Encoding UTF8 $json | ConvertFrom-Json\n"
        "foreach ($it in $items) {\n"
        "  $s.SetOutputToWaveFile((Join-Path $outdir ($it.key + '.wav')), $fmt)\n"
        "  $s.Speak($it.text)\n"
        "}\n"
        "$s.SetOutputToNull()\n$s.Dispose()\n", encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
                    str(tmp / "lines.json"), str(tmp)], check=True)

    def read(path):
        with wave.open(str(path)) as w:
            assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    def silence(sec):
        return np.zeros(int(round(sec * RATE)), dtype=np.int16)

    timeline, chunks, clock = [], [], 0.0
    for name, _, sents in SCENES:
        start = clock
        chunks.append(silence(LEAD))
        clock += LEAD
        cues, ends = [], []
        for i in range(len(sents)):
            pcm = read(tmp / f"{name}_{i}.wav")
            cues.append(clock - start)
            chunks.append(pcm)
            clock += len(pcm) / RATE
            ends.append(clock - start)
            chunks.append(silence(GAP))
            clock += GAP
        chunks.append(silence(TAIL))
        clock += TAIL
        timeline.append((start, clock, cues, ends))
    audio = np.concatenate(chunks)
    with wave.open(str(tmp / "voice.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())
    return timeline, len(audio) / RATE


# ------------------------------------------------------------------- video
def subtitle(fig, t, cues, ends, sents):
    for i, (t0, t1) in enumerate(zip(cues, ends)):
        if t0 <= t <= t1 + GAP:
            wrapped = textwrap.fill(sents[i], 105)
            fig.text(0.5, 0.035, wrapped, ha="center", va="bottom", size=13, color=FG,
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6, edgecolor="none"))
            return


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        print("synthesizing voiceover ...")
        timeline, total = synthesize(tmp)
        print(f"voiceover: {total:.1f} s")
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", str(tmp / "voice.wav"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
               "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(OUT)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        for (name, draw, sents), (start, end, cues, ends) in zip(SCENES, timeline):
            f0, f1 = int(round(start * FPS)), int(round(end * FPS))
            print(f"rendering {name}: {f1 - f0} frames")
            for f in range(f0, f1):
                t = f / FPS - start
                fig.clf()
                fig.set_facecolor(BG)
                draw(fig, t, cues, ends)
                subtitle(fig, t, cues, ends, sents)
                fig.canvas.draw()
                proc.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({total:.1f} s)")


if __name__ == "__main__":
    main()
