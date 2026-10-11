"""Turn results/*.json into LaTeX tables so report numbers come from the code.

Reads results/ and report/template/main_template.tex, writes report/main.tex.
Run from the project root:  python -m scripts.make_report
"""

import json
import re

import numpy as np

from hw2.io_utils import REPORT_DIR, RESULTS_DIR
from hw2.tabular import Discretizer
from hw2.training import moving_average

BIN_REASONS = {
    "phi": r"Arm must stay in $\pm\pi$ (unsafe limit); coarse, since $\phi$ only matters near the limit.",
    "phi_dot": r"0.02\,N\,m gives about 11\,rad/s$^2$ near the bottom, so speeds of order 10\,rad/s occur; outer bins are open-ended.",
    "theta": r"Edges at $\pm20^\circ$ make the center bin the success region; finer near upright, one split at $\pm\pi$.",
    "theta_dot": r"Natural frequency about 6.9\,rad/s; $\pm8$\,rad/s covers typical swings, outer bins catch the rest.",
}
RANGES = {"phi": r"$[-\pi,\pi]$", "phi_dot": "$[-10,10]$", "theta": r"$[-\pi,\pi]$", "theta_dot": "$[-8,8]$"}
NAMES = {"phi": r"$\phi$", "phi_dot": r"$\dot\phi$", "theta": r"$\widetilde\theta$", "theta_dot": r"$\dot\theta$"}


def load(name):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def num(x, nd=4):
    return f"{x:.{nd}f}"


def vec(v, nd=4):
    return "$[" + ", ".join(num(x, nd) for x in v) + "]$"


def vec_inline(v, nd=4):
    """Vector text for use INSIDE an existing math environment (no $ signs)."""
    return "[" + ", ".join(num(x, nd) for x in v) + "]"


def row(*cells):
    return " & ".join(str(c) for c in cells) + r" \\"


def table(colspec, header, rows):
    return [r"\begin{tabular}{" + colspec + "}", r"\toprule", row(*header), r"\midrule",
            *rows, r"\bottomrule", r"\end{tabular}"]


CONTENT = {}


def write(name, lines):
    """Collect a table/macro block in memory; flatten() inlines it into main.tex."""
    CONTENT[name] = "\n".join(lines) + "\n"


def flatten():
    """Inline every \\input{tables/<name>} of main_template.tex into one main.tex."""
    text = (REPORT_DIR / "template" / "main_template.tex").read_text(encoding="utf-8")
    text = re.sub(r"\\input\{tables/(\w+)\}", lambda m: CONTENT[m.group(1)].rstrip("\n"), text)
    (REPORT_DIR / "main.tex").write_text(text, encoding="utf-8")

COLORS = ["blue", "red", "green!50!black"]


def coords(xs, ys):
    return " ".join(f"({x:g},{y:.4g})" for x, y in zip(xs, ys))


def pgf_figure(xlabel, ylabel, series, band=None):
    """series: list of (label, xs, ys). Returns a self-contained pgfplots picture."""
    out = [r"\begin{tikzpicture}", r"\begin{axis}[width=\linewidth,height=5.2cm,grid=major,"
           r"xlabel={" + xlabel + "},ylabel={" + ylabel + r"},legend style={font=\tiny},"
           r"label style={font=\small},tick label style={font=\tiny}]"]
    if band:
        out.append(r"\addplot[draw=none,fill=green!30,opacity=0.4,forget plot] coordinates "
                   f"{{({band[0]},-20) ({band[1]},-20) ({band[1]},20) ({band[0]},20)}} " + r"\closedcycle;")
    for i, (label, xs, ys) in enumerate(series):
        out.append(r"\addplot[thick,color=" + COLORS[i % 3] + "] coordinates {" + coords(xs, ys) + "};")
        out.append(r"\addlegendentry{" + label + "}")
    out += [r"\end{axis}", r"\end{tikzpicture}"]
    return out


def add_figures(cfg):
    data = np.load(RESULTS_DIR / "experiment.npz")
    w, dt, seeds = cfg["moving_window"], cfg["dt"], cfg["seeds"]
    step = 10

    def per_seed(key, ma):
        res = []
        for i, seed in enumerate(seeds):
            y = moving_average(data[key][i], w) if ma else data[key][i]
            xs = np.arange(0, len(y), step)
            res.append((f"seed {seed}", xs, y[::step]))
        return res

    write("fig_returns", pgf_figure("episode", f"return (MA {w})", per_seed("returns", True)))
    write("fig_durations", pgf_figure("episode", f"duration [steps] (MA {w})", per_seed("durations", True)))
    write("fig_visited", pgf_figure("episode", "visited fraction of (s,a)", per_seed("visited_frac", False)))
    ev, tmax = [], 0.0
    for seed in seeds:
        th = np.rad2deg(data[f"eval_theta_{seed}"])
        t = np.arange(len(th)) * dt
        tmax = max(tmax, t[-1])
        ev.append((f"seed {seed}", t, th))
    write("fig_eval", pgf_figure("time [s]", r"$\widetilde{\theta}$ [deg]", ev, band=(0, f"{tmax:g}")))
    evp = []
    for seed in seeds:
        ph = data[f"eval_phi_{seed}"]
        evp.append((f"seed {seed}", np.arange(len(ph)) * dt, ph))
    write("fig_evalphi", pgf_figure("time [s]", r"arm angle $\phi$ [rad]", evp))


def main():
    env = load("verify_env")
    disc = load("verify_discretizer")
    eps = load("verify_epsilon_greedy")
    aud = load("audit_updates")
    exp = load("experiment")
    cfg = exp["config"]

    # numbers.tex
    t = env["transition"]
    nums = {
        "nstates": disc["n_states"], "nq": disc["n_q"], "nactions": disc["n_actions"],
        "shape": " \\times ".join(map(str, disc["shape"])),
        "initstate": vec_inline(env["initial_state"]), "inittheta": num(env["initial_theta_wrapped"]),
        "dt": env["dt"], "successsteps": env["success_steps"], "maxsteps": env["max_steps"],
        "transS": vec_inline(t["S_t"]), "transA": t["A_t"], "transR": num(t["R_t1"]), "transSn": vec_inline(t["S_t1"]),
        "outsidedeg": num(env["outside_upright"]["theta_wrapped_deg"], 1),
        "epsseed": eps["seed"], "epsn": eps["n_samples"],
        "nepisodes": cfg["n_episodes"], "maxepsec": cfg["max_episode_seconds"],
        "cfgalpha": cfg["alpha"], "cfggamma": cfg["gamma"], "epsstart": cfg["eps_start"],
        "epsend": cfg["eps_end"], "mawindow": cfg["moving_window"],
        "expseeds": ", ".join(map(str, cfg["seeds"])),
    }
    rc, zt = env["upright_counter_reset"], env["outcomes"]["zero_torque_downward_full_episode"]
    nums.update({
        "ctrBefore": rc["upright_steps_before_exit"], "ctrAfter": rc["upright_steps_after_exit"],
        "ctrTermExit": str(rc["terminated_on_exit"]).lower(),
        "ctrReentry": rc["steps_after_reentry_until_done"], "ctrOutcome": rc["outcome"],
        "zeroSteps": zt["steps"], "zeroTerm": str(zt["terminated"]).lower(),
        "zeroTrunc": str(zt["truncated"]).lower(), "zeroOutcome": zt["outcome"],
    })
    raw = env["raw_env"]
    nums.update({
        "rawinit": vec_inline(raw["initial_state"]), "rawR": num(raw["transition"]["R_t1"]),
        "rawTerm": str(raw["transition"]["terminated"]).lower(),
        "rawlimit": f'{raw["angle_limit_deg"]:g}',
        "rawDownSteps": raw["downward_start"]["steps"],
        "rawDownReward": f'${raw["downward_start"]["reward"]:g}$',
        "rawDownTheta": f'{raw["downward_start"]["theta_deg"]:g}',
        "rawDownTerm": str(raw["downward_start"]["terminated"]).lower(),
    })
    for k in ("success", "unsafe", "timeout"):
        o = env["outcomes"][k]
        nums[f"out{k}"] = f'{o["outcome"]} after {o["steps"]} steps (terminated={o["terminated"]}, truncated={o["truncated"]})'
    write("numbers", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()])

    # actions
    write("actions", table("cc", ["Action index", r"Torque [N\,m]"],
                           [row(a["index"], f'{a["torque"]:+.2f}') for a in env["actions"]]))

    # bins
    n_bins = dict(zip(("phi", "phi_dot", "theta", "theta_dot"), disc["shape"]))
    write("bins", table("llcp{6.5cm}", ["Variable", "Selected range", "Bins", "Reason"],
                         [row(NAMES[k], RANGES[k], n_bins[k], BIN_REASONS[k]) for k in NAMES]))

    # discretizer tests
    def state_str(s):
        return "$[" + ", ".join(f"{x:.3f}" for x in s) + "]$"
    write("disc_tests", table("lcc", ["Test", "Continuous state", "Tuple / index"],
                              [row(x["name"].replace("_", r"\_"), state_str(x["state"]), f'{tuple(x["tuple"])} / {x["index"]}')
                               for x in disc["tests"]]))

    # epsilon-greedy
    rows = []
    for r in eps["runs"]:
        rows.append(row(f'{r["epsilon"]:.2f}', "observed", *[num(x) for x in r["observed"]]))
        rows.append(row("", "implied", *[num(x) for x in r["implied"]]))
    write("eps", table("llccc", [r"$\epsilon$", "", "action 0", "action 1", "action 2"], rows))

    # audit
    nt, tm = aud["nonterminal"], aud["terminal"]
    a_rows = [
        row(r"$S_t, A_t$", f'{nt["s_t"]}, {nt["a_t"]}', "", ""),
        row(r"$S_t$ bins $(\phi,\dot\phi,\widetilde\theta,\dot\theta)$", str(tuple(nt["s_t_tuple"])), "", ""),
        row(r"$S_{t+1}$ bins", str(tuple(nt["s_t1_tuple"])), "", ""),
        row(r"$R_{t+1}$", num(nt["r_t1"]), "", ""),
        row(r"$S_{t+1}, A_{t+1}$", f'{nt["s_t1"]}, {nt["a_t1"]}', "", ""),
        row(r"$Q_{old}(S_t,A_t)$", num(nt["q_old_sa"]), "", ""),
        row(r"$Q_{old}(S_{t+1},\cdot)$", vec(nt["q_old_next"]), "", ""),
        row(r"$\alpha,\gamma$", f'{aud["alpha"]}, {aud["gamma"]}', "", ""),
        row("", "", "SARSA", "Q-learning"),
        row("Target $Y$", "", num(nt["y_sarsa"]), num(nt["y_q"])),
        row(r"$Q_{new}(S_t,A_t)$", "", num(nt["qnew_sarsa"]), num(nt["qnew_q"])),
    ]
    t_rows = [
        row(r"$S_t$ index / bins", f'{tm["s_t"]} / {tuple(tm["s_t_tuple"])}', "", ""),
        row(r"$S_{t+1}$ index / bins", f'{tm["s_t1"]} / {tuple(tm["s_t1_tuple"])}', "", ""),
        row(r"$A_t$", tm["a_t"], "", ""),
        row(r"$R_{t+1}$ (unsafe termination)", num(tm["r_t1"]), "", ""),
        row(r"$Q_{old}(S_t,A_t)$", num(tm["q_old_sa"]), "", ""),
        row(r"$Q(S_{t+1},\cdot)$ variant 1", vec(tm["q_next_variants"][0]), "", ""),
        row(r"$Q(S_{t+1},\cdot)$ variant 2", vec(tm["q_next_variants"][1], 1), "", ""),
        row("", "", "SARSA", "Q-learning"),
        row("Target $Y$ (both variants)", "", num(tm["y_sarsa"]), num(tm["y_q"])),
        row(r"$Q_{new}(S_t,A_t)$", "", num(tm["qnew_sarsa"]), num(tm["qnew_q"])),
    ]
    assert tm["s_t_and_s_t1_share_q_row"] and tm["s_t"] == tm["s_t1"]
    assert all(v[tm["a_t"]] == tm["q_old_sa"] for v in tm["q_next_variants"])
    assert nt["crosses_pi_wrap_seam"]
    nums2 = {
        "termS": vec_inline(tm["S_t"], 2), "termSn": vec_inline(tm["S_t1"], 2), "termA": tm["a_t"],
        "termQsa": num(tm["q_old_sa"], 1), "termR": num(tm["r_t1"], 0),
        "seamFrom": nt["s_t_tuple"][2], "seamTo": nt["s_t1_tuple"][2],
        "auditgamma": aud["gamma"], "auditalpha": aud["alpha"],
    }
    write("numbers_audit", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums2.items()])
    write("audit", table("lccc", ["Nonterminal", "", "", ""], a_rows)
          + [r"\medskip"] + table("lccc", ["Terminal", "", "", ""], t_rows))

    # experiment
    data0 = np.load(RESULTS_DIR / "experiment.npz")
    seeds = cfg["seeds"]
    nw = 100
    first = [float(np.mean(data0["durations"][i][:nw])) for i in range(len(seeds))]
    last = [float(np.mean(data0["durations"][i][-nw:])) for i in range(len(seeds))]
    e_rows = []
    for i, s_ in enumerate(exp["seeds"]):
        oc = s_["outcome_counts"]
        e_rows.append(row(s_["seed"], num(s_["final_moving_average"], 2), num(first[i], 0), num(last[i], 0),
                          num(s_["visited_fraction_final"], 3), f'{100 * oc.get("unsafe", 0) / cfg["n_episodes"]:.1f}'))
    write("experiment", table("cccccc",
          ["Seed", "Final MA return", f"Mean dur. first {nw} ep.", f"Mean dur. last {nw} ep.",
           "Visited frac.", r"Unsafe episodes [\%]"], e_rows))
    ev_rows = []
    for s_ in exp["seeds"]:
        e = s_["eval"]
        ev_rows.append(row(s_["seed"], e["classification"].replace("_", r"\_"), e["outcome"], e["steps"],
                           num(e["max_deviation_from_downward_deg"], 1), num(e["min_abs_theta_deg"], 1),
                           e["torque_switches"], e["n_swing_peaks"], num(e["final_phi"], 2)))
    write("evaltable", table("ccccrrrrr",
          ["Seed", "Class", "Outcome", "Steps", r"Max dev. [$^\circ$]", r"Min $|\widetilde\theta|$ [$^\circ$]",
           "Torque switches", "Swing peaks", r"final $\phi$ [rad]"], ev_rows))
    sd = exp["seeds"]
    assert len({x["eval"]["classification"] for x in sd}) == 1
    assert all(x["eval"]["outcome"] == "timeout" for x in sd)

    def span(vals, nd=0):
        lo, hi = min(vals), max(vals)
        return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}--{hi:.{nd}f}"
    n_ep = cfg["n_episodes"]
    unsafe = sum(x["outcome_counts"].get("unsafe", 0) for x in sd)
    timeout = sum(x["outcome_counts"].get("timeout", 0) for x in sd)
    succ = sum(x["outcome_counts"].get("success", 0) for x in sd)
    tot = n_ep * len(sd)
    dd = Discretizer()
    vis = data0["state_visits"].sum(axis=0).reshape(dd.shape)
    share = [vis.sum(axis=tuple(i for i in range(4) if i != ax)) / vis.sum() for ax in range(4)]
    th_deg = np.rad2deg(dd.bins[2])
    nth = dd.shape[2]
    inner = [i for i in range(nth) if 0 < i < nth - 1 and th_deg[i - 1] >= -75 and th_deg[i] <= 75]
    near = [1, nth - 2]
    upright_bin = int(np.searchsorted(th_deg, 0.0, side="right"))
    pct = lambda x: f"{100 * x:.1f}"
    nums3 = {
        "oldDurFrom": "705", "oldDurTo": "267--369",
        "durFirstLast": "; ".join(f"seed {x['seed']}: {first[i]:.0f} to {last[i]:.0f}" for i, x in enumerate(sd)),
        "durFirstN": nw,
        "evClass": sd[0]["eval"]["classification"].replace("_", r"\_"),
        "evMaxDev": span([x["eval"]["max_deviation_from_downward_deg"] for x in sd], 0),
        "evMinTheta": span([x["eval"]["min_abs_theta_deg"] for x in sd], 0),
        "evSwitches": span([x["eval"]["torque_switches"] for x in sd]),
        "evPeaks": span([x["eval"]["n_swing_peaks"] for x in sd]),
        "evSteps": span([x["eval"]["steps"] for x in sd]),
        "visFrac": span([100 * x["visited_fraction_final"] for x in sd], 0),
        "unsafeCount": unsafe, "timeoutCount": timeout, "successCount": succ, "totalEps": tot,
        "unsafePct": f"{100 * unsafe / tot:.0f}",
        "phidotCenter": pct(share[1][dd.shape[1] // 2]),
        "thdotCenter": pct(share[3][dd.shape[3] // 2]),
        "seamLow": pct(share[2][0]), "seamHigh": pct(share[2][-1]),
        "thInnerShare": f"{100 * sum(share[2][i] for i in inner):.2f}",
        "thUprightShare": f"{100 * share[2][upright_bin]:.3f}",
        "thNearShare": pct(sum(share[2][i] for i in near)),
        "thSeamBins": f"{0} and {nth - 1}",
    }
    write("numbers_exp", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums3.items()])

    # per-variable visit marginals (seed-pooled), for part 6(b)
    d = Discretizer()
    data = np.load(RESULTS_DIR / "experiment.npz")
    visits = data["state_visits"].sum(axis=0).reshape(d.shape)
    total = visits.sum()
    v_rows = []
    for axis, key in enumerate(("phi", "phi_dot", "theta", "theta_dot")):
        other = tuple(i for i in range(4) if i != axis)
        share = visits.sum(axis=other) / total
        v_rows.append(row(NAMES[key], ", ".join(f"{100 * x:.1f}" for x in share)))
    write("visits", table("lp{9cm}", ["Variable", r"Share of visits per bin [\%], low to high bin index"], v_rows))
    add_figures(cfg)
    flatten()
    print("wrote", REPORT_DIR / "main.tex")


if __name__ == "__main__":
    main()
