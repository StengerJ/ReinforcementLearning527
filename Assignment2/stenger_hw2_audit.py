import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "futura_env"))
from furuta_env import FurutaEnv
from stenger_hw2_state_discretization import N_ACTIONS, N_STATES, discretize, state_index

RESULTS = HERE / "results"
ALPHA = 0.1
GAMMA = 0.9
SEED = 4


def td_target(kind, reward, gamma, q_next_row, next_action, terminated):
    if terminated:
        return float(reward)
    if kind == "sarsa":
        return float(reward + gamma * q_next_row[next_action])
    if kind == "qlearning":
        return float(reward + gamma * np.max(q_next_row))
    raise ValueError(f"unknown kind {kind!r}")


def td_update(q, s, a, kind, reward, gamma, alpha, s_next, next_action, terminated):
    target = td_target(kind, reward, gamma, q[s_next], next_action, terminated)
    q[s, a] += alpha * (target - q[s, a])
    return target


def audit(name, q, s, a, r, s2, a2, terminated, state, next_state):
    q_old_sa = float(q[s, a])
    q_old_next = q[s2].copy()
    cont = 0.0 if terminated else 1.0
    y_sarsa = r + GAMMA * cont * q_old_next[a2]
    y_q = r + GAMMA * cont * q_old_next.max()
    new_sarsa = q_old_sa + ALPHA * (y_sarsa - q_old_sa)
    new_q = q_old_sa + ALPHA * (y_q - q_old_sa)

    q_s, q_q = q.copy(), q.copy()
    prog_y_sarsa = td_update(q_s, s, a, "sarsa", r, GAMMA, ALPHA, s2, a2, terminated)
    prog_y_q = td_update(q_q, s, a, "qlearning", r, GAMMA, ALPHA, s2, a2, terminated)
    assert np.isclose(prog_y_sarsa, y_sarsa) and np.isclose(q_s[s, a], new_sarsa)
    assert np.isclose(prog_y_q, y_q) and np.isclose(q_q[s, a], new_q)
    assert np.count_nonzero(q_s != q) <= 1 and np.count_nonzero(q_q != q) <= 1

    rec = {
        "name": name, "S_t": [float(v) for v in state], "S_t_bins": list(discretize(state)),
        "S_t_index": s, "A_t": a, "R": r, "S_next": [float(v) for v in next_state],
        "S_next_bins": list(discretize(next_state)), "S_next_index": s2, "A_next": a2,
        "terminated": terminated, "Q_old_sa": q_old_sa, "Q_old_next": q_old_next.tolist(),
        "Y_sarsa": y_sarsa, "Y_q": y_q, "new_sarsa": new_sarsa, "new_q": new_q,
        "prog_Y_sarsa": prog_y_sarsa, "prog_Y_q": prog_y_q,
        "prog_new_sarsa": float(q_s[s, a]), "prog_new_q": float(q_q[s, a]),
    }
    print(f"--- {name} transition (alpha={ALPHA}, gamma={GAMMA})")
    for key in ("S_t", "S_t_index", "A_t", "R", "S_next", "S_next_index", "A_next",
                "terminated", "Q_old_sa", "Q_old_next"):
        print(f"  {key:13s} = {rec[key]}")
    print(f"  Y_SARSA = {r:.6f} + {GAMMA}*{cont:g}*{q_old_next[a2]:.4f} = {y_sarsa:.6f}"
          f"  (program {prog_y_sarsa:.6f})")
    print(f"  Y_Q     = {r:.6f} + {GAMMA}*{cont:g}*{q_old_next.max():.4f} = {y_q:.6f}"
          f"  (program {prog_y_q:.6f})")
    print(f"  Q_new SARSA = {q_old_sa:.4f} + {ALPHA}*({y_sarsa:.6f} - {q_old_sa:.4f}) = {new_sarsa:.6f}"
          f"  (program {q_s[s, a]:.6f})")
    print(f"  Q_new Q     = {q_old_sa:.4f} + {ALPHA}*({y_q:.6f} - {q_old_sa:.4f}) = {new_q:.6f}"
          f"  (program {q_q[s, a]:.6f})")
    return rec


def main():
    rng = np.random.default_rng(SEED)
    q = np.round(rng.uniform(0.0, 5.0, size=(N_STATES, N_ACTIONS)), 2)

    env = FurutaEnv(seed=SEED)
    state, _ = env.reset()
    s = state_index(state)
    a = 2
    next_state, r, terminated, truncated, _ = env.step(a)
    assert not terminated and not truncated
    s2 = state_index(next_state)
    a2 = int(np.argmin(q[s2]))
    assert len(set(q[s2])) > 1, "next-state values must not all be equal"
    audit("nonterminal", q, s, a, r, s2, a2, terminated, state, next_state)

    state = np.array([0.0, 0.0, np.deg2rad(19.9), 2.0])
    env.reset(initial_state=state)
    a = 1
    next_state, r, terminated, truncated, _ = env.step(a)
    assert terminated, "this transition must leave the 20-degree region"
    s, s2 = state_index(state), state_index(next_state)
    q[s2] = [100.0, 200.0, 300.0]
    a2 = 2
    audit("terminal", q, s, a, r, s2, a2, terminated, state, next_state)


if __name__ == "__main__":
    with io.StringIO() as buffer:
        with contextlib.redirect_stdout(buffer):
            main()
        report = buffer.getvalue()
    print(report, end="")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "part4_audit.txt").write_text(report, encoding="utf-8")
