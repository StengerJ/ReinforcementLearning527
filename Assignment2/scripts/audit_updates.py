"""Part 4: audit one nonterminal and one terminal SARSA / Q-learning update.

The "hand" values are computed with plain arithmetic (no td_* functions) and
compared to the program's td_update results.
"""

import math

import numpy as np

from hw2.task import SwingUpTask
from hw2.io_utils import save_json
from hw2.tabular import Discretizer, td_update

ALPHA, GAMMA = 0.1, 0.9
Q_OLD_SA = 0.5
Q_NEXT = [1.0, 3.0, 2.0]   # unequal on purpose
A_NEXT = 0                 # an exploratory (non-greedy) next action


def find_transition(disc):
    """Hold +0.02 N m until the discrete state changes; return the real transition."""
    env = SwingUpTask(seed=0)
    state, _ = env.reset(seed=0)
    for _ in range(500):
        next_state, reward, terminated, truncated, _ = env.step(2)
        assert not terminated
        if disc.index(next_state) != disc.index(state):
            return state, 2, reward, next_state
        state = next_state
    raise RuntimeError("discrete state never changed")


def real_terminal_transition():
    """Record an actual unsafe termination from the environment (arm past +-pi)."""
    env = SwingUpTask()
    state, _ = env.reset(initial_state=np.array([3.1, 5.0, np.pi, 0.0]))
    for _ in range(2000):
        next_state, reward, terminated, truncated, info = env.step(1)
        if terminated:
            assert info["outcome"] == "unsafe"
            return state, 1, reward, next_state
        state = next_state
    raise RuntimeError("episode never became unsafe")


def audit(disc, kind, s, a, r, s2, a2, q_next, terminated):
    Q = np.zeros((disc.n_states, 3))
    Q[s2] = q_next
    Q[s, a] = Q_OLD_SA
    y_prog = td_update(Q, s, a, kind, r, GAMMA, ALPHA, s2, a2, terminated)
    if terminated:
        y_hand = r
    elif kind == "sarsa":
        y_hand = r + GAMMA * q_next[a2]
    else:
        y_hand = r + GAMMA * max(q_next)
    qnew_hand = Q_OLD_SA + ALPHA * (y_hand - Q_OLD_SA)
    assert math.isclose(y_prog, y_hand) and math.isclose(Q[s, a], qnew_hand)
    return y_hand, qnew_hand


def main():
    disc = Discretizer()
    state, a, r, next_state = find_transition(disc)
    s, s2 = disc.index(state), disc.index(next_state)
    t_s, t_s2 = disc.indices(state), disc.indices(next_state)
    assert s != s2
    y_sarsa, qn_sarsa = audit(disc, "sarsa", s, a, r, s2, A_NEXT, Q_NEXT, False)
    y_q, qn_q = audit(disc, "qlearning", s, a, r, s2, A_NEXT, Q_NEXT, False)
    n_theta = disc.shape[2]
    seam = {t_s[2], t_s2[2]} == {0, n_theta - 1}
    nonterminal = {
        "S_t": state, "S_t1": next_state,
        "s_t": s, "a_t": a, "r_t1": r, "s_t1": s2, "a_t1": A_NEXT,
        "s_t_tuple": t_s, "s_t1_tuple": t_s2,
        "crosses_pi_wrap_seam": seam,
        "note": (f"theta bin {t_s[2]} -> {t_s2[2]}: S_t and S_t+1 lie on opposite sides of the "
                 "+-pi wrap seam, so this is one small physical step" if seam
                 else "transition does not cross the +-pi wrap seam"),
        "q_old_sa": Q_OLD_SA, "q_old_next": Q_NEXT,
        "y_sarsa": y_sarsa, "y_q": y_q, "qnew_sarsa": qn_sarsa, "qnew_q": qn_q,
    }

    state_t, a_t, r_term, state_t1 = real_terminal_transition()
    st, st1 = disc.index(state_t), disc.index(state_t1)
    # The arm saturates in the last phi bin, so S_t and S_t+1 can share a Q row; then
    # Q(S_t, A_t) is one entry of that row and is held at Q_OLD_SA in both variants.
    variants = [[1.0, 3.0, 2.0], [1e6, -1e6, 5.0]]
    for v in variants:
        if st == st1:
            v[a_t] = Q_OLD_SA
    term_results = [
        [audit(disc, k, st, a_t, r_term, st1, None, v, True) for k in ("sarsa", "qlearning")]
        for v in variants
    ]
    assert all(res == term_results[0] for res in term_results)
    (y_s, qn_s), (y_qq, qn_qq) = term_results[0]
    terminal = {
        "S_t": state_t, "S_t1": state_t1, "s_t": st, "a_t": a_t, "r_t1": r_term, "s_t1": st1,
        "s_t_tuple": disc.indices(state_t), "s_t1_tuple": disc.indices(state_t1),
        "source": "real unsafe termination recorded from SwingUpTask (reset to [3.1, 5.0, pi, 0])",
        "s_t_and_s_t1_share_q_row": st == st1,
        "q_old_sa": Q_OLD_SA,
        "q_next_variants": variants,
        "y_sarsa": y_s, "y_q": y_qq, "qnew_sarsa": qn_s, "qnew_q": qn_qq,
    }
    out = {"alpha": ALPHA, "gamma": GAMMA, "nonterminal": nonterminal, "terminal": terminal}
    save_json("audit_updates", out)
    return out


if __name__ == "__main__":
    result = main()
    for k in ("nonterminal", "terminal"):
        print(k, result[k])
