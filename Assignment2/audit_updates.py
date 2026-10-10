"""Part 4: audit one nonterminal and one terminal SARSA / Q-learning update.

The "hand" values are computed with plain arithmetic (no td_* functions) and
compared to the program's td_update results.
"""

import math

import numpy as np

from furuta_env import FurutaEnv
from io_utils import save_json
from tabular import Discretizer, td_update

ALPHA, GAMMA = 0.1, 0.9
Q_OLD_SA = 0.5
Q_NEXT = [1.0, 3.0, 2.0]   # unequal on purpose
A_NEXT = 0                 # an exploratory (non-greedy) next action


def find_transition(disc):
    """Hold +0.02 N m until the discrete state changes; return (S, A, R, S')."""
    env = FurutaEnv(seed=0)
    state, _ = env.reset(seed=0)
    s = disc.index(state)
    for _ in range(500):
        next_state, reward, terminated, truncated, _ = env.step(2)
        s2 = disc.index(next_state)
        assert not terminated
        if s2 != s:
            return s, 2, reward, s2
        s = s2
    raise RuntimeError("discrete state never changed")


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
    s, a, r, s2 = find_transition(disc)
    assert s != s2
    y_sarsa, qn_sarsa = audit(disc, "sarsa", s, a, r, s2, A_NEXT, Q_NEXT, False)
    y_q, qn_q = audit(disc, "qlearning", s, a, r, s2, A_NEXT, Q_NEXT, False)
    nonterminal = {
        "s_t": s, "a_t": a, "r_t1": r, "s_t1": s2, "a_t1": A_NEXT,
        "q_old_sa": Q_OLD_SA, "q_old_next": Q_NEXT,
        "y_sarsa": y_sarsa, "y_q": y_q, "qnew_sarsa": qn_sarsa, "qnew_q": qn_q,
    }

    r_term = -10.0
    variants = [[1.0, 3.0, 2.0], [1e6, -1e6, 5.0]]
    term_results = [
        [audit(disc, k, s, a, r_term, s2, None, v, True) for k in ("sarsa", "qlearning")]
        for v in variants
    ]
    assert all(res == term_results[0] for res in term_results)
    (y_s, qn_s), (y_qq, qn_qq) = term_results[0]
    terminal = {
        "s_t": s, "a_t": a, "r_t1": r_term, "q_old_sa": Q_OLD_SA,
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
