"""HW2 Part 3: epsilon-greedy behaviour policy and its empirical verification.

Tie rule: with probability epsilon pick uniformly among ALL actions (greedy
ones included); otherwise pick uniformly among the actions tied for the maximum.
So P(a) = epsilon / n_a + (1 - epsilon) / |greedy|  if a is greedy,
     P(a) = epsilon / n_a                              otherwise.

Run:  python stenger_hw2_epsilon_greedy.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SEED = 527
N_SAMPLES = 10_000
Q_TEST = np.array([1.0, 2.0, 2.0])


def greedy_actions(q_row):
    """Indices of all actions whose value equals the maximum (exact ties)."""
    q_row = np.asarray(q_row, dtype=float)
    return np.flatnonzero(q_row == np.max(q_row))


def epsilon_greedy(q_row, epsilon, rng):
    """Sample one action from the epsilon-greedy policy for the row Q(s, .)."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    return int(rng.choice(greedy_actions(q_row)))


def action_probabilities(q_row, epsilon):
    """Exact probabilities implied by ``epsilon_greedy`` (used for comparison)."""
    n = len(q_row)
    greedy = greedy_actions(q_row)
    probs = np.full(n, epsilon / n)
    probs[greedy] += (1.0 - epsilon) / len(greedy)
    return probs


def main():
    out = {"seed": SEED, "n": N_SAMPLES, "q": Q_TEST.tolist(), "runs": []}
    for epsilon in (0.20, 0.0):
        rng = np.random.default_rng(SEED)
        q_before = Q_TEST.copy()
        counts = np.zeros(len(Q_TEST), dtype=int)
        for _ in range(N_SAMPLES):
            counts[epsilon_greedy(Q_TEST, epsilon, rng)] += 1
        assert np.array_equal(Q_TEST, q_before), "Q table must not change"
        freq = counts / N_SAMPLES
        expected = action_probabilities(Q_TEST, epsilon)
        # Binomial standard error of each empirical frequency.
        se = np.sqrt(expected * (1 - expected) / N_SAMPLES)
        print(f"epsilon={epsilon:.2f} seed={SEED}")
        print("  counts      :", counts.tolist())
        print("  empirical   :", np.round(freq, 4).tolist())
        print("  expected    :", np.round(expected, 4).tolist())
        print("  std. error  :", np.round(se, 4).tolist())
        out["runs"].append({"epsilon": epsilon, "counts": counts.tolist(),
                            "freq": freq.tolist(), "expected": expected.tolist(),
                            "se": se.tolist()})
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "part3_epsilon_greedy.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
