"""Part 3: empirical epsilon-greedy action frequencies vs implied probabilities."""

import numpy as np

from hw2.io_utils import save_json
from hw2.tabular import action_probabilities, epsilon_greedy

SEED = 0
N_SAMPLES = 10_000
Q_ROW = np.array([1.0, 2.0, 2.0])


def sample_frequencies(q_row, epsilon, n, seed):
    rng = np.random.default_rng(seed)
    q = q_row.copy()
    counts = np.zeros(len(q))
    for _ in range(n):
        counts[epsilon_greedy(q, epsilon, rng)] += 1
    assert np.array_equal(q, q_row), "Q table must not change"
    return counts / n


def main():
    runs = []
    for eps in (0.20, 0.0):
        runs.append({
            "epsilon": eps,
            "observed": sample_frequencies(Q_ROW, eps, N_SAMPLES, SEED),
            "implied": action_probabilities(Q_ROW, eps),
        })
    out = {"seed": SEED, "n_samples": N_SAMPLES, "q_row": Q_ROW, "runs": runs}
    save_json("verify_epsilon_greedy", out)
    return out


if __name__ == "__main__":
    result = main()
    for r in result["runs"]:
        print(f'eps={r["epsilon"]}: observed={r["observed"]} implied={r["implied"]}')
