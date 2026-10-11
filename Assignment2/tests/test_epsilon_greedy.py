import numpy as np
import pytest

from hw2.tabular import action_probabilities, epsilon_greedy, greedy_actions


def sample(q_row, eps, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    q = np.array(q_row, dtype=float)
    counts = np.zeros(len(q))
    for _ in range(n):
        counts[epsilon_greedy(q, eps, rng)] += 1
    assert np.array_equal(q, q_row)  # Q untouched
    return counts / n


def test_greedy_actions_ties():
    assert list(greedy_actions(np.array([1.0, 2.0, 2.0]))) == [1, 2]


def test_implied_probabilities():
    p = action_probabilities(np.array([1.0, 2.0, 2.0]), 0.2)
    assert p == pytest.approx([0.2 / 3, 0.2 / 3 + 0.4, 0.2 / 3 + 0.4])
    assert p.sum() == pytest.approx(1.0)


def test_empirical_matches_implied_eps_020():
    q = [1.0, 2.0, 2.0]
    freq = sample(q, 0.2)
    assert freq == pytest.approx(action_probabilities(np.array(q), 0.2), abs=0.02)


def test_eps_zero_never_picks_nongreedy():
    freq = sample([1.0, 2.0, 2.0], 0.0)
    assert freq[0] == 0.0
    assert freq[1] == pytest.approx(0.5, abs=0.02)


def test_all_equal_row_is_uniform():
    freq = sample([0.0, 0.0, 0.0], 0.0)
    assert freq == pytest.approx([1 / 3] * 3, abs=0.02)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_epsilon_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        epsilon_greedy(np.zeros(3), bad, np.random.default_rng(0))
