import numpy as np
import pytest

from tabular import td_target, td_update

R, GAMMA, ALPHA = 1.0, 0.9, 0.1


def make_Q():
    Q = np.zeros((4, 3))
    Q[0, 1] = 0.5
    Q[2] = [1.0, 3.0, 2.0]
    return Q


def test_sarsa_and_q_targets_differ_on_unequal_next_values():
    q_next = np.array([1.0, 3.0, 2.0])
    assert td_target("sarsa", R, GAMMA, q_next, 0, False) == pytest.approx(1.9)
    assert td_target("qlearning", R, GAMMA, q_next, 0, False) == pytest.approx(3.7)


def test_targets_equal_when_next_action_is_greedy():
    q_next = np.array([1.0, 3.0, 2.0])
    assert td_target("sarsa", R, GAMMA, q_next, 1, False) == td_target(
        "qlearning", R, GAMMA, q_next, 1, False
    )


def test_updates_match_hand_calculation():
    Q = make_Q()
    y = td_update(Q, 0, 1, "sarsa", R, GAMMA, ALPHA, 2, 0, False)
    assert y == pytest.approx(1.9)
    assert Q[0, 1] == pytest.approx(0.5 + 0.1 * (1.9 - 0.5))  # 0.64

    Q = make_Q()
    y = td_update(Q, 0, 1, "qlearning", R, GAMMA, ALPHA, 2, 0, False)
    assert y == pytest.approx(3.7)
    assert Q[0, 1] == pytest.approx(0.82)


@pytest.mark.parametrize("kind", ["sarsa", "qlearning"])
def test_terminal_ignores_next_state_values(kind):
    results = []
    for next_row in ([1.0, 3.0, 2.0], [1e6, -1e6, 5.0]):
        Q = make_Q()
        Q[2] = next_row
        y = td_update(Q, 0, 1, kind, -10.0, GAMMA, ALPHA, 2, None, True)
        results.append((y, Q[0, 1]))
    assert results[0] == results[1]
    assert results[0][0] == -10.0
    assert results[0][1] == pytest.approx(0.5 + 0.1 * (-10.0 - 0.5))  # -0.55


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        td_target("expected_sarsa", R, GAMMA, np.zeros(3), 0, False)
