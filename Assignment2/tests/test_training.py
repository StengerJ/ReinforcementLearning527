import numpy as np
import pytest

from furuta_env import FurutaEnv
from tabular import Discretizer
from training import classify_evaluation, evaluate, moving_average, run_episode, train


def test_moving_average():
    assert moving_average([1, 2, 3, 4], 2) == pytest.approx([1.0, 1.5, 2.5, 3.5])


def test_truncation_still_bootstraps():
    disc = Discretizer()
    env = FurutaEnv(max_steps=1, seed=0)
    Q = np.full((disc.n_states, 3), 5.0)
    before = Q.copy()
    rng = np.random.default_rng(0)
    ret, steps, outcome = run_episode(env, disc, Q, rng, 0.0, 0.1, 0.99)
    assert outcome == "timeout" and steps == 1
    changed = np.argwhere(Q != before)
    assert len(changed) == 1
    s, a = changed[0]
    assert Q[s, a] == pytest.approx(5.0 + 0.1 * (ret + 0.99 * 5.0 - 5.0))


def test_evaluation_does_not_change_q():
    disc = Discretizer()
    env = FurutaEnv(max_steps=40, seed=1)
    Q = np.random.default_rng(0).normal(size=(disc.n_states, 3))
    before = Q.copy()
    ev = evaluate(env, disc, Q, seed=5)
    assert np.array_equal(Q, before)
    assert len(ev["theta_wrapped"]) == ev["steps"] + 1


def test_learn_false_leaves_q_unchanged():
    disc = Discretizer()
    env = FurutaEnv(max_steps=20, seed=0)
    Q = np.zeros((disc.n_states, 3))
    run_episode(env, disc, Q, np.random.default_rng(0), 1.0, 0.1, 0.99, learn=False)
    assert not Q.any()


def test_train_is_reproducible_and_reports_visits():
    kw = dict(n_episodes=3, max_steps=30, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05)
    a = train(seed=2, **kw)
    b = train(seed=2, **kw)
    assert np.array_equal(a["returns"], b["returns"])
    assert 0.0 < a["visited_frac"][-1] <= 1.0
    assert np.all(np.diff(a["visited_frac"]) >= 0)
    assert a["state_visits"].sum() == a["durations"].sum()
    assert a["Q"].any()


def test_train_single_episode_does_not_divide_by_zero():
    res = train(seed=0, n_episodes=1, max_steps=10, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05)
    assert len(res["returns"]) == 1


def test_classify_evaluation():
    near = np.full(10, np.pi - 0.1)
    assert classify_evaluation(near, "timeout") == "near_downward"
    swing = np.array([np.pi, np.pi - 1.0, np.pi])
    assert classify_evaluation(swing, "timeout") == "growing_swing"
    upright = np.array([np.pi, 1.0, 0.2])
    assert classify_evaluation(upright, "unsafe") == "reached_upright_region"
    assert classify_evaluation(upright, "success") == "completed_5s_balance"
