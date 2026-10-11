import numpy as np
import pytest

from hw2.task import SwingUpTask, wrap_to_pi


def test_wrap_to_pi_values():
    assert wrap_to_pi(0.0) == 0.0
    assert wrap_to_pi(np.pi) == np.pi  # +-pi are left untouched
    assert wrap_to_pi(-np.pi) == -np.pi
    assert wrap_to_pi(2 * np.pi + 0.1) == pytest.approx(0.1)
    assert wrap_to_pi(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    arr = wrap_to_pi(np.array([0.0, 2 * np.pi + 0.5]))
    assert arr == pytest.approx([0.0, 0.5])


def test_reset_starts_near_downward():
    env = SwingUpTask(seed=0)
    state, info = env.reset()
    assert state.shape == (4,)
    assert abs(abs(wrap_to_pi(state[2])) - np.pi) <= 0.05 + 1e-12
    assert abs(state[0]) <= 0.05
    assert state[1] == 0.0 and state[3] == 0.0


def test_reset_is_reproducible_with_seed():
    env = SwingUpTask()
    a, _ = env.reset(seed=3)
    b, _ = env.reset(seed=3)
    assert np.array_equal(a, b)


def test_action_torque_mapping():
    env = SwingUpTask(seed=0)
    env.reset()
    for idx, torque in enumerate((-0.02, 0.0, 0.02)):
        env.reset()
        _, _, _, _, info = env.step(idx)
        assert info["torque"] == torque


def test_outside_upright_region_does_not_terminate():
    env = SwingUpTask(seed=0)
    env.reset()
    _, _, terminated, truncated, info = env.step(1)
    assert abs(info["theta_wrapped"]) > np.deg2rad(20.0)
    assert terminated is False and truncated is False
    assert info["outcome"] is None


def test_success_after_five_seconds_upright():
    env = SwingUpTask()
    env.reset(initial_state=np.zeros(4))  # exact equilibrium stays put at zero torque
    assert env.success_steps == 500
    for _ in range(499):
        _, _, terminated, truncated, info = env.step(1)
        assert not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(1)
    assert terminated and not truncated
    assert info["outcome"] == "success"
    assert reward == pytest.approx(1.0)


def test_unsafe_when_arm_exceeds_limit():
    env = SwingUpTask()
    env.reset(initial_state=np.array([3.14, 5.0, np.pi, 0.0]))
    _, reward, terminated, truncated, info = env.step(1)
    assert terminated and not truncated
    assert info["outcome"] == "unsafe"
    assert reward == -10.0


def test_timeout_truncation():
    env = SwingUpTask(max_steps=50, seed=0)
    env.reset()
    for _ in range(49):
        _, _, terminated, truncated, _ = env.step(1)
        assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(1)
    assert truncated and not terminated
    assert info["outcome"] == "timeout"


def test_reward_range_downward_and_invalid_action():
    env = SwingUpTask(seed=0)
    env.reset()
    _, reward, _, _, _ = env.step(1)
    assert 0.0 <= reward <= 0.01  # (1 + cos(~pi)) / 2 ~ 0, no torque penalty
    with pytest.raises(ValueError):
        env.step(3)


def test_reward_nonnegative_unless_unsafe_and_unsafe_is_minus_ten():
    env = SwingUpTask(seed=0, max_steps=300)
    rng = np.random.default_rng(0)
    for ep in range(3):
        env.reset(seed=ep)
        for _ in range(300):
            _, reward, terminated, truncated, info = env.step(int(rng.integers(3)))
            if info["outcome"] == "unsafe":
                assert reward == -10.0
            else:
                assert reward >= -1e-6
            if terminated or truncated:
                break
    env.reset(initial_state=np.array([3.14, 5.0, np.pi, 0.0]))
    _, reward, _, _, info = env.step(2)
    assert info["outcome"] == "unsafe" and reward == -10.0


def test_reward_upright_is_one_and_bottom_is_zero():
    env = SwingUpTask()
    env.reset(initial_state=np.zeros(4))
    assert env.step(1)[1] == pytest.approx(1.0)
    env.reset(initial_state=np.array([0.0, 0.0, np.pi, 0.0]))
    assert env.step(1)[1] == pytest.approx(0.0, abs=1e-6)


def test_upright_counter_resets_when_leaving_region():
    env = SwingUpTask()
    env.reset(initial_state=np.zeros(4))
    for _ in range(499):
        assert not env.step(1)[2]
    assert env.upright_steps == 499
    env.state = np.array([0.0, 0.0, 0.5, 0.0])  # outside the 20 deg region
    _, _, terminated, _, _ = env.step(1)
    assert not terminated and env.upright_steps == 0
    env.state = np.zeros(4)                      # re-enter
    for k in range(499):
        _, _, terminated, _, _ = env.step(1)
        assert not terminated and env.upright_steps == k + 1
    _, _, terminated, _, info = env.step(1)
    assert terminated and info["outcome"] == "success"


def test_task_ignores_env_termination_from_downward_start():
    task = SwingUpTask(seed=0)
    task.reset()
    _, reward, terminated, truncated, info = task.step(1)
    assert info["env_terminated"] is True and info["env_reward"] == -10.0
    assert terminated is False and truncated is False
    assert info["outcome"] is None
    assert 0.0 <= reward <= 0.01


def test_task_wraps_unmodified_env_and_exposes_attributes():
    from furuta_env import FurutaEnv

    task = SwingUpTask(seed=0)
    assert type(task.env) is FurutaEnv
    assert task.dt == 0.01 and task.number_of_actions == 3 and task.max_steps == 2000
    assert task.success_steps == 500
    s, _ = task.reset()
    assert np.array_equal(task.state, s) and task.steps == 0


def test_task_reward_and_outcome_logic():
    task = SwingUpTask()
    task.reset(initial_state=np.array([0.0, 0.0, 0.0, 0.0]))
    _, reward, _, _, info = task.step(1)
    assert reward == pytest.approx(1.0) and info["env_reward"] != -10.0
    task.reset(initial_state=np.array([0.0, 0.0, 0.0, 0.0]))
    _, reward, _, _, info = task.step(2)   # torque penalty only
    assert reward == pytest.approx(1.0 - 0.001 * 0.02**2, abs=1e-5)
    task.reset(initial_state=np.array([3.14, 5.0, np.pi, 0.0]))
    _, reward, terminated, truncated, info = task.step(1)
    assert reward == -10.0 and terminated and not truncated and info["outcome"] == "unsafe"
    # theta beyond +-pi wraps for the reward and the upright test
    task.reset(initial_state=np.array([0.0, 0.0, 2 * np.pi, 0.0]))
    _, reward, _, _, info = task.step(1)
    assert reward == pytest.approx(1.0) and abs(info["theta_wrapped"]) < 1e-6
