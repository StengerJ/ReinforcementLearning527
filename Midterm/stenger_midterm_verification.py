import numpy as np

from stenger_midterm_agent import (
    BALANCE, HOLD_STEPS, N_ACTIONS, SWINGUP, Discretizer, Episode, Hyper, epsilon_greedy, evaluate,
    qlearning_target, sarsa_target, train, wrap_to_pi)
from furuta_env import FurutaEnv


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def test_environment():
    env = FurutaEnv(seed=0)
    state, _ = env.reset()
    check("reset returns a 4-element state", state.shape == (4,))
    torques = []
    for a in range(env.number_of_actions):
        env.reset(initial_state=state)
        torques.append(env.step(a)[4]["torque"])
    check("actions 0,1,2 apply -0.02, 0, +0.02 N m", np.allclose(torques, [-0.02, 0.0, 0.02]))
    try:
        env.step(3)
        check("illegal action is rejected", False)
    except ValueError:
        check("illegal action is rejected", True)
    env.reset(initial_state=[0, 0, np.pi, 0])
    check("default (balance) env terminates outside 20 deg", env.step(1)[2])
    sw = SWINGUP.make_env(0)
    sw.reset(initial_state=[0, 0, np.pi, 0])
    check("swing-up env does not terminate outside the upright region", not any(sw.step(1)[2] for _ in range(300)))
    sw.reset(initial_state=[3.1, 5.0, np.pi, 0])
    check("swing-up env keeps the supplied arm-angle safety limit", sw.step(1)[2])
    short = FurutaEnv(max_steps=3, angle_limit=np.inf)
    short.reset(initial_state=[0, 0, np.pi, 0])
    check("time limit sets truncated, not terminated", [short.step(1)[2:4] for _ in range(3)][-1] == (False, True))
    rng = np.random.default_rng(3)
    starts = [wrap_to_pi(SWINGUP.reset(sw, rng)[2]) for _ in range(400)]
    check("swing-up starts within 5 deg of downward, on both sides of pi",
          all(abs(abs(t) - np.pi) <= np.deg2rad(5) + 1e-9 for t in starts) and min(starts) < 0 < max(starts))


def test_discretizer():
    for task in (BALANCE, SWINGUP):
        disc = Discretizer(task.bins)
        rng = np.random.default_rng(1)
        xs = rng.uniform(-50, 50, size=(5000, 4))
        check(f"{task.name}: fast bins equal the supplied discretize_state helper",
              all(disc.bins_of(x) == disc.reference_bins(x) for x in xs))
        flat = [disc.index(x) for x in xs]
        check(f"{task.name}: extreme states map to valid indices", min(flat) >= 0 and max(flat) < disc.n_states)
        check(f"{task.name}: flat index round-trips",
              all(np.unravel_index(disc.index(x), disc.shape) == disc.bins_of(x) for x in xs[:200]))
    disc = Discretizer(SWINGUP.bins)
    check("swing-up: upright and downward fall in different theta bins",
          disc.bins_of([0, 0, 0, 0])[2] != disc.bins_of([0, 0, np.pi, 0])[2])
    check("wrap_to_pi(3*pi/2) = -pi/2", np.isclose(wrap_to_pi(1.5 * np.pi), -0.5 * np.pi))


def test_epsilon_greedy():
    q = np.array([1.0, 2.0, 2.0])
    rng = np.random.default_rng(0)
    draws = [epsilon_greedy(q, 0.0, rng) for _ in range(3000)]
    check("eps=0 never picks a non-greedy action", 0 not in draws)
    check("eps=0 breaks ties uniformly between greedy actions", abs(draws.count(1) / 3000 - 0.5) < 0.04)
    counts = np.bincount([epsilon_greedy(q, 1.0, rng) for _ in range(30000)], minlength=3) / 30000
    check("eps=1 is uniform over all actions", np.allclose(counts, 1 / 3, atol=0.015))
    h = Hyper(episodes=100, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_fraction=0.5)
    check("epsilon schedule decays linearly then stays flat",
          h.epsilon(0) == 1.0 and np.isclose(h.epsilon(25), 0.55) and h.epsilon(50) == h.epsilon(99) and np.isclose(h.epsilon(99), 0.1))


def test_targets():
    q_next = np.array([1.0, 5.0, 3.0])
    check("SARSA target uses Q(S',A')", sarsa_target(1.0, 0.9, q_next, 2, False) == 1.0 + 0.9 * 3.0)
    check("Q-learning target uses max_a Q(S',a)", qlearning_target(1.0, 0.9, q_next, False) == 1.0 + 0.9 * 5.0)
    check("targets differ for the same transition when A' is not greedy",
          sarsa_target(1.0, 0.9, q_next, 2, False) != qlearning_target(1.0, 0.9, q_next, False))
    check("terminal continuation value is zero for both",
          sarsa_target(-5.0, 0.9, q_next * 1e6, 1, True) == -5.0 == qlearning_target(-5.0, 0.9, q_next * 1e6, True))


def test_reward_and_success():
    t = SWINGUP
    check("baseline reward is +1 upright and -1 downward",
          np.isclose(t.reward(0, [0, 0, 0, 0], 0.0, False, False), 1.0)
          and np.isclose(t.reward(0, [0, 0, np.pi, 0], 0.0, False, False), -1.0))
    check("reward penalizes arm angle and torque",
          np.isclose(t.reward(0, [1.0, 0, 0, 0], 0.02, False, False), 1 - t.lambda_phi - t.lambda_u * 0.02**2))
    check("success bonus and unsafe penalty",
          np.isclose(t.reward(0, [0, 0, 0, 0], 0, True, False), 1 + t.success_bonus)
          and t.reward(0, [0, 0, 0, 0], 0, False, True) == t.unsafe_penalty)
    check("control interval: every 10 ms near upright, every 50 ms during the swing",
          SWINGUP.hold_steps([0, 0, np.deg2rad(10), 0]) == 1 and SWINGUP.hold_steps([0, 0, np.pi, 0]) == 5
          and BALANCE.hold_steps([0, 0, np.pi, 0]) == 1)
    check("balance task keeps the supplied reward", BALANCE.reward(0.123, [0, 0, 3, 0], 0.02, False, False) == 0.123)

    class ScriptedEnv:
        def __init__(self, thetas):
            self.thetas, self.i, self.state = thetas, 0, np.zeros(4)

        def reset(self, initial_state=None, seed=None):
            self.state = np.array([0.0, 0.0, np.pi, 0.0])
            return self.state.copy(), {}

        def step(self, action):
            self.state = np.array([0.0, 0.0, self.thetas[self.i], 0.0])
            self.i += 1
            return self.state.copy(), 0.0, False, False, {"torque": 0.0}

    thetas = [0.0] * 300 + [1.0] + [2 * np.pi] * HOLD_STEPS
    ep = Episode(SWINGUP, ScriptedEnv(thetas), np.random.default_rng(0))
    flags = [ep.sim_step(1) for _ in thetas]
    check("hold counter resets on leaving upright and success needs 500 consecutive steps (wrapped theta)",
          not any(f[2] for f in flags[:-1]) and flags[-1][2] and ep.log.outcome == "success"
          and ep.log.first_upright_step == 1)


def test_training_and_evaluation():
    hyper = Hyper(episodes=3)
    q, visits, hist = train("sarsa", BALANCE, hyper, seed=0)
    check("training updates the Q table", np.any(q != hyper.q_init) and visits.sum() > 0)
    q2, _, _ = train("sarsa", BALANCE, hyper, seed=0)
    check("training is reproducible for a fixed seed", np.array_equal(q, q2))
    before = q.copy()
    logs = evaluate(q, BALANCE, [1, 2])
    check("evaluation leaves Q unchanged", np.array_equal(q, before) and len(logs) == 2)
    check("evaluation is deterministic for fixed seeds",
          [e.steps for e in logs] == [e.steps for e in evaluate(q, BALANCE, [1, 2])])
    check("Q table has one row per discrete state and one column per action",
          q.shape == (Discretizer(BALANCE.bins).n_states, N_ACTIONS))


def test_two_agent_controller():
    from stenger_midterm_agent import BALANCE_AGENT, SWINGUP_AGENT, TwoAgentController
    q_swing = np.zeros((Discretizer(SWINGUP_AGENT.bins).n_states, N_ACTIONS))
    q_bal = np.zeros((Discretizer(BALANCE_AGENT.bins).n_states, N_ACTIONS))
    ctrl = TwoAgentController(q_swing, q_bal, SWINGUP_AGENT, BALANCE_AGENT)
    rng = np.random.default_rng(0)
    _, n = ctrl.choose([0, 0, np.pi, 0], rng)
    check("hanging down: swing-up agent acts with a 50 ms hold", ctrl.mode == "swing-up" and n == 5)
    _, n = ctrl.choose([0, 0, np.deg2rad(10), 1.0], rng)
    check("inside the handoff region: balancing agent takes over at 10 ms", ctrl.mode == "balance" and n == 1
          and ctrl.handoffs == 1)
    ctrl.choose([0, 0, np.deg2rad(30), 0], rng)
    check("leaving the upright region hands control back to the swing-up agent", ctrl.mode == "swing-up")
    check("fast arrival (|theta_dot| > 1.5) is not handed off",
          not SWINGUP_AGENT.in_handoff([0, 0, np.deg2rad(5), 3.0]))
    check("energy reward is 0 at upright rest and -2 hanging at rest",
          np.isclose(SWINGUP_AGENT.reward(0, [0, 0, 0, 0], 0, False, False), 0.0)
          and np.isclose(SWINGUP_AGENT.reward(0, [0, 0, np.pi, 0], 0, False, False), -2.0))


if __name__ == "__main__":
    test_two_agent_controller()
    test_environment()
    test_discretizer()
    test_epsilon_greedy()
    test_targets()
    test_reward_and_success()
    test_training_and_evaluation()
    print("All checks passed.")
