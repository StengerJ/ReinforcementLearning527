import sys
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "futura_env"))
from furuta_env import FurutaEnv, discretize_state

TS = 0.01
N_ACTIONS = 3
UPRIGHT_RAD = np.deg2rad(20.0)
HOLD_STEPS = int(round(5.0 / TS))


def wrap_to_pi(angle):
    angle = float(angle)
    if -np.pi <= angle <= np.pi:
        return angle
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


SMALL_SWING_PERIOD = 0.91
OMEGA = 2.0 * np.pi / SMALL_SWING_PERIOD


def pendulum_energy(state):
    return 0.5 * (state[3] / OMEGA) ** 2 - (1.0 - np.cos(wrap_to_pi(state[2])))


class Discretizer:
    def __init__(self, bins):
        self.bins = tuple(np.asarray(edges, dtype=float) for edges in bins)
        self.shape = tuple(len(edges) + 1 for edges in self.bins)
        self.n_states = int(np.prod(self.shape))
        self._edges = [edges.tolist() for edges in self.bins]
        self._strides = [int(np.prod(self.shape[i + 1:])) for i in range(4)]

    def bins_of(self, state):
        values = (float(state[0]), float(state[1]), wrap_to_pi(state[2]), float(state[3]))
        return tuple(bisect_right(edges, v) for edges, v in zip(self._edges, values))

    def reference_bins(self, state):
        s = np.array(state, dtype=float)
        s[2] = wrap_to_pi(s[2])
        return discretize_state(s, self.bins)

    def index(self, state):
        return sum(b * k for b, k in zip(self.bins_of(state), self._strides))


BALANCE_BINS = (
    [-1.0, -0.3, 0.3, 1.0],
    [-1.5, -0.3, 0.0, 0.3, 1.5],
    np.deg2rad([-12.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 12.0]),
    [-2.0, -1.0, -0.4, -0.1, 0.0, 0.1, 0.4, 1.0, 2.0],
)

SWINGUP_BINS = (
    [-2.0, -1.0, -0.3, 0.3, 1.0, 2.0],
    [-3.0, -1.5, -0.3, 0.0, 0.3, 1.5, 3.0],
    np.deg2rad([-135.0, -90.0, -45.0, -20.0, -12.0, -6.0, -3.0, -1.0, 0.0,
                1.0, 3.0, 6.0, 12.0, 20.0, 45.0, 90.0, 135.0]),
    [-8.0, -4.0, -2.0, -1.0, -0.4, -0.1, 0.0, 0.1, 0.4, 1.0, 2.0, 4.0, 8.0],
)


@dataclass(frozen=True)
class Task:
    name: str
    swing_up: bool
    bins: tuple
    max_steps: int = 2000
    lambda_phi: float = 0.1
    lambda_u: float = 1.0
    lambda_phidot: float = 0.0
    success_bonus: float = 100.0
    unsafe_penalty: float = -200.0
    init_angle_deg: float = 5.0
    init_rate: float = 0.05
    decision_steps: int = 1
    upright_start_fraction: float = 0.0
    fine_control_deg: float = 0.0
    handoff: tuple = None
    start_box: tuple = None
    start_box_fraction: float = 0.0
    reward_kind: str = "baseline"

    def make_env(self, seed):
        if self.swing_up:
            return FurutaEnv(seed=seed, max_steps=self.max_steps, dt=TS, angle_limit=np.inf)
        return FurutaEnv(seed=seed, max_steps=self.max_steps, dt=TS)

    def in_handoff(self, state):
        if self.handoff is None:
            return False
        arm_ok = len(self.handoff) < 3 or abs(state[1]) <= self.handoff[2]
        return abs(wrap_to_pi(state[2])) <= np.deg2rad(self.handoff[0]) and abs(state[3]) <= self.handoff[1] and arm_ok

    def reset(self, env, rng, training=False):
        if not self.swing_up:
            if training and self.start_box and rng.random() < self.start_box_fraction:
                phi, phi_dot, theta_deg, theta_dot = (rng.uniform(-w, w) for w in self.start_box)
                return env.reset(initial_state=[phi, phi_dot, np.deg2rad(theta_deg), theta_dot])[0]
            return env.reset(seed=int(rng.integers(2**31)))[0]
        if training and rng.random() < self.upright_start_fraction:
            return env.reset(seed=int(rng.integers(2**31)))[0]
        alpha = np.deg2rad(rng.uniform(-self.init_angle_deg, self.init_angle_deg))
        rates = rng.uniform(-self.init_rate, self.init_rate, size=2)
        return env.reset(initial_state=[0.0, rates[0], np.pi + alpha, rates[1]])[0]

    def reward(self, env_reward, next_state, torque, success, unsafe):
        if not self.swing_up:
            return env_reward
        if unsafe:
            return self.unsafe_penalty
        if self.reward_kind == "energy":
            shape = -abs(pendulum_energy(next_state))
        else:
            shape = np.cos(wrap_to_pi(next_state[2]))
        r = (shape - self.lambda_phi * next_state[0] ** 2 - self.lambda_phidot * next_state[1] ** 2
             - self.lambda_u * torque**2)
        return r + self.success_bonus if success else r

    def hold_steps(self, state):
        if abs(wrap_to_pi(state[2])) < np.deg2rad(self.fine_control_deg):
            return 1
        return self.decision_steps

    def ends_on_success(self):
        return self.swing_up


BALANCE = Task("balance", swing_up=False, bins=BALANCE_BINS)
BALANCE_AGENT = Task("balance_agent", swing_up=False, bins=BALANCE_BINS, start_box=(0.5, 4.0, 12.0, 2.0),
                     start_box_fraction=0.5)
SWINGUP = Task("swingup", swing_up=True, bins=SWINGUP_BINS, decision_steps=5, fine_control_deg=25.0)
SWINGUP_AGENT = replace(SWINGUP, name="swingup_agent", handoff=(15.0, 1.5), reward_kind="energy")


@dataclass(frozen=True)
class Hyper:
    episodes: int = 3000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_fraction: float = 0.7
    q_init: float = 0.0

    def epsilon(self, episode):
        decay = max(1, int(self.epsilon_decay_fraction * self.episodes))
        frac = min(episode / decay, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


def greedy_actions(q_row):
    best = max(q_row)
    return [a for a in range(len(q_row)) if q_row[a] == best]


def epsilon_greedy(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    ties = greedy_actions(q_row)
    return ties[0] if len(ties) == 1 else int(ties[rng.integers(len(ties))])


def sarsa_target(reward, gamma, q_next_row, next_action, terminal):
    if terminal:
        return reward
    return reward + gamma * q_next_row[next_action]


def qlearning_target(reward, gamma, q_next_row, terminal):
    if terminal:
        return reward
    return reward + gamma * max(q_next_row)


@dataclass
class EpisodeLog:
    ret: float = 0.0
    steps: int = 0
    outcome: str = "time-limit"
    first_upright_step: int = -1
    max_hold: int = 0
    closest_deg: float = 180.0
    downward_start: bool = True
    theta: list = field(default_factory=list)


class Episode:
    def __init__(self, task, env, rng, record=False, training=False):
        self.task, self.env = task, env
        self.state = task.reset(env, rng, training)
        self.hold = 0
        self.log = EpisodeLog(downward_start=abs(wrap_to_pi(self.state[2])) > np.pi / 2)
        self.record = record
        if record:
            self.log.theta.append(wrap_to_pi(self.state[2]))

    def step(self, action):
        total = 0.0
        for _ in range(self.task.hold_steps(self.state)):
            next_state, reward, terminal, done = self.sim_step(action)
            total += reward
            if done:
                break
        return next_state, total, terminal, done

    def sim_step(self, action):
        next_state, env_reward, terminated, truncated, info = self.env.step(action)
        theta = wrap_to_pi(next_state[2])
        upright = abs(theta) <= UPRIGHT_RAD
        self.hold = self.hold + 1 if upright else 0
        log = self.log
        log.steps += 1
        if upright and log.first_upright_step < 0:
            log.first_upright_step = log.steps
        log.max_hold = max(log.max_hold, self.hold)
        log.closest_deg = min(log.closest_deg, abs(np.rad2deg(theta)))
        if self.task.handoff is not None:
            success = self.task.in_handoff(next_state)
        else:
            success = self.hold >= HOLD_STEPS and self.task.ends_on_success()
        reward = self.task.reward(env_reward, next_state, info["torque"], success, terminated)
        log.ret += reward
        if self.record:
            log.theta.append(theta)
        terminal = terminated or success
        if terminated:
            log.outcome = "unsafe" if self.task.swing_up else "fell"
        elif success:
            log.outcome = "success"
        elif truncated:
            log.outcome = "time-limit"
        self.state = next_state
        return next_state, reward, terminal, terminal or truncated


def run_sarsa_episode(q, task, disc, env, rng, hyper, epsilon, visits):
    ep = Episode(task, env, rng, training=True)
    s = disc.index(ep.state)
    a = epsilon_greedy(q[s], epsilon, rng)
    while True:
        next_state, r, terminal, done = ep.step(a)
        s2 = disc.index(next_state)
        a2 = epsilon_greedy(q[s2], epsilon, rng)
        target = sarsa_target(r, hyper.gamma, q[s2], a2, terminal)
        q[s, a] += hyper.alpha * (target - q[s, a])
        visits[s, a] += 1
        if done:
            return ep.log
        s, a = s2, a2


def run_qlearning_episode(q, task, disc, env, rng, hyper, epsilon, visits):
    ep = Episode(task, env, rng, training=True)
    s = disc.index(ep.state)
    while True:
        a = epsilon_greedy(q[s], epsilon, rng)
        next_state, r, terminal, done = ep.step(a)
        s2 = disc.index(next_state)
        target = qlearning_target(r, hyper.gamma, q[s2], terminal)
        q[s, a] += hyper.alpha * (target - q[s, a])
        visits[s, a] += 1
        if done:
            return ep.log
        s = s2


EPISODE_RUNNERS = {"sarsa": run_sarsa_episode, "qlearning": run_qlearning_episode}


def train(method, task, hyper, seed, progress=None):
    disc = Discretizer(task.bins)
    rng = np.random.default_rng(seed)
    env = task.make_env(seed)
    q = np.full((disc.n_states, N_ACTIONS), float(hyper.q_init))
    visits = np.zeros((disc.n_states, N_ACTIONS), dtype=np.int64)
    run_episode = EPISODE_RUNNERS[method]
    n = hyper.episodes
    hist = {k: np.zeros(n) for k in ("returns", "durations", "success", "max_hold", "closest_deg", "downward_start",
                                     "coverage")}
    for ep in range(n):
        log = run_episode(q, task, disc, env, rng, hyper, hyper.epsilon(ep), visits)
        hist["returns"][ep] = log.ret
        hist["durations"][ep] = log.steps * TS
        hist["success"][ep] = float(log.outcome == "success" or log.max_hold >= HOLD_STEPS)
        hist["max_hold"][ep] = log.max_hold * TS
        hist["closest_deg"][ep] = log.closest_deg
        hist["downward_start"][ep] = float(log.downward_start)
        hist["coverage"][ep] = np.count_nonzero(visits) / visits.size
        if progress and (ep + 1) % progress == 0:
            lo = max(0, ep + 1 - progress)
            print(f"[{task.name}/{method}/seed{seed}] ep {ep + 1}: "
                  f"ret {hist['returns'][lo:ep + 1].mean():.1f} "
                  f"succ {hist['success'][lo:ep + 1].mean():.2f} "
                  f"hold {hist['max_hold'][lo:ep + 1].mean():.2f}s "
                  f"closest {hist['closest_deg'][lo:ep + 1].mean():.1f}deg", flush=True)
    return q, visits, hist


def evaluate(q, task, eval_seeds):
    disc = Discretizer(task.bins)
    frozen = q.copy()
    frozen.setflags(write=False)
    episodes = []
    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        env = task.make_env(seed)
        ep = Episode(task, env, rng, record=True)
        done = False
        while not done:
            a = epsilon_greedy(frozen[disc.index(ep.state)], 0.0, rng)
            _, _, _, done = ep.step(a)
        episodes.append(ep.log)
    return episodes


class TwoAgentController:
    def __init__(self, q_swing, q_balance, swing_task, balance_task):
        self.q_swing, self.q_balance = q_swing, q_balance
        self.swing_task, self.balance_task = swing_task, balance_task
        self.d_swing, self.d_balance = Discretizer(swing_task.bins), Discretizer(balance_task.bins)
        self.mode = "swing-up"
        self.handoffs = 0

    def choose(self, state, rng):
        if self.mode == "balance" and abs(wrap_to_pi(state[2])) > UPRIGHT_RAD:
            self.mode = "swing-up"
        elif self.mode == "swing-up" and self.swing_task.in_handoff(state):
            self.mode = "balance"
            self.handoffs += 1
        if self.mode == "balance":
            return epsilon_greedy(self.q_balance[self.d_balance.index(state)], 0.0, rng), 1
        a = epsilon_greedy(self.q_swing[self.d_swing.index(state)], 0.0, rng)
        return a, self.swing_task.hold_steps(state)

    def choose_balance(self, state, rng):
        return epsilon_greedy(self.q_balance[self.d_balance.index(state)], 0.0, rng)

    def should_switch(self, state):
        return self.mode == "swing-up" and self.swing_task.in_handoff(state)


def evaluate_combined(q_swing, q_balance, swing_task, balance_task, eval_seeds):
    task = replace(swing_task, handoff=None, upright_start_fraction=0.0)
    episodes = []
    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        env = task.make_env(seed)
        ep = Episode(task, env, rng, record=True)
        ctrl = TwoAgentController(q_swing, q_balance, swing_task, balance_task)
        done = False
        while not done:
            a, n = ctrl.choose(ep.state, rng)
            for _ in range(n):
                _, _, _, done = ep.sim_step(a)
                if done or ctrl.should_switch(ep.state):
                    break
        ep.log.handoffs = ctrl.handoffs
        episodes.append(ep.log)
    return episodes
