"""Part 5: Q-learning, 3 seeds. Saves results/experiment.{npz,json}."""

import time
from collections import Counter

import numpy as np

from furuta_env import FurutaEnv
from io_utils import RESULTS_DIR, save_json
from tabular import Discretizer
from training import train

CONFIG = dict(
    kind="qlearning", seeds=[0, 1, 2], n_episodes=2000, max_steps=1000,
    alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05,
    q_init=0.0, moving_window=50,
)


def main():
    cfg = CONFIG
    disc = Discretizer()
    cfg_out = dict(cfg, dt=FurutaEnv().dt, n_states=disc.n_states, n_actions=3,
                   eps_schedule="linear from eps_start to eps_end over episodes",
                   max_episode_seconds=cfg["max_steps"] * FurutaEnv().dt)
    arrays = {"returns": [], "durations": [], "visited_frac": [], "state_visits": [], "Q": []}
    seeds_out = []
    for seed in cfg["seeds"]:
        t0 = time.perf_counter()
        res = train(
            seed=seed, n_episodes=cfg["n_episodes"], max_steps=cfg["max_steps"],
            alpha=cfg["alpha"], gamma=cfg["gamma"], eps_start=cfg["eps_start"],
            eps_end=cfg["eps_end"], kind=cfg["kind"], disc=disc,
        )
        runtime = time.perf_counter() - t0
        for k in arrays:
            if k in res:
                arrays[k].append(res[k])
        arrays[f"eval_theta_{seed}"] = res["eval"]["theta_wrapped"]
        th = np.abs(res["eval"]["theta_wrapped"])
        w = cfg["moving_window"]
        seeds_out.append({
            "seed": seed,
            "final_moving_average": float(np.mean(res["returns"][-w:])),
            "mean_duration_last_window": float(np.mean(res["durations"][-w:])),
            "visited_fraction_final": float(res["visited_frac"][-1]),
            "outcome_counts": dict(Counter(map(str, res["outcomes"]))),
            "eval": {
                "outcome": res["eval"]["outcome"],
                "steps": res["eval"]["steps"],
                "classification": res["eval_class"],
                "min_abs_theta_deg": float(np.rad2deg(th.min())),
                "max_deviation_from_downward_deg": float(np.rad2deg((np.pi - th).max())),
            },
            "runtime_s": runtime,
        })
        print(f"seed {seed}: {runtime:.0f}s, eval={res['eval_class']}, "
              f"visited={res['visited_frac'][-1]:.3f}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / "experiment.npz",
        **{k: np.array(v) for k, v in arrays.items() if not k.startswith("eval_theta")},
        **{k: v for k, v in arrays.items() if k.startswith("eval_theta")},
    )
    save_json("experiment", {"config": cfg_out, "seeds": seeds_out})


if __name__ == "__main__":
    main()
