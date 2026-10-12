# COSC 527 Midterm Project A — Furuta Pendulum Swing-Up and Balancing (Joshua Stenger)

Builds on the HW2 tabular agent. The supplied simulator in `futura_env/` is used **exactly as given**: no
edits and no wrapper class. Every script imports `FurutaEnv` / `discretize_state` from
`futura_env/furuta_env.py` and only uses its public constructor arguments, `reset()` and `step()`.

Two stages are run, as instructed:

1. **Balance (no swing-up)** — `FurutaEnv()` with all supplied defaults: start near upright, built-in
   reward, termination at 20 degrees. Shows whether the agent can stabilize the pendulum at all.
2. **Swing-up and balance** — `FurutaEnv(angle_limit=inf)` so that leaving the upright region does not
   end the episode (the project PDF requires this), start near the downward position via
   `reset(initial_state=...)`, the project's baseline reward `cos(theta~) - lambda_phi phi^2 - lambda_u u^2`
   plus a success bonus, and success after 500 consecutive steps within 20 degrees. The supplied arm-angle
   safety limit and `max_steps = 2000` are kept.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run (from this folder)

| Command | Project part | Output |
|---|---|---|
| `python stenger_midterm_main.py` | demo: one greedy episode with a saved Q table, animated in pygame (`--stage balance\|swingup\|comparison`, `--method sarsa\|qlearning`, `--seed`, `--eval-seed`, `--speed`, `--no-render`); keys SPACE pause, +/- speed, R replay, ESC quit | window + one-line summary |
| `python stenger_midterm_test_environment.py` | 1. verify the environment | `results/part1_environment.txt` |
| `python stenger_midterm_discretization.py` | 2. bins, table size, boundary tests | `results/part2_discretization.txt` |
| `python stenger_midterm_verification.py` | automated checks of env use, bins, policy, SARSA/Q targets, reward, success counter | console PASS/FAIL |
| `python stenger_midterm_experiment.py` | 3–5. train SARSA and Q-learning (3 seeds each) for both stages and the controlled comparison, evaluate, plot | `results/*.npz` (learned Q tables + training curves), `results/summary.txt`, `report/figures/*.png` |
| `python stenger_midterm_experiment.py --summarize-only` | re-evaluate the saved Q tables and redraw the plots without retraining | same as above |
| `python stenger_midterm_experiment.py --only balance` | run a single experiment (`balance`, `swingup`, `swingup_comparison`) | |

The full experiment trains 18 runs in parallel (`--workers`, default 12; the reported run used
`--workers 15`) and took about 25 minutes on a 16-thread machine. All training and evaluation use fixed seeds (train seeds 0, 1, 2; evaluation seeds
10000–10019, identical for both methods), so reruns reproduce the reported numbers.

## Where things are

- `stenger_midterm_agent.py` — discretizer, epsilon-greedy policy, the separate SARSA and Q-learning
  episode loops (`sarsa_target` vs `qlearning_target`), training and frozen-table evaluation. All
  hyperparameters and bin edges are dataclass fields / module constants (`BALANCE_BINS`, `SWINGUP_BINS`,
  `Task`, `Hyper`) and the experiment settings live in `EXPERIMENTS` at the top of
  `stenger_midterm_experiment.py`.
- `report/report.tex` — written report (figures in `report/figures/`).
- `slides/slides.tex` — 10-minute Beamer presentation.

Build either PDF with `pdflatex` (run twice) from its folder, or upload the folder to Overleaf.
