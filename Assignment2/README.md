# COSC 527 HW2 — Building and Verifying a Tabular RL Agent (Joshua Stenger)

The professor-supplied environment in `futura_env/` is used **exactly as given**
(no edits, no wrapper). Every script imports `FurutaEnv` / `discretize_state`
from `futura_env/furuta_env.py` directly.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run (from this folder)

| Script | HW part | Output |
|---|---|---|
| `python stenger_hw2_test_environment.py` | 1. environment & episode logic | `results/part1_environment.json` |
| `python stenger_hw2_state_discretization.py` | 2. discretizer + 10 boundary tests | `results/part2_discretization.json` |
| `python stenger_hw2_epsilon_greedy.py` | 3. ε-greedy frequency test (seed 527) | `results/part3_epsilon_greedy.json` |
| `python stenger_hw2_audit.py` | 4. SARSA / Q-learning update audit | `results/part4_audit.json` |
| `python stenger_hw2_experiment.py` | 5. Q-learning, seeds 0,1,2 (a few minutes) | `results/part5_*.npz`, `results/part5_summary.json`, `report/figures/*.png` |
| `python stenger_hw2_verification.py` | automated assert checks of all components | console PASS/FAIL |

All runs are seeded, so re-running reproduces the reported numbers and figures.

## Report

`report/report.tex` (figures in `report/figures/`). Build with
`pdflatex report.tex` run twice from inside `report/`, or upload the `report/` folder to Overleaf.
