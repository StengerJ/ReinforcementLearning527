# COSC 527 HW2: Tabular Q-learning on the Furuta pendulum (swing-up)

The professor-supplied files in `starter/` are used **unmodified**. Our code
wraps them: `hw2/task.py` defines `SwingUpTask`, which uses the supplied
`FurutaEnv` only for the dynamics and builds the swing-up episodes
(downward start, outcomes, reward) around it.

## Layout

```text
docs/                 HW2_Tabular_RL_Checkpoint.pdf (assignment)
starter/              professor files, byte-identical (furuta_env.py, example_run.py,
                      model_reference.md, README.md, requirements.txt)
hw2/                  our package
  task.py             SwingUpTask wrapper + wrap_to_pi
  tabular.py          discretizer, epsilon-greedy, TD updates
  training.py         episode / train / evaluate loops, classifier
  io_utils.py         paths and JSON helper
scripts/              entry points (run from this folder as python -m scripts.<name>)
  verify_env.py verify_discretizer.py verify_epsilon_greedy.py
  audit_updates.py run_experiment.py make_report.py
tests/                pytest suite (includes a hash check of starter/)
results/              generated data (JSON, npz)
report/template/      main_template.tex (hand-edited source)
report/main.tex       generated single-file report for Overleaf
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
```

## Run order (from this folder)

1. `python -m pytest -q` : all tests
2. `python -m scripts.verify_env`, `scripts.verify_discretizer`,
   `scripts.verify_epsilon_greedy`, `scripts.audit_updates` : parts 1-4 (write `results/*.json`)
3. `python -m scripts.run_experiment` : part 5 (about 6 minutes; writes `results/experiment.{npz,json}`)
4. `python -m scripts.make_report` : builds the self-contained `report/main.tex`
   from `results/` and `report/template/main_template.tex`
5. Upload only `report/main.tex` to Overleaf (needs `pgfplots`). Re-running step 4 overwrites it.

## Notes

- `hw2/__init__.py` puts `starter/` on `sys.path`, so `from furuta_env import FurutaEnv, discretize_state` resolves to the supplied file.
- The supplied env is a balancing task (upright start, terminates at |theta| > 20 deg). `SwingUpTask` ignores its termination/truncation/reward and recomputes them from the state.
- Reward in the wrapper: `(1 + cos(theta_wrapped)) / 2 - 0.001 * torque^2` (non-negative living reward), and -10 on unsafe termination (arm past +-pi). An earlier `cos(theta)` reward was negative near the bottom, which made crashing into the arm limit cheaper than surviving.
