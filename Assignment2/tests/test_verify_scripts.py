import json

import pytest

import audit_updates
import io_utils
import verify_discretizer
import verify_env
import verify_epsilon_greedy


@pytest.fixture(autouse=True)
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(io_utils, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_verify_env(tmp_results):
    out = verify_env.main()
    assert len(out["actions"]) == 3
    assert out["outside_upright"]["terminated"] is False
    assert out["outcomes"]["success"]["outcome"] == "success"
    assert out["outcomes"]["unsafe"]["outcome"] == "unsafe"
    assert out["outcomes"]["timeout"]["outcome"] == "timeout"
    assert json.loads((tmp_results / "verify_env.json").read_text())["dt"] == 0.01


def test_verify_discretizer():
    out = verify_discretizer.main()
    assert out["n_states"] == 2275 and out["n_q"] == 6825
    assert len(out["tests"]) >= 8
    assert all(0 <= t["index"] < out["n_states"] for t in out["tests"])


def test_verify_epsilon_greedy():
    out = verify_epsilon_greedy.main()
    assert out["seed"] == 0 and out["n_samples"] == 10_000
    eps0 = [r for r in out["runs"] if r["epsilon"] == 0.0][0]
    assert eps0["observed"][0] == 0.0


def test_audit_updates():
    out = audit_updates.main()
    nt, t = out["nonterminal"], out["terminal"]
    assert nt["y_sarsa"] != nt["y_q"]
    assert nt["s_t"] != nt["s_t1"]
    assert t["y_sarsa"] == t["y_q"] == t["r_t1"]
