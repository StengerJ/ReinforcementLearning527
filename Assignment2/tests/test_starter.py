"""The professor-supplied starter files must stay byte-identical to the originals."""

import hashlib
from pathlib import Path

import numpy as np

STARTER = Path(__file__).resolve().parent.parent / "starter"

# sha256 of the files exactly as supplied.
EXPECTED = {
    "furuta_env.py": "918ad469fd9fe4b2d8a6ee0898eacf0ff4c298c664248480570ca3df1196db3c",
    "example_run.py": "c0d29a7b5604d646ba34335fca3fdfacd6d7f9104dc9c4b651e11a543257f6f9",
    "model_reference.md": "dbff35fc922866165e2ae8e446203994e3382c24f788e6f1436042cbe3f94dc6",
    "README.md": "907606e21f30552affe358cb9e70c808742c5795ad460526930ecb4ef84d8b7e",
    "requirements.txt": "a83a2a4ed7ab2e022b9b8a83cfa2b8cf52cc243006c597e8b96f90a8365590ac",
}


def _digest(path):
    # Ignore CR so a git line-ending conversion cannot cause a false failure.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_starter_files_unmodified():
    for name, digest in EXPECTED.items():
        assert _digest(STARTER / name) == digest, f"{name} differs from the supplied original"


def test_raw_env_is_a_balancing_task():
    from furuta_env import FurutaEnv

    env = FurutaEnv(seed=0)
    state, _ = env.reset()
    assert abs(state[2]) <= 0.02            # upright start
    env.reset(initial_state=np.array([0.0, 0.0, np.pi, 0.0]))
    _, reward, terminated, _, _ = env.step(1)
    assert terminated and reward == -10.0   # downward start ends at once
