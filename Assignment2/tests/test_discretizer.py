import itertools

import numpy as np
import pytest

from hw2.tabular import Discretizer


@pytest.fixture
def disc():
    return Discretizer()


def test_shape_and_counts(disc):
    assert disc.shape == (5, 5, 13, 7)
    assert disc.n_states == 2275


@pytest.mark.parametrize(
    "state,expected",
    [
        ([0, 0, 0, 0], (2, 2, 6, 3)),            # upright
        ([0, 0, np.pi, 0], (2, 2, 12, 3)),       # downward (+pi)
        ([0, 0, -np.pi, 0], (2, 2, 0, 3)),       # downward (-pi)
        ([0, 0, np.pi - 0.01, 0], (2, 2, 12, 3)),
        ([0, 0, -np.pi + 0.01, 0], (2, 2, 0, 3)),
        ([0, 0, np.pi + 0.01, 0], (2, 2, 0, 3)),  # unwrapped angle wraps to -pi+0.01
        ([0, -100.0, 0, 0], (2, 0, 6, 3)),       # phi_dot below range
        ([0, 100.0, 0, 0], (2, 4, 6, 3)),        # phi_dot above range
        ([0, 0, 0, -100.0], (2, 2, 6, 0)),       # theta_dot below range
        ([0, 0, 0, 100.0], (2, 2, 6, 6)),        # theta_dot above range
        ([4.0, 0, 0, 0], (4, 2, 6, 3)),          # phi above range
    ],
)
def test_known_tuples(disc, state, expected):
    assert disc.indices(state) == expected


def test_bin_boundary_goes_to_upper_bin(disc):
    edge = disc.bins[2][6]  # +20 deg
    assert disc.indices([0, 0, edge, 0])[2] == 7
    assert disc.indices([0, 0, edge - 1e-9, 0])[2] == 6


def test_wrapped_equivalence_and_huge_values(disc):
    assert disc.index([0, 0, 7 * np.pi + 0.2, 0]) == disc.index([0, 0, -np.pi + 0.2, 0])
    idx = disc.index([1e6, 1e9, 1e6, -1e9])
    assert 0 <= idx < disc.n_states


def test_non_finite_state_raises(disc):
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            disc.index([0, 0, bad, 0])


def test_wrong_shape_raises(disc):
    with pytest.raises(ValueError):
        disc.index([0, 0, 0])


def test_flat_index_is_a_bijection(disc):
    seen = set()
    for tup in itertools.product(*(range(n) for n in disc.shape)):
        flat = int(np.ravel_multi_index(tup, disc.shape))
        assert disc.unravel(flat) == tup
        seen.add(flat)
    assert seen == set(range(disc.n_states))
