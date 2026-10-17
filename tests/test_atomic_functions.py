import numpy as np

from spinnme.train.atomic_functions import compute_zeeman_strength, get_zeeman_lookup_id, load_zeeman_lookup, threej


def test_effective_triplet_components_have_unit_strength():
    for magnetic_quantum_number in (-1, 0, 1):
        np.testing.assert_allclose(
            compute_zeeman_strength(1, 0, magnetic_quantum_number, 0), 1.0, atol=1e-12
        )


def test_threej_sums_nontrivial_valid_range():
    np.testing.assert_allclose(abs(threej(1, 1, 1, 1, -1, 0)), 1 / np.sqrt(6), atol=1e-12)


def test_zeeman_lookup_key_is_independent_of_scalar_type():
    lookup = load_zeeman_lookup(1, 0)
    assert get_zeeman_lookup_id(np.float32(1), np.float32(0), np.float32(1), np.float32(0)) in lookup
