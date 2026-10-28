"""Physical levels and independently tabulated Zeeman patterns for Hinode/SP."""

import numpy as np
import pytest
import torch
from astropy import units as u

from spinnme.configuration import read_config
from spinnme.instrument import SPECTRAL_LINES, resolve_instrument_line
from spinnme.train.me_atmosphere import MEAtmosphere


@pytest.mark.parametrize('line_id,center,expected', [
    ('fe-i-6301', 6301.5012, {'j_low': 2., 'j_up': 2., 'g_low': 11 / 6, 'g_up': 1.5}),
    ('fe-i-6302', 6302.4931, {'j_low': 1., 'j_up': 0., 'g_low': 2.49, 'g_up': 0.}),
])
def test_hinode_preset_resolves_physical_levels(line_id, center, expected):
    config = read_config('hinode/sp')
    instrument = next(row for row in config['instrument'] if row['line_id'] == line_id)
    dataset = next(row for row in config['data']['train_configs'] if row['line_id'] == line_id)
    assert dataset['wavelength_center'] == center
    resolved, atomic = resolve_instrument_line(
        instrument, {'wavelength_center': dataset['wavelength_center'] * u.AA})
    assert resolved == line_id
    assert atomic == expected


@pytest.mark.parametrize('line_id,expected', [
    ('fe-i-6301', {
        'b': ([7 / 6, 3 / 2, 11 / 6, 13 / 6], [.2, .3, .3, .2]),
        'p': ([-2 / 3, -1 / 3, 0, 1 / 3, 2 / 3], [.4, .1, 0, .1, .4]),
        'r': ([-13 / 6, -11 / 6, -3 / 2, -7 / 6], [.2, .3, .3, .2]),
    }),
    ('fe-i-6302', {'b': ([2.49], [1.]), 'p': ([0.], [1.]), 'r': ([-2.49], [1.])}),
])
def test_hinode_full_zeeman_patterns(line_id, expected):
    atmosphere = MEAtmosphere(**SPECTRAL_LINES[line_id]).double()
    for component, (splittings, strengths) in expected.items():
        indices = [i for i, (name, _) in enumerate(atmosphere._transitions) if name == component]
        actual_strengths = [float(atmosphere.zeeman_strength_lookup[atmosphere._transitions[i][1]])
                            for i in indices]
        np.testing.assert_allclose(atmosphere._transition_splittings[indices], splittings, atol=3e-7)
        np.testing.assert_allclose(actual_strengths, strengths, atol=2e-8)
        np.testing.assert_allclose(sum(actual_strengths), 1., atol=3e-8)
        effective_g = 5 / 3 if line_id == 'fe-i-6301' else 2.49
        centroid = np.dot(splittings, actual_strengths)
        np.testing.assert_allclose(centroid, {'b': effective_g, 'p': 0., 'r': -effective_g}[component],
                                   atol=5e-8)


def test_physical_6302_levels_preserve_equivalent_triplet_profiles():
    physical = MEAtmosphere(**SPECTRAL_LINES['fe-i-6302']).double()
    equivalent = MEAtmosphere(wavelength_center=6302.4931 * u.AA,
                              j_up=1., j_low=0., g_up=2.49, g_low=0.).double()
    inputs = {'nu': torch.linspace(-10, 10, 101, dtype=torch.float64)[None],
              'nu_m': torch.tensor([[.7]], dtype=torch.float64),
              'damping': torch.tensor([[.2]], dtype=torch.float64),
              'lambda_dop': torch.tensor([[.3]], dtype=torch.float64)}
    actual = physical.calculate_voigt_faraday_profiles(**inputs)
    previous = equivalent.calculate_voigt_faraday_profiles(**inputs)
    for key in actual:
        torch.testing.assert_close(actual[key], previous[key], rtol=0, atol=0)


def test_6302_state_round_trip_rebuilds_derived_lookup():
    original = MEAtmosphere(**SPECTRAL_LINES['fe-i-6302']).double()
    restored = MEAtmosphere(**SPECTRAL_LINES['fe-i-6302']).double()
    # Exercise recursive loading with the same prefix as an inversion snapshot.
    parent = torch.nn.ModuleDict({'forward_models': torch.nn.ModuleDict({'HINODE_6302': restored})})
    prefix = 'forward_models.HINODE_6302.'
    state = {prefix + key: value for key, value in original.state_dict().items()}
    assert not any('zeeman_strength_lookup' in key for key in state)
    assert not list(original.zeeman_strength_lookup.parameters())
    parent.load_state_dict(state, strict=True)
    assert set(restored.state_dict()) == set(original.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    assert restored._transitions == original._transitions
    torch.testing.assert_close(restored._transition_splittings, original._transition_splittings)
    inputs = {'nu': torch.linspace(-10, 10, 101, dtype=torch.float64)[None],
              'nu_m': torch.tensor([[.7]], dtype=torch.float64),
              'damping': torch.tensor([[.2]], dtype=torch.float64),
              'lambda_dop': torch.tensor([[.3]], dtype=torch.float64)}
    for key, expected in original.calculate_voigt_faraday_profiles(**inputs).items():
        torch.testing.assert_close(restored.calculate_voigt_faraday_profiles(**inputs)[key], expected,
                                   rtol=0, atol=0)


def test_atomic_state_loading_remains_strict():
    atmosphere = MEAtmosphere(**SPECTRAL_LINES['fe-i-6302'])
    state = atmosphere.state_dict()
    state.pop('g_up')
    state['unrecognized'] = torch.tensor(1.)
    with pytest.raises(RuntimeError, match='Missing key'):
        atmosphere.load_state_dict(state, strict=True)
