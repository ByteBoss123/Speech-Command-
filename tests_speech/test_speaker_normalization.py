import numpy as np
import pytest
from speechlab.speaker_normalized import apply_normalizer, choose, fit_normalizer, METHODS


def test_regularized_within_speaker_whitening_is_finite_and_unit_length():
    x = np.array([[-3., -1.], [-3., 1.], [3., -2.], [3., 2.]])
    params = fit_normalizer(x, ['a', 'a', 'b', 'b'], .5)
    # Within-class covariance is diag(0, 5); 50% trace shrinkage makes diag(1.25, 3.75).
    transformed_covariance = params['transform'].T @ np.diag([1.25, 3.75]) @ params['transform']
    assert np.allclose(transformed_covariance, np.eye(2), atol=1e-6)
    result = apply_normalizer(x, params)
    assert np.allclose(np.linalg.norm(result, axis=1), 1)
    assert np.allclose(params['mean'], [0, 0])


def test_normalizer_rejects_missing_repeats_singular_configuration_and_nonfinite():
    with pytest.raises(ValueError, match='Repeated'):
        fit_normalizer(np.eye(2), ['a', 'b'], .5)
    with pytest.raises(ValueError, match='Invalid'):
        fit_normalizer(np.ones((4, 2)), ['a', 'a', 'b', 'b'], 0)
    with pytest.raises(ValueError, match='Nonfinite'):
        fit_normalizer(np.full((4, 2), np.nan), ['a', 'a', 'b', 'b'], .5)


def test_normalization_selection_respects_cap_and_prefers_original_on_ties():
    values = {name: {'far': .004, 'frr': .13} for name in METHODS}
    assert choose(values) == 'five_raw'
    values['wccn_0.5'] = {'far': .004, 'frr': .08}
    assert choose(values) == 'wccn_0.5'
    values['wccn_0.5']['far'] = .006
    assert choose(values) == 'five_raw'
