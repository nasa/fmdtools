"""Regression tests for constant slices along the bootstrap sample axis."""

import numpy as np
import pytest
from scipy.stats import bootstrap

from fmdtools.analyze.common import calc_metric_ci
from fmdtools.analyze.history import History


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("statistic", [np.mean, np.sum])
def test_constant_slice_uses_basic_interval(axis, statistic):
    values = np.array([[0.0, 10.0], [1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])
    if axis != 0:
        values = values.T
    actual = calc_metric_ci(
        values,
        method=statistic,
        axis=axis,
        n_resamples=499,
        rng=np.random.default_rng(13),
    )
    expected = bootstrap(
        [values],
        statistic,
        axis=axis,
        method="basic",
        n_resamples=499,
        rng=np.random.default_rng(13),
    )
    np.testing.assert_array_equal(actual[0], statistic(values, axis=axis))
    np.testing.assert_array_equal(actual[1], expected.confidence_interval.low)
    np.testing.assert_array_equal(actual[2], expected.confidence_interval.high)
    assert np.isfinite(actual[1]).all()
    assert np.isfinite(actual[2]).all()
    constant = 10.0 if statistic is np.mean else 40.0
    assert actual[1][1] == constant
    assert actual[2][1] == constant


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_multidimensional_sample_axis(axis):
    # The first sample varies by output position. Constant positions therefore
    # cannot be detected by comparing the entire array to one scalar.
    values = np.arange(24.0, dtype=float).reshape(4, 2, 3)
    values[:, 1, 2] = 100.0
    values = np.moveaxis(values, 0, axis)
    actual = calc_metric_ci(
        values, axis=axis, n_resamples=499, rng=np.random.default_rng(17)
    )
    expected = bootstrap(
        [values],
        np.average,
        axis=axis,
        method="basic",
        n_resamples=499,
        rng=np.random.default_rng(17),
    )
    np.testing.assert_array_equal(actual[1], expected.confidence_interval.low)
    np.testing.assert_array_equal(actual[2], expected.confidence_interval.high)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_single_output_axis_does_not_index_another_dimension(axis):
    values = np.arange(5.0, dtype=float).reshape(5, 1)
    if axis != 0:
        values = values.T
    actual = calc_metric_ci(
        values, axis=axis, n_resamples=499, rng=np.random.default_rng(19)
    )
    expected = bootstrap(
        [values], np.average, axis=axis, n_resamples=499, rng=np.random.default_rng(19)
    )
    np.testing.assert_array_equal(actual[1], expected.confidence_interval.low)
    np.testing.assert_array_equal(actual[2], expected.confidence_interval.high)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_all_constant_outputs_follow_return_anyway(axis):
    values = np.full((4, 2), 2.0)
    if axis != 0:
        values = values.T
    with pytest.raises(Exception, match="All data are the same"):
        calc_metric_ci(values, axis=axis)
    result = calc_metric_ci(values, axis=axis, return_anyway=True)
    for entry in result:
        np.testing.assert_array_equal(entry, [2.0, 2.0])


def test_weighted_data_is_checked_after_preprocessing():
    rates = np.array([1.0, 2.0, 4.0, 8.0])
    values = np.column_stack([np.arange(4.0), 10.0 / rates])
    actual = calc_metric_ci(
        values, rates=rates, n_resamples=499, rng=np.random.default_rng(23)
    )
    weighted = values * rates[:, None]
    expected = bootstrap(
        [weighted],
        np.average,
        axis=0,
        method="basic",
        n_resamples=499,
        rng=np.random.default_rng(23),
    )
    np.testing.assert_array_equal(actual[1], expected.confidence_interval.low)
    np.testing.assert_array_equal(actual[2], expected.confidence_interval.high)


def test_history_confidence_bounds_preserve_constant_timesteps():
    hist = History(
        {
            "a.value": [0.0, 10.0],
            "b.value": [1.0, 10.0],
            "c.value": [2.0, 10.0],
            "d.value": [3.0, 10.0],
            "time": [0.0, 1.0],
        }
    )
    result = hist.get_mean_ci_errhist(
        "value", n_resamples=499, rng=np.random.default_rng(29)
    )
    assert result.stat[1] == result.low[1] == result.high[1] == 10.0
    assert np.isfinite(result.low).all()
    assert np.isfinite(result.high).all()


@pytest.mark.parametrize(
    "values", [[1.0, 2.0, 4.0, 8.0], [[1.0, 5.0], [2.0, 6.0], [4.0, 9.0], [8.0, 10.0]]]
)
def test_varying_data_retains_bca_results(values):
    values = np.asarray(values)
    actual = calc_metric_ci(values, n_resamples=499, rng=np.random.default_rng(31))
    expected = bootstrap(
        [values], np.average, axis=0, n_resamples=499, rng=np.random.default_rng(31)
    )
    np.testing.assert_array_equal(actual[1], expected.confidence_interval.low)
    np.testing.assert_array_equal(actual[2], expected.confidence_interval.high)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_distinct_constant_outputs_keep_degenerate_bounds(axis):
    values = np.tile([2.0, 7.0], (4, 1))
    if axis != 0:
        values = values.T
    actual = calc_metric_ci(
        values, axis=axis, n_resamples=499, rng=np.random.default_rng(37)
    )
    expected = bootstrap(
        [values],
        np.average,
        axis=axis,
        method="basic",
        n_resamples=499,
        rng=np.random.default_rng(37),
    )
    np.testing.assert_array_equal(actual[0], [2.0, 7.0])
    np.testing.assert_array_equal(actual[1], expected.confidence_interval.low)
    np.testing.assert_array_equal(actual[2], expected.confidence_interval.high)
