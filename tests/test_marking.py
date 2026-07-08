"""
Unit tests for dispersion.backtest.marking (mark-to-surface rules, README §7.3).
"""
import numpy as np
import pandas as pd
import pytest

from dispersion.backtest.marking import RateCurve, adjust_strike, interp_sigma


# --------------------------------------------------------------------------- #
# interp_sigma — total-variance interpolation + frozen short end
# --------------------------------------------------------------------------- #
def test_exact_at_pillars():
    s30, s60, s91 = 0.30, 0.25, 0.22
    assert interp_sigma(s30, s60, s91, 30) == pytest.approx(s30, rel=1e-12)
    assert interp_sigma(s30, s60, s91, 60) == pytest.approx(s60, rel=1e-12)
    assert interp_sigma(s30, s60, s91, 91) == pytest.approx(s91, rel=1e-12)


def test_linear_in_total_variance_between_pillars():
    s30, s60, s91 = 0.30, 0.25, 0.22
    # tau = 45: w = (w30 + w60)/2 by linearity, sigma = sqrt(w/45)
    w30, w60 = s30**2 * 30, s60**2 * 60
    expected = np.sqrt(((w30 + w60) / 2) / 45)
    assert interp_sigma(s30, s60, s91, 45) == pytest.approx(expected, rel=1e-12)
    # tau = 75.5 (midpoint of [60, 91])
    w91 = s91**2 * 91
    expected = np.sqrt(((w60 + w91) / 2) / 75.5)
    assert interp_sigma(s30, s60, s91, 75.5) == pytest.approx(expected, rel=1e-12)


def test_frozen_below_30d():
    s30, s60, s91 = 0.30, 0.25, 0.22
    for tau in (29.9, 15, 5, 1):
        assert interp_sigma(s30, s60, s91, tau) == pytest.approx(s30, rel=1e-12)


def test_vectorised_and_nan_propagation():
    s30 = np.array([0.30, np.nan, 0.40])
    s60 = np.array([0.25, 0.25, 0.35])
    s91 = np.array([0.22, 0.22, 0.33])
    out = interp_sigma(s30, s60, s91, np.array([45.0, 45.0, 20.0]))
    assert out.shape == (3,)
    assert np.isnan(out[1])            # NaN pillar propagates
    assert out[2] == pytest.approx(0.40)  # frozen short end per element


# --------------------------------------------------------------------------- #
# RateCurve — maturity interpolation + bounded date-ffill
# --------------------------------------------------------------------------- #
def _toy_rates():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-06"] * 3),
        "days": [30, 91, 182] * 2,
        "rate": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],   # percent
    })


def test_rate_exact_and_interpolated():
    rc = RateCurve(_toy_rates())
    assert rc.rate("2020-01-02", 91) == pytest.approx(0.02, abs=1e-12)
    # midpoint of [30, 91]
    assert rc.rate("2020-01-02", 60.5) == pytest.approx(0.015, abs=1e-12)
    # clamped beyond the grid
    assert rc.rate("2020-01-02", 10) == pytest.approx(0.01, abs=1e-12)
    assert rc.rate("2020-01-02", 400) == pytest.approx(0.03, abs=1e-12)


def test_rate_bounded_ffill_and_loud_failure():
    rc = RateCurve(_toy_rates(), max_stale_days=7)
    # 2020-01-03 missing -> uses 01-02 curve (1 day stale, within bound)
    assert rc.rate("2020-01-03", 91) == pytest.approx(0.02, abs=1e-12)
    # far beyond the staleness bound -> loud failure
    with pytest.raises(KeyError):
        rc.rate("2020-02-01", 91)
    # before the first curve -> loud failure
    with pytest.raises(KeyError):
        rc.rate("2019-12-31", 91)


# --------------------------------------------------------------------------- #
# adjust_strike — split handling via cfadj
# --------------------------------------------------------------------------- #
def test_adjust_strike_split():
    # 2:1 split: cfadj 1 -> 2, price scale halves, so the fixed strike halves too
    assert adjust_strike(100.0, 1.0, 2.0) == pytest.approx(50.0)
    # no split: unchanged
    assert adjust_strike(100.0, 1.0, 1.0) == pytest.approx(100.0)
    # settlement consistency: |S - K| is split-invariant once both are rebased
    S_pre, K_pre = 120.0, 100.0
    S_post, cf_e, cf_t = 60.0, 1.0, 2.0
    K_post = adjust_strike(K_pre, cf_e, cf_t)
    qty_scale = cf_t / cf_e
    assert qty_scale * abs(S_post - K_post) == pytest.approx(abs(S_pre - K_pre))