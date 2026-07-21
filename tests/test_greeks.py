"""
Tests for dispersion.utils.greeks: closed-form values and identities, greeks vs
finite differences, the relative-vs-per-contract vega hedge, and reproducing real
SPX premiums.
"""
import numpy as np
import pytest
from scipy.stats import norm

from dispersion.utils.greeks import (
    bs_greeks,
    bs_price,
    black76_price,
    implied_forward,
    implied_vol,
    relative_greeks,
    straddle_greeks,
)

# non-trivial base case: ITM-ish call, with dividends and rates
BASE = dict(S=100.0, K=95.0, sigma=0.25, T=0.25, r=0.04, q=0.015)


# --------------------------------------------------------------------------- #
# 1. Closed form & identities
# --------------------------------------------------------------------------- #
def test_atm_forward_closed_form():
    # BS price against its exact closed form when r = q = 0 and K = S
    S = K = 100.0
    sigma, T = 0.20, 91 / 365
    d = 0.5 * sigma * np.sqrt(T)
    expected_call = S * (norm.cdf(d) - norm.cdf(-d))
    assert bs_price(S, K, sigma, T, 0.0, 0.0, "C") == pytest.approx(expected_call, rel=1e-12)
    # ATM-forward, so call and put match
    assert bs_price(S, K, sigma, T, 0.0, 0.0, "P") == pytest.approx(expected_call, rel=1e-12)


def test_put_call_parity_and_symmetries():
    rng = np.random.default_rng(42)
    for _ in range(200):
        S = rng.uniform(20, 500)
        K = S * rng.uniform(0.7, 1.3)
        sigma = rng.uniform(0.05, 0.9)
        T = rng.uniform(0.02, 2.0)
        r = rng.uniform(0.0, 0.08)
        q = rng.uniform(0.0, 0.05)
        c = bs_greeks(S, K, sigma, T, r, q, "C")
        p = bs_greeks(S, K, sigma, T, r, q, "P")
        # put-call parity: C - P = S e^{-qT} - K e^{-rT}
        assert c["price"] - p["price"] == pytest.approx(
            S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-9
        )
        # delta_C - delta_P = e^{-qT}; gamma and vega equal across the two legs
        assert c["delta"] - p["delta"] == pytest.approx(np.exp(-q * T), abs=1e-12)
        assert c["gamma"] == pytest.approx(p["gamma"], rel=1e-12)
        assert c["vega"] == pytest.approx(p["vega"], rel=1e-12)


def test_black76_equals_spot_form():
    S, K, sigma, T, r, q = BASE.values()
    F = S * np.exp((r - q) * T)
    for cp in ("C", "P"):
        assert black76_price(F, K, sigma, T, r, cp) == pytest.approx(
            bs_price(S, K, sigma, T, r, q, cp), rel=1e-12
        )


def test_implied_forward_parity():
    S, K, sigma, T, r, q = BASE.values()
    c = bs_price(S, K, sigma, T, r, q, "C")
    p = bs_price(S, K, sigma, T, r, q, "P")
    assert implied_forward(c, p, K, T, r) == pytest.approx(S * np.exp((r - q) * T), rel=1e-12)


def test_implied_vol_round_trip():
    S, K, sigma, T, r, q = BASE.values()
    for cp in ("C", "P"):
        price = bs_price(S, K, sigma, T, r, q, cp)
        assert implied_vol(price, S, K, T, r, q, cp) == pytest.approx(sigma, abs=1e-10)


def test_straddle_is_sum_of_legs():
    S, K, sigma, T, r, q = BASE.values()
    st = straddle_greeks(S, K_call=K, K_put=K + 1, sigma_call=sigma, sigma_put=sigma + 0.01,
                         T=T, r=r, q=q)
    c = bs_greeks(S, K, sigma, T, r, q, "C")
    p = bs_greeks(S, K + 1, sigma + 0.01, T, r, q, "P")
    for k in ("price", "delta", "gamma", "vega", "theta"):
        assert st[k] == pytest.approx(c[k] + p[k], rel=1e-12)


# --------------------------------------------------------------------------- #
# 2. Greeks vs bump-and-reprice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cp", ["C", "P"])
def test_greeks_match_finite_differences(cp):
    S, K, sigma, T, r, q = BASE.values()
    g = bs_greeks(S, K, sigma, T, r, q, cp)
    hs, h = 1e-3, 1e-5  # bigger S bump for the second derivative (cancellation noise ~1/h^2)

    delta_fd = (bs_price(S + hs, K, sigma, T, r, q, cp) - bs_price(S - hs, K, sigma, T, r, q, cp)) / (2 * hs)
    gamma_fd = (bs_price(S + hs, K, sigma, T, r, q, cp) - 2 * bs_price(S, K, sigma, T, r, q, cp)
                + bs_price(S - hs, K, sigma, T, r, q, cp)) / hs**2
    vega_fd = (bs_price(S, K, sigma + h, T, r, q, cp) - bs_price(S, K, sigma - h, T, r, q, cp)) / (2 * h)
    # theta is the derivative w.r.t. time passing, i.e. -dP/dT
    theta_fd = -(bs_price(S, K, sigma, T + h, r, q, cp) - bs_price(S, K, sigma, T - h, r, q, cp)) / (2 * h)

    assert g["delta"] == pytest.approx(delta_fd, rel=1e-7)
    assert g["gamma"] == pytest.approx(gamma_fd, rel=1e-5)
    assert g["vega"] == pytest.approx(vega_fd, rel=1e-7)
    assert g["theta"] == pytest.approx(theta_fd, rel=1e-6)

    # second order: volga = d(vega)/d(sigma), vanna = d(vega)/d(S)
    volga_fd = (bs_greeks(S, K, sigma + h, T, r, q, cp)["vega"]
                - bs_greeks(S, K, sigma - h, T, r, q, cp)["vega"]) / (2 * h)
    vanna_fd = (bs_greeks(S + hs, K, sigma, T, r, q, cp)["vega"]
                - bs_greeks(S - hs, K, sigma, T, r, q, cp)["vega"]) / (2 * hs)
    assert g["volga"] == pytest.approx(volga_fd, rel=1e-5)
    assert g["vanna"] == pytest.approx(vanna_fd, rel=1e-5)


def test_second_order_greeks_vanish_at_d2_zero_strike():
    # volga (∝ d1·d2) and vanna (∝ d2) are both zero at the strike where d2 = 0,
    # and grow as a straddle ages away from it -- that drift is the convexity risk.
    S, sigma, T, r, q = 100.0, 0.25, 0.5, 0.03, 0.01
    K0 = S * np.exp((r - q - 0.5 * sigma**2) * T)      # d2 = 0
    g0 = bs_greeks(S, K0, sigma, T, r, q, "C")
    assert abs(g0["vanna"]) < 1e-9 and abs(g0["volga"]) < 1e-9
    # spot drifted down 15%: both are clearly non-zero
    g1 = bs_greeks(S * 0.85, K0, sigma, T, r, q, "C")
    assert abs(g1["vanna"]) > 1e-3 and abs(g1["volga"]) > 1e-2


# --------------------------------------------------------------------------- #
# 3. Relative vs per-contract hedge ratios
# --------------------------------------------------------------------------- #
def test_relative_vega_hedge_is_wealth_neutral_and_per_contract_is_not():
    T, r, q = 0.25, 0.04, 0.0
    # expensive index-like straddle vs cheap single-name-like straddle
    idx = straddle_greeks(S=5000.0, K_call=5000.0, K_put=5000.0,
                          sigma_call=0.12, sigma_put=0.12, T=T, r=r, q=q)
    comp = straddle_greeks(S=50.0, K_call=50.0, K_put=50.0,
                           sigma_call=0.30, sigma_put=0.30, T=T, r=r, q=q)
    idx_rel, comp_rel = relative_greeks(idx), relative_greeks(comp)

    # hedge ratio from the relative vegas
    y = idx_rel["vega"] / comp_rel["vega"]
    # short $1 of index, long $y of component -> zero wealth-vega
    wealth_vega = -1.0 * idx_rel["vega"] + y * comp_rel["vega"]
    assert wealth_vega == pytest.approx(0.0, abs=1e-12)

    # using the per-contract vega ratio instead leaves a large residual
    y_trap = idx["vega"] / comp["vega"]
    wealth_vega_trap = -1.0 * idx_rel["vega"] + y_trap * comp_rel["vega"]
    assert abs(wealth_vega_trap) > 0.1 * idx_rel["vega"]  # off by a lot, not rounding

    # the two ratios differ by exactly the price ratio
    assert y_trap / y == pytest.approx(idx["price"] / comp["price"], rel=1e-12)


# --------------------------------------------------------------------------- #
# 4. Real data: SPX standardised 91d ±50δ, 2024-06-28
# --------------------------------------------------------------------------- #
def test_reproduces_spx_impl_premium_2024():
    S, T, r = 5460.48, 91 / 365, 0.0550  # secprd close, zerocd 91d
    q = 0.0104                            # implied dividend yield, same for both legs
    call = bs_price(S, 5530.623, 0.121323, T, r, q, "C")
    put = bs_price(S, 5531.409, 0.113183, T, r, q, "P")
    assert call == pytest.approx(127.2951, rel=5e-3)
    assert put == pytest.approx(127.7991, rel=5e-3)
