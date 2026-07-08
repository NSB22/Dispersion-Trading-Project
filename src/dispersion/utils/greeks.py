"""
Black–Scholes / Black-76 pricing and greeks, with per-contract <-> relative conversions.

Frozen conventions (README §8bis):
- sigma = ANNUALISED implied volatility in DECIMAL (0.20 = 20%), as in vsurfd.
- T in years (ACT/365: days/365); r, q continuously compounded in decimal
  (zerocd rates come in %, divide by 100).
- vega = dP/d(sigma) with sigma in decimal (price change for +1.00 of vol, NOT per point).
- theta = dP/dt (calendar decay as time passes, per year) = -dP/dT. Negative for long options.
- RELATIVE greeks are per dollar invested: greek / price — the DMV eq. (8)-(11) wealth-weight
  framework. Mixing per-contract greeks with wealth weights silently yields a wrong hedge.

Two equivalent parameterisations:
- spot form  : bs_price/bs_greeks(S, K, sigma, T, r, q, cp) with continuous dividend yield q;
- forward form: black76_price(F, K, sigma, T, r, cp) with F = S*exp((r-q)*T) — preferred, since
  OptionMetrics forwards (fwdprd) or put-call-parity forwards embed dividends model-free.

Usage:
    from dispersion.utils.greeks import bs_greeks, relative_greeks, implied_forward
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

GREEK_KEYS = ("price", "delta", "gamma", "vega", "theta")


# ----------------------------------------------------------------------------- #
# Pricing
# ----------------------------------------------------------------------------- #
def _d1_d2(S, K, sigma, T, r, q):
    st = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / st
    return d1, d1 - st


def bs_price(S, K, sigma, T, r, q, cp: str) -> float:
    """European option price, spot form with continuous dividend yield q."""
    d1, d2 = _d1_d2(S, K, sigma, T, r, q)
    if cp == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if cp == "P":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError(f"cp must be 'C' or 'P', got {cp!r}")


def black76_price(F, K, sigma, T, r, cp: str) -> float:
    """European option price, forward form (Black-76): discounts the forward payoff."""
    # equivalent to the spot form with S = F*exp(-(r-q)*T); set q such that S*e^{(r-q)T}=F
    st = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / st
    d2 = d1 - st
    df = np.exp(-r * T)
    if cp == "C":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if cp == "P":
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    raise ValueError(f"cp must be 'C' or 'P', got {cp!r}")


# ----------------------------------------------------------------------------- #
# Greeks (per contract)
# ----------------------------------------------------------------------------- #
def bs_greeks(S, K, sigma, T, r, q, cp: str) -> dict:
    """
    Price + first-order greeks per contract, spot form.

    Returns dict(price, delta, gamma, vega, theta) — conventions in the module docstring.
    """
    d1, d2 = _d1_d2(S, K, sigma, T, r, q)
    eq, er = np.exp(-q * T), np.exp(-r * T)
    pdf1 = norm.pdf(d1)

    gamma = eq * pdf1 / (S * sigma * np.sqrt(T))
    vega = S * eq * pdf1 * np.sqrt(T)
    common_theta = -S * eq * pdf1 * sigma / (2.0 * np.sqrt(T))

    if cp == "C":
        price = S * eq * norm.cdf(d1) - K * er * norm.cdf(d2)
        delta = eq * norm.cdf(d1)
        theta = common_theta - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
    elif cp == "P":
        price = K * er * norm.cdf(-d2) - S * eq * norm.cdf(-d1)
        delta = -eq * norm.cdf(-d1)
        theta = common_theta + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
    else:
        raise ValueError(f"cp must be 'C' or 'P', got {cp!r}")

    return {"price": float(price), "delta": float(delta), "gamma": float(gamma),
            "vega": float(vega), "theta": float(theta)}


def straddle_greeks(S, K_call, K_put, sigma_call, sigma_put, T, r, q) -> dict:
    """Greeks of a standardised straddle = call + put legs (strikes may differ slightly)."""
    c = bs_greeks(S, K_call, sigma_call, T, r, q, "C")
    p = bs_greeks(S, K_put, sigma_put, T, r, q, "P")
    return {k: c[k] + p[k] for k in GREEK_KEYS}


# ----------------------------------------------------------------------------- #
# Per-contract <-> relative (per dollar invested) — the DMV convention
# ----------------------------------------------------------------------------- #
def relative_greeks(greeks: dict) -> dict:
    """
    Convert per-contract greeks to PER-DOLLAR-INVESTED greeks: greek / price.

    This is the object entering wealth-weighted hedge ratios (DMV eq. 10-11):
        y_i = vega_rel_index / vega_rel_component
    Using per-contract vegas with wealth weights is silently wrong (unit test covers it).
    """
    price = greeks["price"]
    if price <= 0:
        raise ValueError("relative greeks undefined for non-positive price")
    out = {k: greeks[k] / price for k in GREEK_KEYS if k != "price"}
    out["price"] = price
    return out


# ----------------------------------------------------------------------------- #
# Implied quantities
# ----------------------------------------------------------------------------- #
def implied_forward(call_price, put_price, K, T, r) -> float:
    """Model-free forward from put-call parity at a common strike: F = K + e^{rT}(C - P)."""
    return float(K + np.exp(r * T) * (call_price - put_price))


def implied_vol(price, S, K, T, r, q, cp: str, lo=1e-4, hi=5.0) -> float:
    """Invert bs_price for sigma (brentq)."""
    return float(brentq(lambda s: bs_price(S, K, s, T, r, q, cp) - price, lo, hi))
