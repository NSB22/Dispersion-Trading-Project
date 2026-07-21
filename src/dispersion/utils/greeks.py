"""
Black-Scholes / Black-76 prices and greeks, with per-contract <-> relative
conversions. sigma is annualised IV in decimals, T in years, r and q continuous.

Relative greeks are per dollar invested (greek / price) — that's the ratio the
vega hedge uses; mixing per-contract greeks with wealth weights gives a wrong hedge.

Usage:
    from dispersion.utils.greeks import bs_greeks, relative_greeks, implied_forward
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

GREEK_KEYS = ("price", "delta", "gamma", "vega", "theta", "volga", "vanna")


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
    # same as the spot form with S = F*exp(-(r-q)*T)
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
    Price plus first- and second-order greeks per contract, spot form.
    Returns dict(price, delta, gamma, vega, theta, volga, vanna).

    volga and vanna carry the book's convexity risk:
      volga = d(vega)/d(sigma) = vega * d1 * d2 / sigma
      vanna = d(vega)/d(S)     = d(delta)/d(sigma)
    Both vanish at the ATM-forward and grow as a straddle drifts off ATM.
    """
    d1, d2 = _d1_d2(S, K, sigma, T, r, q)
    eq, er = np.exp(-q * T), np.exp(-r * T)
    pdf1 = norm.pdf(d1)

    gamma = eq * pdf1 / (S * sigma * np.sqrt(T))
    vega = S * eq * pdf1 * np.sqrt(T)
    volga = vega * d1 * d2 / sigma
    vanna = -eq * pdf1 * d2 / sigma
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
            "vega": float(vega), "theta": float(theta),
            "volga": float(volga), "vanna": float(vanna)}


def straddle_greeks(S, K_call, K_put, sigma_call, sigma_put, T, r, q) -> dict:
    """Greeks of a standardised straddle = call + put legs (strikes may differ slightly)."""
    c = bs_greeks(S, K_call, sigma_call, T, r, q, "C")
    p = bs_greeks(S, K_put, sigma_put, T, r, q, "P")
    return {k: c[k] + p[k] for k in GREEK_KEYS}


# ----------------------------------------------------------------------------- #
# Per-contract <-> relative (per dollar invested)
# ----------------------------------------------------------------------------- #
def relative_greeks(greeks: dict) -> dict:
    """
    Per-contract greeks -> per-dollar-invested greeks (greek / price).

    These are what go into the wealth-weighted hedge ratio
    y_i = vega_rel_index / vega_rel_component; per-contract vegas with wealth
    weights give the wrong hedge.
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
