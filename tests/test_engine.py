"""
Tests for dispersion.backtest.engine on a synthetic quarter with a closed-form
answer. With r = 0, flat sigmas, constant spots, ATM strikes and q = 0, the
settlement payoffs are 0 and daily hedging is cash-neutral, so the quarterly
return is exactly 1 - sum(y_i). Sizing identities are checked the same way.
"""
import numpy as np
import pandas as pd
import pytest

from dispersion.backtest.engine import (SPX_SECID, exante_quantile_threshold,
                                        implied_q, run_backtest)
from dispersion.utils.greeks import bs_greeks, bs_price

REB = pd.Timestamp("2020-03-31")
NXT = pd.Timestamp("2020-06-30")           # 91 calendar days
T0 = (NXT - REB).days / 365.0
NAMES = {1: dict(S=100.0, w=0.6), 2: dict(S=50.0, w=0.4)}
SPX_S = 1000.0


def _make_data(sig_by_secid):
    dates = pd.DatetimeIndex([REB, "2020-04-30", "2020-05-29", "2020-06-29"])
    surf, spot, rate, wgt = [], [], [], []
    all_ids = {SPX_SECID: SPX_S, **{s: NAMES[s]["S"] for s in NAMES}}

    for secid, S in all_ids.items():
        sig = sig_by_secid[secid]
        prem = {cp: bs_price(S, S, sig, T0, 0.0, 0.0, cp) for cp in ("C", "P")}
        for d in dates:
            for days in (30, 60, 91):
                for cp in ("C", "P"):
                    surf.append((REB, d, secid, days, cp, sig, prem[cp], S))
            spot.append((REB, d, secid, S, 1.0))
    # settlement rows live in the next quarter's partition (date == NXT)
    for secid, S in all_ids.items():
        spot.append((NXT, NXT, secid, S, 1.0))

    for d in list(dates) + [NXT]:
        for days in (10, 30, 60, 91, 122):
            rate.append((d, days, 0.0))

    for rnk, (secid, info) in enumerate(NAMES.items(), start=1):
        wgt.append((REB, secid, info["w"], rnk))
        wgt.append((NXT, secid, info["w"], rnk))   # gives NXT a rebalance so settlement fires

    return {
        "surface": pd.DataFrame(surf, columns=["rebalance_date", "date", "secid",
                                               "days", "cp_flag", "iv", "premium",
                                               "strike"]),
        "spots": pd.DataFrame(spot, columns=["rebalance_date", "date", "secid",
                                             "close", "cfadj"]),
        "rates": pd.DataFrame(rate, columns=["date", "days", "rate"]),
        "weights": pd.DataFrame(wgt, columns=["rebalance_date", "secid", "weight", "rnk"]),
        "signal": pd.DataFrame({"date": [REB], "signal": [0.10]}),
    }


def test_implied_q_roundtrip():
    S, k_c, k_p, sc, sp_, T, r, q_true = 100.0, 101.0, 99.0, 0.25, 0.27, 0.25, 0.03, 0.017
    c = bs_price(S, k_c, sc, T, r, q_true, "C")
    p = bs_price(S, k_p, sp_, T, r, q_true, "P")
    assert implied_q(S, k_c, k_p, sc, sp_, T, r, c, p) == pytest.approx(q_true, abs=1e-8)


def test_flat_quarter_closed_form():
    # same sigma everywhere -> nu*sigma equal across names -> lam = 1, y_sum = 1
    data = _make_data({SPX_SECID: 0.20, 1: 0.20, 2: 0.20})
    res = run_backtest(data=data, min_names=2)
    q = res["quarterly"].iloc[0]

    assert q["traded"] and q["n_names"] == 2
    assert q["lam"] == pytest.approx(1.0, rel=1e-10)
    assert q["y_sum"] == pytest.approx(1.0, rel=1e-10)
    # zero settlement, cash-neutral hedge, r = 0: quarterly return is 1 - y_sum
    assert q["ret_q"] == pytest.approx(1.0 - q["y_sum"], abs=1e-12)
    # compounded daily returns match the quarterly return
    d = res["daily"]
    assert (1 + d["ret"]).prod() - 1 == pytest.approx(q["ret_q"], abs=1e-12)
    assert q["avg_frozen_days"] == 0 and q["n_frozen_settle"] == 0


def test_sizing_identity_heterogeneous_vols():
    sig = {SPX_SECID: 0.18, 1: 0.35, 2: 0.22}
    data = _make_data(sig)
    res = run_backtest(data=data, min_names=2)
    q = res["quarterly"].iloc[0]

    # closed-form lam: nu = straddle vega / straddle price at entry
    def nu(S, s):
        g_c = bs_greeks(S, S, s, T0, 0.0, 0.0, "C")
        g_p = bs_greeks(S, S, s, T0, 0.0, 0.0, "P")
        return (g_c["vega"] + g_p["vega"]) / (g_c["price"] + g_p["price"])

    nu_i = nu(SPX_S, sig[SPX_SECID])
    denom = sum(NAMES[s]["w"] * nu(NAMES[s]["S"], sig[s]) * sig[s] for s in NAMES)
    lam_expected = nu_i * sig[SPX_SECID] / denom
    assert q["lam"] == pytest.approx(lam_expected, rel=1e-9)

    # wealth-vega neutrality under a proportional shock holds by construction
    lhs = sum(q["lam"] * NAMES[s]["w"] * nu(NAMES[s]["S"], sig[s]) * sig[s] for s in NAMES)
    assert lhs == pytest.approx(nu_i * sig[SPX_SECID], rel=1e-9)


def test_settlement_intrinsic_payoff():
    data = _make_data({SPX_SECID: 0.20, 1: 0.20, 2: 0.20})
    # move name 1's settlement spot so its straddle pays |120 - 100| = 20 per unit
    sp = data["spots"]
    sp.loc[(sp["rebalance_date"] == NXT) & (sp["secid"] == 1), "close"] = 120.0
    res = run_backtest(data=data, min_names=2)
    q = res["quarterly"].iloc[0]

    # flat-case return (1 - y_sum) plus name 1's settlement payoff
    price_1 = bs_price(100.0, 100.0, 0.20, T0, 0.0, 0.0, "C") + \
              bs_price(100.0, 100.0, 0.20, T0, 0.0, 0.0, "P")
    y_1 = q["lam"] * NAMES[1]["w"]
    expected = (1.0 - q["y_sum"]) + y_1 / price_1 * 20.0
    assert q["ret_q"] == pytest.approx(expected, abs=1e-12)


def test_parsimonious_leg_keeps_vega_neutrality():
    # n_leg=1 keeps only the top-weight name but rescales it, so wealth-vega
    # neutrality (y_sum ~ 1 in the flat case) and ret_q = 1 - y_sum still hold.
    data = _make_data({SPX_SECID: 0.20, 1: 0.20, 2: 0.20})
    res = run_backtest(data=data, min_names=2, n_leg=1)
    q = res["quarterly"].iloc[0]
    assert q["n_names"] == 1                        # only the top-weight component
    assert q["y_sum"] == pytest.approx(1.0, rel=1e-10)   # still wealth-vega neutral
    assert q["ret_q"] == pytest.approx(1.0 - q["y_sum"], abs=1e-12)


def test_v1_gate_skips_quarter():
    data = _make_data({SPX_SECID: 0.20, 1: 0.20, 2: 0.20})
    res = run_backtest(data=data, min_names=2, threshold=0.50)  # signal = 0.10 < 0.50
    q = res["quarterly"].iloc[0]
    assert not q["traded"]
    assert q["ret_q"] == pytest.approx(0.0, abs=1e-12)          # nothing traded, r = 0 -> flat


def test_costs_closed_form():
    from dispersion.backtest.engine import parametric_spread
    data = _make_data({SPX_SECID: 0.20, 1: 0.20, 2: 0.20})
    res_g = run_backtest(data=data, min_names=2)
    res_n = run_backtest(data=data, min_names=2, costs={"hedge_bps": 0.0})
    qg, qn = res_g["quarterly"].iloc[0], res_n["quarterly"].iloc[0]
    # entry cost = half-spread on the index leg (100% of W) plus each component leg
    expected = 0.5 * parametric_spread(REB.year, "spx") \
        + 0.5 * parametric_spread(REB.year, "large") * qg["y_sum"]  # rnk 1-2 map to "large"
    assert qn["cost_entry"] == pytest.approx(expected, rel=1e-10)
    assert qn["cost_hedge"] == 0.0
    assert qn["ret_q"] == pytest.approx(qg["ret_q"] - expected, abs=1e-12)


def test_exante_quantile_threshold_values():
    sig = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100, freq="D"),
                        "signal": np.linspace(0.0, 1.0, 100)})
    rebs = [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01")]
    thr = exante_quantile_threshold(sig, rebs, q=0.5, warmup=1)
    assert np.isnan(thr.iloc[0])                                # still warming up
    expected = sig.set_index("date")["signal"].loc[:"2020-04-01"].median()
    assert thr.iloc[1] == pytest.approx(expected, rel=1e-12)    # median of history so far


def test_v1_series_gate_warmup_trades():
    data = _make_data({SPX_SECID: 0.20, 1: 0.20, 2: 0.20})
    # NaN threshold at REB (warm-up) trades like v0
    thr = pd.Series({REB: np.nan})
    res = run_backtest(data=data, min_names=2, threshold=thr)
    assert bool(res["quarterly"].iloc[0]["traded"])
    # threshold above the signal (0.10) skips
    thr = pd.Series({REB: 0.50})
    res = run_backtest(data=data, min_names=2, threshold=thr)
    assert not bool(res["quarterly"].iloc[0]["traded"])