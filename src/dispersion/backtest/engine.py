"""
DispersionEngine — dispersion backtest, unconditional v0 and signal-gated v1.

Frozen design (README §7.3): DMV wealth sizing (short 100% of wealth in the SPX
straddle), vega-neutrality to PROPORTIONAL vol shocks (sum y_i nu_i sigma_i =
nu_I sigma_I — for ATM straddles nu ~ 1/sigma so sum y_i ~ 1, matching DMV's
+101.12%), daily index-only delta hedge, daily mark-to-surface (ATM proxy,
sigma(30d) frozen below 30d), intrinsic settlement at the next rebalance
(maturity ≡ next rebalance, ±3d documented), entry at real vendor premiums
(impl_premium), splits handled via cfadj, cash accrues at the short zerocd rate.

Usage:
    from dispersion.backtest.engine import run_backtest
    res = run_backtest()                    # v0 (unconditional)
    res = run_backtest(threshold=0.05)      # v1 (trade only if signal > threshold)
    res["daily"], res["quarterly"]
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from dispersion.backtest.marking import RateCurve, adjust_strike, interp_sigma
from dispersion.utils.greeks import bs_greeks, bs_price

SPX_SECID = 108105
PILLARS = (30, 60, 91)
ACT = 365.0

# Transaction-cost grid (README §8bis, notebook 05): QUOTED relative bid-ask
# spreads (% of premium), linear in time between the 1996 and 2024 anchors,
# clamped outside. Entry pays half of it per leg (impl_premium is mid-like);
# the daily index hedge pays `hedge_bps` of traded notional; settlement is free.
COST_ANCHORS = {"spx": (0.010, 0.006),     # SPX options
                "large": (0.070, 0.012),   # constituents ranked 1-50
                "small": (0.100, 0.030)}   # constituents ranked 51-100
DEFAULT_COSTS = {"hedge_bps": 1e-4, "scale": 1.0}


def parametric_spread(year: int, group: str) -> float:
    """Quoted relative spread for `group` in `year` (linear between anchors)."""
    v0, v1 = COST_ANCHORS[group]
    t = min(max(year, 1996), 2024)
    return v0 + (v1 - v0) * (t - 1996) / (2024 - 1996)


# ----------------------------------------------------------------------------- #
# Entry helpers
# ----------------------------------------------------------------------------- #
def implied_q(S, k_call, k_put, sig_call, sig_put, T, r, c_mkt, p_mkt,
              lo=-0.10, hi=0.25) -> float:
    """
    Entry dividend yield backed out of the vendor premiums via the C - P
    difference (monotone in q, well-conditioned near ATM). Falls back to 0.0
    if no root in [lo, hi].
    """
    def cp_diff(q):
        c = bs_price(S, k_call, sig_call, T, r, q, "C")
        p = bs_price(S, k_put, sig_put, T, r, q, "P")
        return (c - p) - (c_mkt - p_mkt)

    try:
        return float(brentq(cp_diff, lo, hi))
    except ValueError:
        return 0.0


def exante_quantile_threshold(signal: pd.DataFrame, rebalances, q: float = 0.5,
                              warmup: int = 12) -> pd.Series:
    """
    Per-rebalance ex-ante threshold (v1, README §8): at each rebalance, the
    quantile `q` of the DAILY signal history available up to that date
    (expanding window — no look-ahead). The first `warmup` rebalances get NaN,
    which the engine treats as "trade unconditionally" (v0 behaviour).
    """
    s = signal.copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.set_index("date")["signal"].astype(float).sort_index()
    out = {}
    for i, reb in enumerate(rebalances):
        reb = pd.Timestamp(reb)
        hist = s.loc[:reb].dropna()
        out[reb] = np.nan if (i < warmup or hist.empty) else float(hist.quantile(q))
    return pd.Series(out)


def _entry_leg(rows, cp):
    r = rows[rows["cp_flag"] == cp]
    if len(r) != 1:
        return None
    r = r.iloc[0]
    return float(r["iv"]), float(r["premium"]), float(r["strike"])


class _Position:
    """One straddle position (fixed strikes, fixed entry q-hat, split-aware)."""

    __slots__ = ("secid", "qty0", "cfadj0", "k_call", "k_put", "q_hat",
                 "last_mark", "last_greeks", "frozen_days")

    def __init__(self, secid, qty0, cfadj0, k_call, k_put, q_hat, entry_price):
        self.secid = secid
        self.qty0 = qty0
        self.cfadj0 = cfadj0
        self.k_call = k_call
        self.k_put = k_put
        self.q_hat = q_hat
        self.last_mark = entry_price          # per-unit straddle value
        self.last_greeks = {"delta_S": 0.0, "vega_c": 0.0, "vega_p": 0.0,
                            "gamma": 0.0, "theta": 0.0, "S": np.nan,
                            "sig_c": np.nan, "sig_p": np.nan}
        self.frozen_days = 0

    def qty(self, cfadj_now):
        return self.qty0 * cfadj_now / self.cfadj0

    def strikes(self, cfadj_now):
        return (adjust_strike(self.k_call, self.cfadj0, cfadj_now),
                adjust_strike(self.k_put, self.cfadj0, cfadj_now))


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def run_backtest(
    processed_dir: str = "data/processed",
    data: dict | None = None,
    threshold: float | None = None,
    costs: dict | None = None,
    cash_tenor: float = 10.0,
    min_names: int = 80,
) -> dict:
    """
    Run the dispersion backtest over every rebalance. `threshold=None` -> v0
    (always trade); a float or a per-rebalance pd.Series gates the quarters (v1).
    `costs=None` -> gross; `costs={}` or DEFAULT_COSTS -> net of the parametric
    spread grid (+ hedge_bps on hedge notional; optional 'scale' multiplier).
    `data` may inject the input frames directly (tests); else read parquets.

    Returns dict(daily=DataFrame, quarterly=DataFrame, ledger=DataFrame).
    """
    if data is None:
        data = {k: pd.read_parquet(os.path.join(processed_dir, f"{k}.parquet"))
                for k in ("surface", "spots", "rates", "weights", "signal")}
    surface, spots, rates = data["surface"], data["spots"], data["rates"]
    weights, signal = data["weights"], data["signal"]

    for df in (surface, spots, signal):
        df["date"] = pd.to_datetime(df["date"])
    for df in (surface, spots, weights):
        df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])

    rc = RateCurve(rates)
    sig_at = signal.set_index("date")["signal"]
    rebals = sorted(weights["rebalance_date"].unique())

    use_costs = costs is not None
    cost_scale = (costs or {}).get("scale", DEFAULT_COSTS["scale"])
    hedge_bps = (costs or {}).get("hedge_bps", DEFAULT_COSTS["hedge_bps"]) if use_costs else 0.0

    W = 1.0
    prev_nav = W
    daily_rows, q_rows, ledger_rows = [], [], []

    for k, reb in enumerate(rebals[:-1]):
        reb = pd.Timestamp(reb)
        nxt = pd.Timestamp(rebals[k + 1])

        surf_q = surface[surface["rebalance_date"] == reb]
        spot_q = spots[spots["rebalance_date"] == reb]
        w_q = weights[weights["rebalance_date"] == reb]

        # pivots for the quarter: sigma pillars per (secid, cp, days); spot close/cfadj
        piv_iv = surf_q.pivot_table(index="date", columns=["secid", "cp_flag", "days"],
                                    values="iv", aggfunc="first").astype("float64")
        piv_close = spot_q.pivot_table(index="date", columns="secid", values="close",
                                       aggfunc="first").astype("float64")
        piv_cfadj = spot_q.pivot_table(index="date", columns="secid", values="cfadj",
                                       aggfunc="first").astype("float64")
        # float64 casts: nullable dtypes leak pd.NA through .at lookups, and
        # np.isfinite(pd.NA) is ambiguous — NaN is what the freeze guards expect
        dates = piv_close.index.sort_values()

        # gate: None -> v0; float -> fixed threshold; pd.Series -> per-rebalance
        # ex-ante threshold (NaN entries = warm-up -> trade unconditionally)
        if threshold is None:
            traded = True
        else:
            thr = float(threshold.get(reb, np.nan)) if isinstance(threshold, pd.Series) \
                else float(threshold)
            if np.isnan(thr):
                traded = True
            else:
                s_val = float(sig_at.loc[reb]) if reb in sig_at.index else np.nan
                traded = bool(np.isfinite(s_val) and s_val > thr)

        # ------------------------------------------------------------------ #
        # Entry at the rebalance close
        # ------------------------------------------------------------------ #
        book, cash, n_hedge = [], W, 0.0
        y_sum = lam = np.nan
        held_n = 0
        c_entry_q = c_hedge_q = 0.0
        if traded:
            T0 = (nxt - reb).days / ACT
            r0 = rc.rate(reb, (nxt - reb).days)
            e = surf_q[(surf_q["date"] == reb) & (surf_q["days"] == 91)]

            entries = {}
            for secid, rows in e.groupby("secid"):
                c = _entry_leg(rows, "C")
                p = _entry_leg(rows, "P")
                S0 = piv_close.at[reb, secid] if secid in piv_close.columns else np.nan
                if c is None or p is None or not np.isfinite(S0):
                    continue
                sc, prem_c, kc = c
                sp_, prem_p, kp = p
                qh = implied_q(S0, kc, kp, sc, sp_, T0, r0, prem_c, prem_p)
                g_c = bs_greeks(S0, kc, sc, T0, r0, qh, "C")
                g_p = bs_greeks(S0, kp, sp_, T0, r0, qh, "P")
                price = prem_c + prem_p
                entries[secid] = dict(
                    price=price, kc=kc, kp=kp, qh=qh,
                    nu=(g_c["vega"] + g_p["vega"]) / price,   # relative vega
                    sig_bar=0.5 * (sc + sp_),
                    cfadj0=float(piv_cfadj.at[reb, secid]),
                )

            if SPX_SECID not in entries:
                raise ValueError(f"no SPX entry at {reb.date()}")
            idx_e = entries.pop(SPX_SECID)

            w_map = w_q.dropna(subset=["secid"]).set_index("secid")["weight"]
            held = [s for s in entries if s in w_map.index]
            held_n = len(held)
            if held_n < min_names:
                raise ValueError(f"only {held_n} names with entry premiums at {reb.date()}")

            w_til = w_map.loc[held] / w_map.loc[held].sum()
            denom = sum(w_til[s] * entries[s]["nu"] * entries[s]["sig_bar"] for s in held)
            lam = idx_e["nu"] * idx_e["sig_bar"] / denom      # proportional-shock neutrality
            y = {s: lam * w_til[s] for s in held}
            y_sum = float(sum(y.values()))

            # index short: -100% of wealth; components: +y_i of wealth
            book.append(_Position(SPX_SECID, -W / idx_e["price"], idx_e["cfadj0"],
                                  idx_e["kc"], idx_e["kp"], idx_e["qh"], idx_e["price"]))
            cash += W                                          # short proceeds
            for s in held:
                q_i = y[s] * W / entries[s]["price"]
                book.append(_Position(s, q_i, entries[s]["cfadj0"], entries[s]["kc"],
                                      entries[s]["kp"], entries[s]["qh"], entries[s]["price"]))
                cash -= y[s] * W

            if use_costs:  # half the quoted spread per entry leg (README §8bis)
                rnk_map = w_q.dropna(subset=["secid"]).set_index("secid")["rnk"] \
                    if "rnk" in w_q.columns else pd.Series(dtype=float)
                c_entry_q = 0.5 * cost_scale * parametric_spread(reb.year, "spx") * W
                for s in held:
                    grp = "large" if float(rnk_map.get(s, 1)) <= 50 else "small"
                    c_entry_q += 0.5 * cost_scale * parametric_spread(reb.year, grp) * y[s] * W
                cash -= c_entry_q

        # ------------------------------------------------------------------ #
        # Daily loop: mark, delta-hedge with the index, accrue cash
        # ------------------------------------------------------------------ #
        def _mark(pos, t, tau_days):
            """Per-unit straddle mark + greeks; freezes on missing data."""
            secid = pos.secid
            S = piv_close.at[t, secid] if (secid in piv_close.columns
                                           and t in piv_close.index) else np.nan
            cf = piv_cfadj.at[t, secid] if np.isfinite(S) else np.nan
            sigs = {}
            for cp in ("C", "P"):
                try:
                    s30 = piv_iv.at[t, (secid, cp, 30)]
                    s60 = piv_iv.at[t, (secid, cp, 60)]
                    s91 = piv_iv.at[t, (secid, cp, 91)]
                except KeyError:
                    s30 = s60 = s91 = np.nan
                sigs[cp] = interp_sigma(s30, s60, s91, tau_days)
            if not (np.isfinite(S) and np.isfinite(sigs["C"]) and np.isfinite(sigs["P"])
                    and np.isfinite(cf)):
                pos.frozen_days += 1
                return pos.last_mark, pos.last_greeks, None
            kc, kp = pos.strikes(cf)
            T = max(tau_days, 0.5) / ACT
            r = rc.rate(t, max(tau_days, 1.0))
            g_c = bs_greeks(S, kc, sigs["C"], T, r, pos.q_hat, "C")
            g_p = bs_greeks(S, kp, sigs["P"], T, r, pos.q_hat, "P")
            mark = g_c["price"] + g_p["price"]
            greeks = {"delta_S": (g_c["delta"] + g_p["delta"]) * S,
                      "vega_c": g_c["vega"], "vega_p": g_p["vega"],
                      "gamma": g_c["gamma"] + g_p["gamma"],
                      "theta": g_c["theta"] + g_p["theta"],
                      "S": S, "sig_c": sigs["C"], "sig_p": sigs["P"]}
            pos.last_mark, pos.last_greeks = mark, greeks
            return mark, greeks, cf

        def _book_state(t, tau_days):
            st = {"nav_pos": 0.0, "delta_dollar": 0.0, "nav_idx": 0.0, "nav_comp": 0.0,
                  "theta": 0.0, "vega": 0.0, "gamma_S2": 0.0}
            for pos in book:
                mark, greeks, cf = _mark(pos, t, tau_days)
                qty = pos.qty(cf) if cf is not None else pos.qty(pos.cfadj0)
                v = qty * mark
                st["nav_pos"] += v
                st["delta_dollar"] += qty * greeks["delta_S"]
                st["theta"] += qty * greeks["theta"]
                st["vega"] += qty * (greeks["vega_c"] + greeks["vega_p"])
                if np.isfinite(greeks["S"]):
                    st["gamma_S2"] += qty * greeks["gamma"] * greeks["S"] ** 2
                if pos.secid == SPX_SECID:
                    st["nav_idx"] += v
                else:
                    st["nav_comp"] += v
            return st

        # entry-day state (hedge set at the entry close)
        spx_S = lambda t: piv_close.at[t, SPX_SECID]
        if traded:
            st0 = _book_state(reb, (nxt - reb).days)
            n_hedge = -st0["delta_dollar"] / spx_S(reb)
            cash -= n_hedge * spx_S(reb)
            c = hedge_bps * abs(n_hedge) * spx_S(reb)
            cash -= c
            c_hedge_q += c

        prev_t = reb
        for t in dates[dates > reb]:
            dt_cal = (t - prev_t).days
            cash *= float(np.exp(rc.rate(prev_t, cash_tenor) * dt_cal / ACT))
            tau = (nxt - t).days
            if traded:
                st = _book_state(t, tau)
                nav = cash + st["nav_pos"] + n_hedge * spx_S(t)
                ledger_rows.append((t, reb, nav, cash, st["nav_idx"], st["nav_comp"],
                                    n_hedge, float(spx_S(t)), st["theta"], st["vega"],
                                    st["gamma_S2"], st["delta_dollar"]))
                # re-hedge at the close
                n_new = -st["delta_dollar"] / spx_S(t)
                cash -= (n_new - n_hedge) * spx_S(t)
                c = hedge_bps * abs(n_new - n_hedge) * spx_S(t)
                cash -= c
                c_hedge_q += c
                n_hedge = n_new
            else:
                nav = cash
            daily_rows.append((t, reb, nav / prev_nav - 1.0, nav, held_n, traded))
            prev_nav = nav
            prev_t = t

        # ------------------------------------------------------------------ #
        # Settlement at the next rebalance (intrinsic payoff)
        # ------------------------------------------------------------------ #
        dt_cal = (nxt - prev_t).days
        cash *= float(np.exp(rc.rate(prev_t, cash_tenor) * dt_cal / ACT))
        n_frozen_settle = 0
        if traded:
            spot_n = spots[(spots["rebalance_date"] == nxt) & (spots["date"] == nxt)]
            s_map = spot_n.set_index("secid")["close"]
            cf_map = spot_n.set_index("secid")["cfadj"]
            for pos in book:
                if pos.secid in s_map.index:
                    S, cf = float(s_map.loc[pos.secid]), float(cf_map.loc[pos.secid])
                else:  # left the universe / delisted: last frozen mark's spot
                    S, cf = pos.last_greeks["S"], pos.cfadj0
                    n_frozen_settle += 1
                    if not np.isfinite(S):
                        S, cf = 0.0, pos.cfadj0   # worthless fallback, counted
                kc, kp = pos.strikes(cf)
                payoff = max(S - kc, 0.0) + max(kp - S, 0.0)
                cash += pos.qty(cf) * payoff
            s_settle = float(s_map.loc[SPX_SECID])
            cash += n_hedge * s_settle
            c = hedge_bps * abs(n_hedge) * s_settle
            cash -= c
            c_hedge_q += c
            n_hedge = 0.0

        nav = cash
        daily_rows.append((nxt, reb, nav / prev_nav - 1.0, nav, held_n, traded))
        prev_nav = nav

        q_rows.append((reb, nxt, traded, held_n, nav / W - 1.0, y_sum, lam,
                       float(np.mean([p.frozen_days for p in book])) if book else 0.0,
                       n_frozen_settle, c_entry_q / W, c_hedge_q / W))
        W = nav

    daily = pd.DataFrame(daily_rows,
                         columns=["date", "rebalance_date", "ret", "nav",
                                  "n_names", "traded"])
    quarterly = pd.DataFrame(q_rows,
                             columns=["rebalance_date", "settle_date", "traded",
                                      "n_names", "ret_q", "y_sum", "lam",
                                      "avg_frozen_days", "n_frozen_settle",
                                      "cost_entry", "cost_hedge"])
    ledger = pd.DataFrame(ledger_rows,
                          columns=["date", "rebalance_date", "nav", "cash", "nav_idx",
                                   "nav_comp", "hedge_shares", "spx_close", "theta",
                                   "vega", "gamma_S2", "delta_dollar"])
    return {"daily": daily, "quarterly": quarterly, "ledger": ledger}
