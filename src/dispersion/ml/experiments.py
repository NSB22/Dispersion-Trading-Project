"""
Week-4 experimental levers (parallel to the main ML — nothing here overwrites it).

Lever 1 — DAILY spike target: predict the forward correlation spike
    y_spike(t) = rho_trail63(t+63) − rho_trail63(t)
on ~7,000 daily rows (vs ~90 quarterly trade-return rows), walk-forward with
purge 63d + embargo 21d. The spike is more predictable than the trade return
(correlation is persistent) and it is the actual loss channel. Gate: veto the
trade if the predicted spike at the rebalance is in the high (dangerous) tail.

Lever 3 — REGIME-ONLY gate: no supervised model at all — veto if the causal
regime probability f_regime at the rebalance is in the high tail.

Both are meta-labelling vetoes ON TOP of the v1_rmt primary gate, for a clean
comparison. Pre-registered veto = ex-ante expanding 67th percentile (veto the
top third of predicted danger).

Usage:
    from dispersion.ml.experiments import run_levers
    run_levers()      # writes backtest_regonly_* and backtest_spike_* + a summary
"""
import json
import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..backtest.engine import exante_quantile_threshold, run_backtest

HORIZON, EMBARGO = 63, 21
MIN_TRAIN_DAYS = 756
VETO_Q = 0.67                      # veto the top third of predicted danger (ex-ante)

SPIKE_FEATS = ["f_vix", "f_term_slope", "f_lam1", "f_dlam1_63", "f_turb21", "f_k",
               "f_rot21", "f_vrp", "f_drho_imp_21", "f_sig_rmt", "f_regime"]
XGB = dict(objective="reg:squarederror", max_depth=3, n_estimators=200,
           learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=20, reg_lambda=1.0, random_state=0)


def predict_spike_daily(processed_dir="data/processed", features=SPIKE_FEATS):
    """
    Walk-forward daily spike prediction: refit at each rebalance on the expanding
    daily history (purged), predict that quarter's days. Returns (yhat_at_reb dict,
    oos_corr) where oos_corr = corr(predicted spike, realised spike) over all
    out-of-sample days — the key 'is the spike predictable?' number.
    """
    ml = pd.read_parquet(os.path.join(processed_dir, "ml_dataset.parquet"))
    weights = pd.read_parquet(os.path.join(processed_dir, "weights.parquet"))
    ml["date"] = pd.to_datetime(ml["date"])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    rebals = sorted(weights["rebalance_date"].unique())
    idx = ml.set_index("date")

    purge_cal = int((HORIZON + EMBARGO) * 7 / 5)
    yhat_reb = {}
    oos_pred, oos_true = [], []
    for k, reb in enumerate(rebals):
        reb = pd.Timestamp(reb)
        nxt = pd.Timestamp(rebals[k + 1]) if k + 1 < len(rebals) else idx.index[-1] + pd.Timedelta(days=1)
        cutoff = reb - pd.Timedelta(days=purge_cal)
        tr = ml[ml["date"] <= cutoff].dropna(subset=features + ["y_spike"])
        if len(tr) < MIN_TRAIN_DAYS:
            continue
        m = XGBRegressor(**XGB).fit(tr[features].to_numpy("float64"),
                                    tr["y_spike"].to_numpy("float64"))
        # gate value at the rebalance
        row = idx.loc[[reb], features] if reb in idx.index else None
        if row is not None and row.notna().all(axis=1).iloc[0]:
            yhat_reb[reb] = float(m.predict(row.to_numpy("float64"))[0])
        # OOS predictive check on the quarter's days
        seg = ml[(ml["date"] >= reb) & (ml["date"] < nxt)].dropna(subset=features + ["y_spike"])
        if not seg.empty:
            p = m.predict(seg[features].to_numpy("float64"))
            oos_pred.extend(p.tolist())
            oos_true.extend(seg["y_spike"].tolist())
    oos_corr = float(np.corrcoef(oos_pred, oos_true)[0, 1]) if len(oos_pred) > 10 else np.nan
    return yhat_reb, oos_corr


def _high_veto(vals_at_reb: dict, q=VETO_Q, warmup=MIN_TRAIN_DAYS // 21) -> set:
    """Veto rebalances whose value exceeds the ex-ante expanding q-quantile."""
    dates = sorted(vals_at_reb)
    veto, hist = set(), []
    for d in dates:
        v = vals_at_reb[d]
        if len(hist) >= warmup and np.isfinite(v) and v > np.quantile(hist, q):
            veto.add(pd.Timestamp(d))
        if np.isfinite(v):
            hist.append(v)
    return veto


def _stats(q, d):
    r, dr = q["ret_q"].astype(float), d["ret"].astype(float)
    cum = (1 + dr).cumprod()
    return dict(trades=int(q["traded"].sum()), n=len(q), sharpe=r.mean() / r.std() * 2,
                skew=float(r.skew()), maxDD=float((cum / cum.cummax() - 1).min()),
                cumul=float(cum.iloc[-1]), worst=float(r.min()))


def run_levers(processed_dir="data/processed"):
    """Backtest lever 1 (spike) and lever 3 (regime-only) vs v1_rmt. Writes outputs."""
    ml = pd.read_parquet(os.path.join(processed_dir, "ml_dataset.parquet"))
    signal_rmt = pd.read_parquet(os.path.join(processed_dir, "signal_rmt.parquet"))
    weights = pd.read_parquet(os.path.join(processed_dir, "weights.parquet"))
    ml["date"] = pd.to_datetime(ml["date"])
    signal_rmt["date"] = pd.to_datetime(signal_rmt["date"])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    rebals = sorted(weights["rebalance_date"].unique())
    base = {kk: pd.read_parquet(os.path.join(processed_dir, f"{kk}.parquet"))
            for kk in ("surface", "spots", "rates", "weights")}
    v1rmt_thr = exante_quantile_threshold(signal_rmt, rebals, q=0.5, warmup=12)

    # lever 3: regime at rebalances
    reg_at = {pd.Timestamp(r): float(ml.loc[ml["date"] == pd.Timestamp(r), "f_regime"].iloc[0])
              for r in rebals if (ml["date"] == pd.Timestamp(r)).any()
              and np.isfinite(ml.loc[ml["date"] == pd.Timestamp(r), "f_regime"].iloc[0])}
    veto_reg = _high_veto(reg_at)

    # lever 1: daily spike predictor
    yhat_reb, oos_corr = predict_spike_daily(processed_dir)
    veto_spike = _high_veto(yhat_reb)

    out = {"oos_corr_spike": oos_corr, "n_veto_regime": len(veto_reg),
           "n_veto_spike": len(veto_spike), "arms": {}}
    for name, veto in [("regonly", veto_reg), ("spike", veto_spike)]:
        thr = v1rmt_thr.copy()
        for d in veto:
            thr[pd.Timestamp(d)] = 1e9
        for tag, kw in [("gross", {}), ("net", dict(costs={}))]:
            bt = run_backtest(data={**{k: v.copy() for k, v in base.items()},
                                    "signal": signal_rmt.copy()}, threshold=thr, **kw)
            bt["quarterly"].to_parquet(
                os.path.join(processed_dir, f"backtest_{name}_{tag}_quarterly.parquet"), index=False)
            bt["daily"].to_parquet(
                os.path.join(processed_dir, f"backtest_{name}_{tag}_daily.parquet"), index=False)
            out["arms"].setdefault(name, {})[tag] = _stats(bt["quarterly"], bt["daily"])
    json.dump(out, open(os.path.join(processed_dir, "ml_levers_summary.json"), "w"), indent=2)
    return out
