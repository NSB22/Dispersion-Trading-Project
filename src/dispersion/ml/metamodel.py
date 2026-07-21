"""
Meta-model + veto (steps ④-⑤): predict each quarter's dispersion-trade return
from the features, walk-forward with purge+embargo, then VETO the trades the
meta-model expects in the bad tail. On top of the v1_rmt primary gate.

Pre-registered (plan.md): shptallow XGBoost with FIXED conservative hyper-params
(no CV tuning on ~12 crises), purge = 63 trading days (label horizon) + embargo =
21 days, veto if predicted return < ex-ante EXPANDING 33rd percentile of past
predictions. Central test compares feature sets:
  A "VIX only"      : f_vix, f_term_slope, f_regime_vix
  B "VIX + spectral": A + f_lam1, f_dlam1_63, f_turb21, f_k, f_rot21, f_regime
  Full              : the 10 features + f_regime (adds f_vrp, f_drho_imp_21, f_sig_rmt)

Sample is tiny (~90 quarterly predictions, ~12 crises) — report with uncertainty.

Usage:
    from dispersion.ml.metamodel import run_central_test
    res = run_central_test()
"""
import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..backtest.engine import exante_quantile_threshold, run_backtest
from .regime import regime_probs

HORIZON, EMBARGO = 63, 21          # trading days
MIN_TRAIN = 24                     # quarters before the first prediction (~6y)
VETO_PCTILE = 0.33                 # veto the bottom third of predicted returns (ex-ante)

XGB_PARAMS = dict(objective="reg:squarederror", max_depth=2, n_estimators=100,
                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, random_state=0)

FEATURE_SETS = {
    "A_vix":       ["f_vix", "f_term_slope", "f_regime_vix"],
    "B_spectral":  ["f_vix", "f_term_slope", "f_regime_vix",
                    "f_lam1", "f_dlam1_63", "f_turb21", "f_k", "f_rot21", "f_regime"],
    "Full":        ["f_vix", "f_term_slope", "f_lam1", "f_dlam1_63", "f_turb21",
                    "f_k", "f_rot21", "f_vrp", "f_drho_imp_21", "f_sig_rmt", "f_regime"],
}


def _snapshots(processed_dir):
    """Feature snapshot at each rebalance close + the quarter's v0 trade return (label)."""
    ml = pd.read_parquet(os.path.join(processed_dir, "ml_dataset.parquet"))
    ml["date"] = pd.to_datetime(ml["date"])
    q0 = pd.read_parquet(os.path.join(processed_dir, "backtest_v0_quarterly.parquet"))
    q0["rebalance_date"] = pd.to_datetime(q0["rebalance_date"])

    # add the VIX-only regime variant (fit on vol features only — keeps the central test clean)
    weights = pd.read_parquet(os.path.join(processed_dir, "weights.parquet"))
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    rebals = sorted(weights["rebalance_date"].unique())
    feat_vix = ml.set_index("date")[["f_vix", "f_term_slope"]]
    pg, ph = regime_probs(feat_vix, rebals, core=["f_vix", "f_term_slope"])
    ml["f_regime_vix"] = np.nanmean(np.c_[pg.to_numpy(), ph.to_numpy()], axis=1)

    snap = ml.merge(q0[["rebalance_date", "ret_q"]].rename(
        columns={"rebalance_date": "date", "ret_q": "y"}), on="date", how="inner")
    return snap.sort_values("date").reset_index(drop=True)


def walk_forward_predict(snap: pd.DataFrame, features: list) -> pd.Series:
    """
    Expanding walk-forward prediction of y at each rebalance, training only on
    quarters whose label horizon + embargo ends before the test date (purge).
    """
    dates = snap["date"].to_numpy()
    yhat = pd.Series(np.nan, index=snap.index)
    for i in range(len(snap)):
        test_date = snap["date"].iloc[i]
        cutoff = test_date - pd.Timedelta(days=int((HORIZON + EMBARGO) * 7 / 5))  # td->cal
        train = snap.iloc[:i][snap["date"].iloc[:i] <= cutoff]
        train = train.dropna(subset=features + ["y"])
        if len(train) < MIN_TRAIN:
            continue
        Xtr, ytr = train[features].to_numpy("float64"), train["y"].to_numpy("float64")
        xte = snap[features].iloc[[i]]
        if xte.isna().any(axis=None):
            continue
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(Xtr, ytr)
        yhat.iloc[i] = float(model.predict(xte.to_numpy("float64"))[0])
    return yhat


def _veto_mask(yhat: pd.Series) -> pd.Series:
    """Veto quarter i if yhat_i < ex-ante expanding VETO_PCTILE of past predictions."""
    veto = pd.Series(False, index=yhat.index)
    hist = []
    for i in range(len(yhat)):
        yi = yhat.iloc[i]
        if np.isfinite(yi) and len(hist) >= MIN_TRAIN:
            veto.iloc[i] = yi < np.quantile(hist, VETO_PCTILE)
        if np.isfinite(yi):
            hist.append(yi)
    return veto


def predictive_metrics(snap, yhat, crisis_q=0.10):
    """Directional accuracy + precision/recall of the veto on crisis quarters."""
    m = snap.assign(yhat=yhat.to_numpy()).dropna(subset=["yhat", "y"])
    if len(m) < MIN_TRAIN:
        return {}
    crisis = m["y"] <= m["y"].quantile(crisis_q)          # ex-post worst quarters (eval only)
    veto = _veto_mask(pd.Series(m["yhat"].to_numpy(), index=m.index)).to_numpy()
    tp = int((veto & crisis.to_numpy()).sum())
    fp = int((veto & ~crisis.to_numpy()).sum())
    fn = int((~veto & crisis.to_numpy()).sum())
    return {
        "n_pred": int(len(m)),
        "corr_yhat_y": float(np.corrcoef(m["yhat"], m["y"])[0, 1]),
        "dir_acc": float((np.sign(m["yhat"]) == np.sign(m["y"])).mean()),
        "n_crisis": int(crisis.sum()),
        "recall_crisis": tp / (tp + fn) if tp + fn else np.nan,
        "precision_veto": tp / (tp + fp) if tp + fp else np.nan,
    }


def feature_importance(snap, features):
    """Mean XGBoost gain importance over a few expanding folds (interpretability)."""
    imp = np.zeros(len(features))
    n = 0
    for cut in (0.5, 0.7, 0.9):
        k = int(len(snap) * cut)
        tr = snap.iloc[:k].dropna(subset=features + ["y"])
        if len(tr) < MIN_TRAIN:
            continue
        m = XGBRegressor(**XGB_PARAMS).fit(tr[features].to_numpy("float64"),
                                           tr["y"].to_numpy("float64"))
        imp += m.feature_importances_
        n += 1
    imp = imp / max(n, 1)
    return dict(sorted(zip(features, imp.tolist()), key=lambda kv: -kv[1]))


def run_central_test(processed_dir: str = "data/processed"):
    """Full ④-⑤: predictions, veto, backtests (gross+net) for A/B/Full + references."""
    snap = _snapshots(processed_dir)
    signal_rmt = pd.read_parquet(os.path.join(processed_dir, "signal_rmt.parquet"))
    weights = pd.read_parquet(os.path.join(processed_dir, "weights.parquet"))
    signal_rmt["date"] = pd.to_datetime(signal_rmt["date"])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    rebals = sorted(weights["rebalance_date"].unique())
    base = {k: pd.read_parquet(os.path.join(processed_dir, f"{k}.parquet"))
            for k in ("surface", "spots", "rates", "weights")}
    v1rmt_thr = exante_quantile_threshold(signal_rmt, rebals, q=0.5, warmup=12)

    out = {"snap_dates": snap["date"], "y": snap["y"], "arms": {}}
    for name, feats in FEATURE_SETS.items():
        yhat = walk_forward_predict(snap, feats)
        veto = _veto_mask(yhat)
        veto_dates = set(snap.loc[veto.to_numpy(), "date"])
        thr = v1rmt_thr.copy()
        for d in veto_dates:
            thr[pd.Timestamp(d)] = 1e9                     # force-skip vetoed quarters
        res = {}
        for tag, kw in [("gross", {}), ("net", dict(costs={}))]:
            bt = run_backtest(data={**{k: v.copy() for k, v in base.items()},
                                    "signal": signal_rmt.copy()}, threshold=thr, **kw)
            bt["quarterly"].to_parquet(
                os.path.join(processed_dir, f"backtest_ml_{name}_{tag}_quarterly.parquet"), index=False)
            bt["daily"].to_parquet(
                os.path.join(processed_dir, f"backtest_ml_{name}_{tag}_daily.parquet"), index=False)
            res[tag] = bt["quarterly"]
        out["arms"][name] = {
            "yhat": yhat, "veto": veto, "n_veto": int(veto.sum()),
            "pred": predictive_metrics(snap, yhat),
            "importance": feature_importance(snap, feats),
        }
    return out
