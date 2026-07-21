"""
Daily RMT series: a cleaned rho-bar variant of the signal plus spectral regime
features for the ML layer. Builds rmt_daily.parquet and signal_rmt.parquet.

signal_rmt keeps a `premium` column only to match signal.parquet's schema —
don't use it. A 252d trailing leg shifted by 63 rows overlaps itself ~75%, so
that premium is mechanically contaminated; only `signal` is safe to trade on.

Usage:
    from dispersion.rmt.daily import build_rmt_daily, build_signal_rmt
    build_rmt_daily()      # ~7,200 window eigendecompositions, a few minutes
    build_signal_rmt()
"""
import os

import numpy as np
import pandas as pd

from ..data.returns import average_correlation
from .cleaning import (EWMA_LAMBDA, MIN_OBS, T_WIN, corr_window, devolatilise,
                       laloux_clip, spectral_features)


def _rotation(v1: pd.Series, v1_prev: pd.Series | None) -> float:
    """
    Day-to-day rotation of the dominant eigenvector: 1 - |<v1_t, v1_{t-1}>| over
    the common names (each subvector renormalised). NaN with no previous vector
    or no overlap.
    """
    if v1_prev is None:
        return np.nan
    common = v1.index.intersection(v1_prev.index)
    if len(common) < 10:
        return np.nan
    a = v1.loc[common].to_numpy(dtype="float64")
    b = v1_prev.loc[common].to_numpy(dtype="float64")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(1.0 - abs(a @ b) / (na * nb))


def build_rmt_daily(
    processed_dir: str = "data/processed",
    t_win: int = T_WIN,
    min_obs: int = MIN_OBS,
    lam: float = EWMA_LAMBDA,
    out_file: str | None = "rmt_daily.parquet",
) -> pd.DataFrame:
    """Daily cleaned rho-bar + spectral features, frozen universe per quarter."""
    returns = pd.read_parquet(os.path.join(processed_dir, "returns.parquet"))
    weights = pd.read_parquet(os.path.join(processed_dir, "weights.parquet"))
    signal = pd.read_parquet(os.path.join(processed_dir, "signal.parquet"))

    returns["date"] = pd.to_datetime(returns["date"])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    spine = pd.DatetimeIndex(pd.to_datetime(signal["date"])).sort_values()

    ret = returns.pivot(index="date", columns="permno", values="ret").sort_index().astype("float64")
    z = devolatilise(ret, lam=lam)
    sig252 = (np.log1p(ret)).rolling(t_win, min_periods=min_obs).std() * np.sqrt(252.0)

    rebals = sorted(weights["rebalance_date"].unique())
    rows = []
    for k, reb in enumerate(rebals):
        reb = pd.Timestamp(reb)
        nxt = pd.Timestamp(rebals[k + 1]) if k + 1 < len(rebals) else spine[-1] + pd.Timedelta(days=1)
        active = spine[(spine >= reb) & (spine < nxt)]
        wq = weights[weights["rebalance_date"] == reb]
        wmap = wq.set_index("permno")["weight"].astype("float64")
        v1_prev, lam1_prev = None, np.nan

        for d in active:
            res = corr_window(z, d, wmap.index.tolist(), t_win=t_win, min_obs=min_obs)
            if res is None:
                rows.append((d,) + (np.nan,) * 12 + (0,))
                v1_prev, lam1_prev = None, np.nan
                continue
            C, t_eff = res                              # effective T for the MP edge
            C_clean, diag = laloux_clip(C, t_win=t_eff)
            feats, v1 = spectral_features(C, t_win=t_eff)

            kept = C.columns
            w_kept = wmap.reindex(kept)
            s_kept = sig252.loc[d, kept]                # KeyError if d is missing, rather than a stale row
            rho_raw = average_correlation(C, w_kept, s_kept, "weighted")
            rho_clean = average_correlation(C_clean, w_kept, s_kept, "weighted")

            # cross-name dispersion of average correlation
            A = C.to_numpy(dtype="float64")
            corr_xdisp = float(np.std(A.mean(axis=1)))
            # Mahalanobis turbulence on the cleaned inverse — inverting a noisy C
            # is where cleaning actually matters
            zrow = z.loc[d, kept].to_numpy(dtype="float64")
            mask = np.isfinite(zrow)
            if mask.sum() >= 30:
                Ainv = np.linalg.pinv(C_clean.to_numpy(dtype="float64")[np.ix_(mask, mask)])
                zc = zrow[mask]
                turb = float(zc @ Ainv @ zc / mask.sum())
            else:
                turb = np.nan

            rows.append((d, rho_raw, rho_clean, feats["lam1_share"], feats["k_signal"],
                         feats["absorption_top"],
                         feats["lam1_share"] - lam1_prev if np.isfinite(lam1_prev) else np.nan,
                         _rotation(v1, v1_prev), feats["lam2_share"], feats["spec_entropy"],
                         feats["pr_v1"], corr_xdisp, turb, len(kept)))
            v1_prev, lam1_prev = v1, feats["lam1_share"]

    out = pd.DataFrame(rows, columns=["date", "rho_rmt_raw", "rho_rmt_clean", "lam1_share",
                                      "k_signal", "absorption_top", "d_lam1", "rotation",
                                      "lam2_share", "spec_entropy", "pr_v1", "corr_xdisp",
                                      "turb", "n_names_rmt"])
    if out_file:
        out.to_parquet(os.path.join(processed_dir, out_file), index=False)
    return out


def build_signal_rmt(
    processed_dir: str = "data/processed",
    horizon: int = 63,
    out_file: str | None = "signal_rmt.parquet",
) -> pd.DataFrame:
    """
    Signal variant whose trailing realised leg is the RMT-cleaned 252d rho-bar.
    Same columns as signal.parquet so the engine gate works unchanged.
    """
    sig = pd.read_parquet(os.path.join(processed_dir, "signal.parquet"))
    rmt = pd.read_parquet(os.path.join(processed_dir, "rmt_daily.parquet"))
    sig["date"] = pd.to_datetime(sig["date"])
    rmt["date"] = pd.to_datetime(rmt["date"])

    m = sig[["date", "rho_implied", "n_names"]].merge(
        rmt[["date", "rho_rmt_clean", "n_names_rmt"]], on="date", how="left", validate="1:1"
    ).sort_values("date").reset_index(drop=True)
    m["rho_trailing"] = m["rho_rmt_clean"].astype("float64")
    m["rho_forward"] = m["rho_trailing"].shift(-horizon)
    m["premium"] = m["rho_implied"] - m["rho_forward"]
    m["signal"] = m["rho_implied"] - m["rho_trailing"]
    m = m[["date", "rho_implied", "rho_trailing", "rho_forward", "premium", "signal",
           "n_names", "n_names_rmt"]]
    if out_file:
        m.to_parquet(os.path.join(processed_dir, out_file), index=False)
    return m
