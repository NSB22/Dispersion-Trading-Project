"""
ML dataset — pre-registered features and label (plan.md; protocol v2, enriched
17 Jul 2026 BEFORE any walk-forward result was computed).

Feature blocks (all from existing parquets — no new WRDS pull):
  A. deep spectral (from rmt_daily): lam1/lam2 shares, k_signal, absorption,
     entropy, participation ratio, rotation, corr cross-dispersion, and the
     Mahalanobis turbulence computed on the CLEANED inverse;
  B. vol structure (surface/iv_index): SPX term slope 30→91d, ±50Δ skew proxy,
     vol-of-vol;
  C. realised & VRP (spots SPX): rv21/rv63, variance risk premium, 252d
     drawdown, momentum;
  D. IV cross-section (iv_components): weighted component IV, IV dispersion;
  E. correlation levels/dynamics + ex-ante signal percentile + era cost.
Label: y_spike(t) = rho_trail63(t+63) − rho_trail63(t) — TRAINING ONLY, purged.

Dimensionality discipline (pre-registered): full set for XGBoost; the frozen
8-feature core set for GMM/HMM lives in CORE_SET below.

Usage:
    from dispersion.ml.features import build_ml_dataset
    df = build_ml_dataset()        # writes data/processed/ml_dataset.parquet
"""
import os

import numpy as np
import pandas as pd

from ..backtest.engine import SPX_SECID, parametric_spread

HORIZON = 63          # trading days ≈ the 91-calendar-day decision horizon
D21 = 21              # short dynamics window (pre-registered)

# frozen core set for the Gaussian models (GMM/HMM) — full-cov in dim 32 would
# not survive ~2,000 walk-forward observations
CORE_SET = ["f_lam1", "f_dlam1_21", "f_turb21", "f_vrp", "f_term_slope",
            "f_anchor_gap", "f_iv_spx", "f_rot21"]


def build_ml_dataset(
    processed_dir: str = "data/processed",
    out_file: str | None = "ml_dataset.parquet",
) -> pd.DataFrame:
    sig = pd.read_parquet(os.path.join(processed_dir, "signal.parquet"))
    sigr = pd.read_parquet(os.path.join(processed_dir, "signal_rmt.parquet"))
    rmt = pd.read_parquet(os.path.join(processed_dir, "rmt_daily.parquet"))
    ivx = pd.read_parquet(os.path.join(processed_dir, "iv_index.parquet"))
    surface = pd.read_parquet(os.path.join(processed_dir, "surface.parquet"))
    spots = pd.read_parquet(os.path.join(processed_dir, "spots.parquet"))
    ivc = pd.read_parquet(os.path.join(processed_dir, "iv_components.parquet"))
    wts = pd.read_parquet(os.path.join(processed_dir, "weights.parquet"))
    vix = pd.read_parquet(os.path.join(processed_dir, "vix.parquet"))  # CBOE (cboe.cboe)
    for df in (sig, sigr, rmt, ivx, surface, spots, ivc, vix):
        df["date"] = pd.to_datetime(df["date"])

    # ---- bloc B: SPX term slope (30d -> 91d ATM pillars, C/P averaged) ------- #
    spx_surf = surface[(surface["secid"] == SPX_SECID) & (surface["days"].isin([30, 91]))]
    piv = (spx_surf.groupby(["date", "days"])["iv"].mean().unstack("days")
           .astype("float64"))
    term = ((piv[91] - piv[30]) / piv[30]).rename("f_term_slope")

    # ---- bloc C: SPX realised block (vendor closes, same source as strikes) -- #
    spx_close = (spots[spots["secid"] == SPX_SECID]
                 .drop_duplicates("date").set_index("date")["close"]
                 .astype("float64").sort_index())
    lr = np.log(spx_close).diff()
    rv21 = (lr.rolling(D21, min_periods=15).std() * np.sqrt(252.0)).rename("f_rv21")
    rv63 = (lr.rolling(63, min_periods=50).std() * np.sqrt(252.0)).rename("f_rv63")
    dd252 = (spx_close / spx_close.rolling(252, min_periods=100).max() - 1.0).rename("f_dd252")
    mom63 = (spx_close.pct_change(63)).rename("f_mom63")

    # ---- bloc D: component-IV cross-section (weighted, renormalised) --------- #
    ivw = ivc.merge(wts[["rebalance_date", "permno", "weight"]],
                    on=["rebalance_date", "permno"], how="left")
    ivw = ivw.dropna(subset=["iv_atm"])
    g = ivw.assign(ws=ivw["weight"] * ivw["iv_atm"]).groupby("date")
    iv_comp = (g["ws"].sum() / g["weight"].sum()).astype("float64").rename("f_iv_comp")
    iv_xdisp = g["iv_atm"].std().astype("float64").rename("f_iv_xdisp")

    # ---- assemble on the master spine ---------------------------------------- #
    m = (
        sig[["date", "rho_implied", "rho_trailing", "signal"]]
        .rename(columns={"rho_trailing": "rho_trail63", "signal": "f_sig_base"})
        .merge(sigr[["date", "signal"]].rename(columns={"signal": "f_sig_rmt"}),
               on="date", validate="1:1")
        .merge(rmt[["date", "rho_rmt_clean", "lam1_share", "k_signal", "absorption_top",
                    "rotation", "lam2_share", "spec_entropy", "pr_v1", "corr_xdisp",
                    "turb"]], on="date", validate="1:1")
        .merge(ivx[["date", "iv_atm", "iv_call_50", "iv_put_50"]], on="date",
               how="left", validate="1:1")
        .sort_values("date").reset_index(drop=True)
    )
    for s in (term, rv21, rv63, dd252, mom63, iv_comp, iv_xdisp):
        m = m.merge(s.reset_index(), on="date", how="left")
    m = m.merge(vix, on="date", how="left", validate="1:1")

    out = pd.DataFrame({"date": m["date"]})
    # A. spectral
    out["f_lam1"] = m["lam1_share"]
    out["f_lam2"] = m["lam2_share"]
    out["f_k"] = m["k_signal"]
    out["f_abs"] = m["absorption_top"]
    out["f_entropy"] = m["spec_entropy"]
    out["f_prv1"] = m["pr_v1"]
    out["f_xdisp"] = m["corr_xdisp"]
    out["f_dlam1_21"] = m["lam1_share"].diff(D21)
    out["f_dlam1_63"] = m["lam1_share"].diff(63)
    out["f_rot21"] = m["rotation"].rolling(D21, min_periods=10).mean()
    out["f_turb"] = m["turb"]
    out["f_turb21"] = m["turb"].rolling(D21, min_periods=10).mean()
    # B. vol structure
    out["f_iv_spx"] = m["iv_atm"]
    out["f_div_21"] = m["iv_atm"].diff(D21)
    out["f_vix"] = m["vix"] / 100.0                 # CBOE VIX (MFIV 30d), decimal units
    out["f_term_slope"] = m["f_term_slope"]
    out["f_skew50"] = m["iv_put_50"] - m["iv_call_50"]
    out["f_volofvol"] = m["iv_atm"].rolling(D21, min_periods=15).std()
    # C. realised & VRP
    out["f_rv21"] = m["f_rv21"]
    out["f_rv63"] = m["f_rv63"]
    out["f_vrp"] = m["iv_atm"] - m["f_rv63"]
    out["f_dd252"] = m["f_dd252"]
    out["f_mom63"] = m["f_mom63"]
    # D. IV cross-section
    out["f_iv_comp"] = m["f_iv_comp"]
    out["f_iv_xdisp"] = m["f_iv_xdisp"]
    # E. correlation levels / signal / cost
    out["f_rho_imp"] = m["rho_implied"]
    out["f_drho_imp_21"] = m["rho_implied"].diff(D21)   # le marché COMMENCE à pricer le stress
    out["f_rho_trail63"] = m["rho_trail63"]
    out["f_rho_clean252"] = m["rho_rmt_clean"]
    out["f_drho_21"] = m["rho_trail63"].diff(D21)
    out["f_sig_base"] = m["f_sig_base"]
    out["f_sig_rmt"] = m["f_sig_rmt"]
    out["f_anchor_gap"] = m["rho_trail63"] - m["rho_rmt_clean"]
    out["f_sig_rmt_pct"] = (m["f_sig_rmt"].expanding(min_periods=252)
                            .rank(pct=True))          # ex-ante expanding percentile
    year = m["date"].dt.year
    out["f_cost_era"] = [
        0.5 * (parametric_spread(y, "spx")
               + 0.7 * parametric_spread(y, "large") + 0.3 * parametric_spread(y, "small"))
        for y in year
    ]
    # label — TRAINING ONLY, purged in every fit
    out["y_spike"] = m["rho_trail63"].shift(-HORIZON) - m["rho_trail63"]

    # hard float64 everywhere (nullable dtypes break numpy/sklearn — recurring lesson)
    num_cols = [c for c in out.columns if c != "date"]
    out[num_cols] = out[num_cols].astype("float64")

    if out_file:
        out.to_parquet(os.path.join(processed_dir, out_file), index=False)
    return out
