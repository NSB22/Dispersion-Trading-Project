"""
Implied correlation, correlation-risk premium and dispersion signal.

Inverts the one-factor index-variance identity (DMV 2009, eq. 2) on ATM 91-day
IVs — the same functional form as the realised rho-bar (option B), so the
spread is apples-to-apples — then builds the two window-matched series:

    premium(t) = rho_implied(t) - rho_trailing(t + horizon)    (ex-post validation)
    signal(t)  = rho_implied(t) - rho_trailing(t)              (ex-ante, tradeable)

Implementation choices (README §5.1): daily weight renormalisation over the
names with a valid IV (+ coverage floor), raw values stored (no clipping).
Validated in notebooks/03_implied_correlation.ipynb.

Usage:
    from dispersion.signal.implied_corr import build_signal
    sig = build_signal()          # reads data/processed/, writes signal.parquet
"""
import os

import numpy as np
import pandas as pd


def implied_correlation(
    iv_components: pd.DataFrame,
    weights: pd.DataFrame,
    iv_index: pd.DataFrame,
    n_min: int = 90,
) -> pd.DataFrame:
    """
    Daily implied correlation from the one-factor inversion.

    rho_implied = (sigma_I^2 - S2) / (S1^2 - S2), with S1 = sum(w_hat * sigma)
    and S2 = sum(w_hat^2 * sigma^2) over the names with a valid IV that day,
    where w_hat renormalises the frozen rebalance weights over that set.

    Parameters
    ----------
    iv_components : long frame [rebalance_date, date, permno, secid, iv_atm]
    weights       : long frame [rebalance_date, permno, weight, ...]
    iv_index      : frame [date, iv_atm] (SPX leg)
    n_min         : coverage floor — the day is invalidated below n_min names

    Returns
    -------
    DataFrame indexed by date with columns [rho_implied, n_names].
    """
    m = iv_components.merge(
        weights[["rebalance_date", "permno", "weight"]],
        on=["rebalance_date", "permno"],
        how="left",
        validate="m:1",
    )
    if m["weight"].isna().any():
        raise ValueError("component IV rows without a matching universe weight")

    valid = m.dropna(subset=["iv_atm"]).copy()
    valid["w_sig"] = valid["weight"] * valid["iv_atm"]
    valid["w2_sig2"] = valid["weight"] ** 2 * valid["iv_atm"] ** 2

    g = valid.groupby("date").agg(
        w_sum=("weight", "sum"),
        s1_raw=("w_sig", "sum"),
        s2_raw=("w2_sig2", "sum"),
        n_names=("weight", "size"),
    )
    s1 = g["s1_raw"] / g["w_sum"]  # renormalised weighted-average component IV
    s2 = g["s2_raw"] / g["w_sum"] ** 2  # renormalised diagonal term

    sig_i = iv_index.set_index("date")["iv_atm"].reindex(g.index)
    n_missing = int(sig_i.isna().sum())
    if n_missing:  # guard: a silent NaN here would masquerade as a coverage-floor day
        raise ValueError(f"{n_missing} component-IV dates have no index IV — inconsistent inputs")
    rho = (sig_i ** 2 - s2) / (s1 ** 2 - s2)
    rho[g["n_names"] < n_min] = np.nan  # coverage floor (guard — never hit so far)

    return pd.DataFrame(
        {"rho_implied": rho.astype("float64"), "n_names": g["n_names"].astype("Int64")}
    )


def build_signal(
    processed_dir: str = "data/processed",
    horizon: int = 63,
    n_min: int = 90,
    out_file: str | None = "signal.parquet",
) -> pd.DataFrame:
    """
    Assemble the full signal table and (optionally) write it to parquet.

    Columns: date, rho_implied, rho_trailing, rho_forward, premium, signal,
    n_names. rho_forward(t) = rho_trailing shifted back `horizon` rows of the
    master calendar (window-matching; the shift crosses the Aug-2020 vendor
    gap, documented).
    """
    iv_index = pd.read_parquet(os.path.join(processed_dir, "iv_index.parquet"))
    iv_comp = pd.read_parquet(os.path.join(processed_dir, "iv_components.parquet"))
    weights = pd.read_parquet(os.path.join(processed_dir, "weights.parquet"))
    rcorr = pd.read_parquet(os.path.join(processed_dir, "realized_corr.parquet"))

    iv_comp["secid"] = iv_comp["secid"].astype("Int64")
    iv_comp["permno"] = iv_comp["permno"].astype("Int64")
    weights["permno"] = weights["permno"].astype("Int64")
    for df in (iv_index, iv_comp, weights, rcorr):
        for c in ("date", "rebalance_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])

    rho = implied_correlation(iv_comp, weights, iv_index, n_min=n_min)

    spine = rcorr.set_index("date").sort_index()  # master calendar
    lost = rho.index.difference(spine.index)
    extra = spine.index.difference(rho.index)
    if len(lost) or len(extra):  # guard: the right-join must be lossless (verified invariant)
        raise ValueError(
            f"calendar mismatch: {len(lost)} rho dates off-spine, {len(extra)} spine dates without rho"
        )
    sig = rho.join(spine[["rho_bar"]], how="right").sort_index()
    sig = sig.rename(columns={"rho_bar": "rho_trailing"})
    sig["rho_forward"] = sig["rho_trailing"].shift(-horizon)
    sig["premium"] = sig["rho_implied"] - sig["rho_forward"]
    sig["signal"] = sig["rho_implied"] - sig["rho_trailing"]

    cols = ["rho_implied", "rho_trailing", "rho_forward", "premium", "signal"]
    sig[cols] = sig[cols].astype("float64")
    sig["n_names"] = sig["n_names"].astype("Int64")
    sig = sig.reset_index(names="date")[["date"] + cols + ["n_names"]]

    if out_file:
        sig.to_parquet(os.path.join(processed_dir, out_file), index=False)
    return sig
