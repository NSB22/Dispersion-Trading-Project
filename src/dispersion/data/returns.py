"""
Returns extraction from CRSP `dsf` and realized-volatility computation.

Usage:
    from dispersion.data.returns import get_returns, realized_vol
    px = get_returns(db, permnos=[14593, 10107], date_start="2019-01-01", date_end="2020-12-31")
    vol = realized_vol(px)          # {21: df, 63: df}, annualised
"""
import numpy as np
import pandas as pd
import wrds

TRADING_DAYS = 252


def get_returns(
    db: wrds.Connection,
    permnos: list[int],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """Daily total returns (dividend- and split-adjusted) for the given permnos.

    CRSP `ret` is a holding-period return, so dividends and splits are already
    baked in — no manual price adjustment.

    Wide DataFrame: index = date, columns = permno, values = daily return; NaN
    where a security did not trade.
    """
    permno_list = ",".join(str(int(p)) for p in permnos)
    q = f"""
    SELECT permno, date, ret
    FROM crsp.dsf
    WHERE permno IN ({permno_list})
      AND date BETWEEN '{date_start}' AND '{date_end}'
      AND ret IS NOT NULL
    """
    long = db.raw_sql(q)
    if long.empty:
        return pd.DataFrame()

    wide = long.pivot(index="date", columns="permno", values="ret").sort_index()
    wide.columns.name = None
    return wide


def realized_vol(
    returns: pd.DataFrame,
    windows: tuple[tuple[int, int], ...] = ((21, 17), (63, 50)),
) -> dict[int, pd.DataFrame]:
    """Rolling annualised realized vol per security, one DataFrame per window.

    Uses log-returns; a window is NaN until it has at least `min_obs`
    observations. `windows` is a tuple of (length, min_obs) pairs in trading days.
    Returns dict {window_length: wide DataFrame (date x permno)}.
    """
    log_ret = np.log1p(returns)
    out = {}
    for window, min_obs in windows:
        vol = log_ret.rolling(window=window, min_periods=min_obs).std(ddof=1)
        out[window] = vol * np.sqrt(TRADING_DAYS)
    return out


def realized_corr_matrix(
    returns: pd.DataFrame,
    end_date: str,
    window: int = 63,
    min_obs: int = 50,
    method: str = "listwise",
) -> pd.DataFrame | None:
    """Realized correlation matrix over the `window` days ending at `end_date`.

    `method` is 'listwise' (complete-case rows -> PSD) or 'pairwise'. Returns an
    N x N DataFrame (index/columns = permno), or None if there isn't enough data.
    """
    log_ret = np.log1p(returns)
    win = log_ret.loc[:end_date].tail(window)
    # drop names with too few obs in the window
    win = win.loc[:, win.notna().sum() >= min_obs]
    if win.shape[1] < 2:
        return None

    if method == "listwise":
        win = win.dropna(axis=0, how="any")  # keep only complete rows so it stays PSD
        if len(win) < min_obs:
            return None
        corr = win.corr()
    elif method == "pairwise":
        corr = win.corr(min_periods=min_obs)
    else:
        raise ValueError(f"method must be 'listwise' or 'pairwise', got {method!r}")
    return corr


def average_correlation(
    corr: pd.DataFrame,
    weights: pd.Series,
    vols: pd.Series,
    method: str = "weighted",
) -> float:
    """Collapse a correlation matrix into a single average correlation rho-bar.

    'weighted' (default) is the formula-consistent average from the index
    variance decomposition (matches the implied-correlation definition):
        rho_bar = (v' R v - sum v_i^2) / ((sum v_i)^2 - sum v_i^2),  v_i = w_i * sigma_i
    'equal' is the plain mean of off-diagonal pairwise correlations.

    `weights` are the frozen rebalance cap weights; `vols` are realized sigma_i on
    the window's end date. Returns NaN if not computable.
    """
    permnos = corr.index
    w = weights.reindex(permnos)
    s = vols.reindex(permnos)
    valid = w.notna() & s.notna()
    permnos = permnos[valid]
    if len(permnos) < 2:
        return float("nan")

    R = corr.loc[permnos, permnos].to_numpy()

    if method == "weighted":
        v = (w[permnos] * s[permnos]).to_numpy()
    elif method == "equal":
        v = np.ones(len(permnos))
    else:
        raise ValueError(f"method must be 'weighted' or 'equal', got {method!r}")

    sum_sq = np.sum(v**2)
    num = v @ R @ v - sum_sq          # off-diagonal weighted sum of rho_ij
    den = v.sum() ** 2 - sum_sq       # off-diagonal weighted sum (rho_ij = 1)
    return float(num / den) if den != 0 else float("nan")
