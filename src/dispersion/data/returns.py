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
    """
    Daily total returns (dividend- and split-adjusted) for given CRSP securities.

    Pure data extraction. `ret` is CRSP's holding-period return: it already accounts
    for dividends, splits and other distributions — no manual price adjustment needed.

    Parameters
    ----------
    db          : open wrds.Connection
    permnos     : list of CRSP permno identifiers
    date_start  : 'YYYY-MM-DD' inclusive
    date_end    : 'YYYY-MM-DD' inclusive

    Returns
    -------
    Wide DataFrame: index = date, columns = permno, values = simple daily return.
    Calendars are aligned across securities (NaN where a security did not trade).
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
    """
    Rolling annualised realized volatility per security, for each window.

    Uses log-returns ln(1+ret); a window value is NaN unless it has >= min_obs
    valid observations (guards against unstable estimates on sparse history).

    Parameters
    ----------
    returns : wide simple-return panel from get_returns (date x permno)
    windows : tuple of (window_length, min_obs) pairs (trading days)

    Returns
    -------
    dict {window_length: wide DataFrame of annualised vol (date x permno)}
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
    """
    Realized correlation matrix over the `window` days ending at `end_date`.

    Parameters
    ----------
    returns  : wide simple-return panel from get_returns (date x permno)
    end_date : last date of the window (inclusive)
    window   : window length in trading days
    min_obs  : minimum valid observations required
    method   : 'listwise' (complete-case -> PSD matrix, default) or 'pairwise'

    Returns
    -------
    N x N correlation DataFrame (index/columns = permno), or None if insufficient data.
    """
    log_ret = np.log1p(returns)
    win = log_ret.loc[:end_date].tail(window)
    # drop securities with insufficient coverage in this window
    win = win.loc[:, win.notna().sum() >= min_obs]
    if win.shape[1] < 2:
        return None

    if method == "listwise":
        win = win.dropna(axis=0, how="any")  # complete-case rows -> PSD
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
    """
    Collapse a correlation matrix into a single average correlation rho-bar.

    'weighted' (default) is the formula-consistent average that appears in the index
    variance decomposition (matches the implied-correlation definition):
        rho_bar = (v' R v - sum v_i^2) / ((sum v_i)^2 - sum v_i^2),  v_i = w_i * sigma_i
    'equal' is the plain mean of off-diagonal pairwise correlations.

    Parameters
    ----------
    corr    : N x N correlation matrix (index/columns = permno)
    weights : cap weights (Series indexed by permno) — frozen rebalance weights
    vols    : realized vols sigma_i (Series indexed by permno), same date as the window
    method  : 'weighted' (default) or 'equal'

    Returns
    -------
    rho_bar (float), or NaN if not computable.
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
