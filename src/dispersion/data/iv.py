"""
Implied-volatility extraction from OptionMetrics `vsurfd` (volatility surface).

Usage:
    from dispersion.data.iv import get_iv, get_surface
    df = get_iv(db, secids=[108105, 101594], date_start="2020-01-01", date_end="2020-12-31")
"""
import numpy as np
import pandas as pd
import wrds


def get_iv(
    db: wrds.Connection,
    secids: list[int],
    date_start: str,
    date_end: str,
    days: int = 91,
) -> pd.DataFrame:
    """ATM (delta ±50) implied vol at a fixed maturity for the given securities.

    Pure extraction — no rebalancing or membership logic. Works the same for the
    index (secid 108105) and its constituents.

    One row per (secid, date): secid, date, iv_call_50 (delta +50), iv_put_50
    (delta -50), iv_atm (mean of the two legs, NaN if either is missing).
    """
    secid_list = ",".join(str(int(s)) for s in secids)
    y0, y1 = int(date_start[:4]), int(date_end[:4])

    frames = []
    for year in range(y0, y1 + 1):
        # clip the window to this year's slice of vsurfd
        lo = max(date_start, f"{year}-01-01")
        hi = min(date_end, f"{year}-12-31")
        q = f"""
        SELECT secid, date, delta, impl_volatility
        FROM optionm.vsurfd{year}
        WHERE secid IN ({secid_list})
          AND days = {days}
          AND delta IN (50, -50)
          AND date BETWEEN '{lo}' AND '{hi}'
          AND impl_volatility IS NOT NULL
          AND impl_volatility > 0
        """
        frames.append(db.raw_sql(q))

    long = pd.concat(frames, ignore_index=True)
    if long.empty:
        return pd.DataFrame(columns=["secid", "date", "iv_call_50", "iv_put_50", "iv_atm"])

    # pivot the two legs (delta +50 -> call, -50 -> put) into columns
    wide = (
        long.pivot_table(index=["secid", "date"], columns="delta", values="impl_volatility")
        .rename(columns={50.0: "iv_call_50", -50.0: "iv_put_50"})
        .reset_index()
    )
    wide.columns.name = None
    # make sure both leg columns exist even if a delta never showed up.
    # use np.nan, not pd.NA — an object column makes mean(skipna=False) raise
    for col in ("iv_call_50", "iv_put_50"):
        if col not in wide.columns:
            wide[col] = np.nan

    # need both legs or it's NaN
    wide["iv_atm"] = wide[["iv_call_50", "iv_put_50"]].mean(axis=1, skipna=False)

    return wide[["secid", "date", "iv_call_50", "iv_put_50", "iv_atm"]].sort_values(
        ["secid", "date"]
    ).reset_index(drop=True)


def get_surface(
    db: wrds.Connection,
    secids: list[int],
    date_start: str,
    date_end: str,
    days: tuple[int, ...] = (10, 30, 60, 91),
) -> pd.DataFrame:
    """Standardised surface (both ±50Δ legs) at several maturity pillars.

    Marking-grade data: a 91d position ages into shorter residual maturities, so
    we pull the sub-91 pillars too. Raw vendor values, no cleaning here.

    One row per (secid, date, days, cp_flag): secid, date, days, cp_flag
    ('C'/'P'), iv, premium, strike.
    """
    secid_list = ",".join(str(int(s)) for s in secids)
    days_list = ",".join(str(int(d)) for d in days)
    y0, y1 = int(date_start[:4]), int(date_end[:4])

    frames = []
    for year in range(y0, y1 + 1):
        lo = max(date_start, f"{year}-01-01")
        hi = min(date_end, f"{year}-12-31")
        q = f"""
        SELECT secid, date, days, cp_flag,
               impl_volatility AS iv, impl_premium AS premium, impl_strike AS strike
        FROM optionm.vsurfd{year}
        WHERE secid IN ({secid_list})
          AND days IN ({days_list})
          AND delta IN (50, -50)
          AND date BETWEEN '{lo}' AND '{hi}'
          AND impl_volatility IS NOT NULL
          AND impl_volatility > 0
        """
        frames.append(db.raw_sql(q))

    surf = pd.concat(frames, ignore_index=True)
    if surf.empty:
        return pd.DataFrame(columns=["secid", "date", "days", "cp_flag", "iv", "premium", "strike"])

    surf["cp_flag"] = surf["cp_flag"].str.strip()
    surf["days"] = surf["days"].astype(int)
    return surf.sort_values(["secid", "date", "days", "cp_flag"]).reset_index(drop=True)
