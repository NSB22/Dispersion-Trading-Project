"""
Implied-volatility extraction from OptionMetrics `vsurfd` (volatility surface).

Usage:
    from dispersion.data.iv import get_iv
    df = get_iv(db, secids=[108105, 101594], date_start="2020-01-01", date_end="2020-12-31")
"""
import pandas as pd
import wrds


def get_iv(
    db: wrds.Connection,
    secids: list[int],
    date_start: str,
    date_end: str,
    days: int = 91,
) -> pd.DataFrame:
    """
    Extract ATM (delta +-50) implied vol at a fixed maturity for given securities.

    Pure data extraction — no knowledge of rebalancing or point-in-time membership.
    Works identically for the index (secid 108105) and its constituents.

    Parameters
    ----------
    db          : open wrds.Connection
    secids      : list of OptionMetrics security ids
    date_start  : 'YYYY-MM-DD' inclusive
    date_end    : 'YYYY-MM-DD' inclusive
    days        : maturity in days (default 91, native in vsurfd)

    Returns
    -------
    Tidy DataFrame, one row per (secid, date):
        secid        – OptionMetrics security id
        date         – trading date
        iv_call_50   – implied vol at delta +50 (the call leg)
        iv_put_50    – implied vol at delta -50 (the put leg)
        iv_atm       – mean of the two legs (NaN if either leg missing — strict ATM)
    """
    secid_list = ",".join(str(int(s)) for s in secids)
    y0, y1 = int(date_start[:4]), int(date_end[:4])

    frames = []
    for year in range(y0, y1 + 1):
        # clip the window to this year's table
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
    # ensure both leg columns exist even if one delta never appeared
    for col in ("iv_call_50", "iv_put_50"):
        if col not in wide.columns:
            wide[col] = pd.NA

    # strict ATM: mean only when BOTH legs present
    wide["iv_atm"] = wide[["iv_call_50", "iv_put_50"]].mean(axis=1, skipna=False)

    return wide[["secid", "date", "iv_call_50", "iv_put_50", "iv_atm"]].sort_values(
        ["secid", "date"]
    ).reset_index(drop=True)
