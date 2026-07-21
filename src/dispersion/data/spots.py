"""Daily spot prices from OptionMetrics `secprd`.

`secprd` is the price series OptionMetrics builds the surface from, so its closes
share the split convention of `impl_strike`. Marking and intrinsic settlement
need S and K on the same convention (|S - K|); mid-quarter splits are handled
with `cfadj`.

Usage:
    from dispersion.data.spots import get_spots
    df = get_spots(db, [108105], "2020-01-01", "2020-12-31")
"""
import pandas as pd
import wrds


def get_spots(
    db: wrds.Connection,
    secids: list[int],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """Close price + cumulative split-adjustment factor per (secid, date)."""
    secid_list = ",".join(str(int(s)) for s in secids)
    y0, y1 = int(date_start[:4]), int(date_end[:4])

    frames = []
    for year in range(y0, y1 + 1):
        lo = max(date_start, f"{year}-01-01")
        hi = min(date_end, f"{year}-12-31")
        q = f"""
        SELECT secid, date, close, cfadj
        FROM optionm.secprd{year}
        WHERE secid IN ({secid_list})
          AND date BETWEEN '{lo}' AND '{hi}'
          AND close IS NOT NULL
        """
        frames.append(db.raw_sql(q))

    spots = pd.concat(frames, ignore_index=True)
    if spots.empty:
        return pd.DataFrame(columns=["secid", "date", "close", "cfadj"])

    spots["close"] = spots["close"].abs()  # close can be signed (midpoint convention); take abs
    return spots.sort_values(["secid", "date"]).reset_index(drop=True)
