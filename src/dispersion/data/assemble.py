"""Build the clean backtest dataset: pull each piece, align to one calendar,
apply the cleaning rules, and write the parquets to data/processed/.

Usage:
    from dispersion.data.assemble import build_dataset
    out = build_dataset(db, "1996-01-01", "2024-12-31")
"""
import os
import numpy as np
import pandas as pd
import wrds

from .universe import get_universe
from .iv import get_iv, get_surface
from .spots import get_spots
from .returns import (
    get_returns,
    realized_vol,
    realized_corr_matrix,
    average_correlation,
)

SPX_SECID = 108105


# ----------------------------------------------------------------------------- #
# Calendar helpers
# ----------------------------------------------------------------------------- #
def _master_calendar(db, iv_index: pd.DataFrame, date_start: str, date_end: str) -> pd.DatetimeIndex:
    """Strict intersection of OptionMetrics (SPX IV) and CRSP (dsf) trading days."""
    iv_dates = pd.to_datetime(iv_index["date"]).unique()
    dsf = db.raw_sql(
        f"SELECT DISTINCT date FROM crsp.dsf WHERE date BETWEEN '{date_start}' AND '{date_end}'"
    )
    dsf_dates = pd.to_datetime(dsf["date"]).unique()
    common = pd.DatetimeIndex(sorted(set(iv_dates) & set(dsf_dates)))
    return common


def _rebalance_dates(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each quarter present in the master calendar."""
    s = pd.Series(calendar, index=calendar)
    grp = s.groupby([calendar.year, calendar.quarter]).max()
    return [pd.Timestamp(x) for x in grp.values]


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def build_dataset(
    db: wrds.Connection,
    date_start: str = "1996-01-01",
    date_end: str = "2024-12-31",
    n: int = 100,
    iv_days: int = 91,
    corr_window: int = 63,
    corr_min_obs: int = 50,
    corr_method: str = "listwise",
    rho_method: str = "weighted",
    ffill_limit: int = 3,
    iv_pct: tuple[float, float] = (0.001, 0.999),
    surface_days: tuple[int, ...] = (10, 30, 60, 91),
    returns_start: str = "1995-01-01",
    out_dir: str | None = "data/processed",
) -> dict[str, pd.DataFrame]:
    """Build and optionally save the clean backtest dataset; returns a dict of DataFrames.

    Writes iv_index, iv_components, weights, realized_vol, realized_corr,
    corr_matrices (cleaned, signal-grade) plus surface, rates, returns (marking/
    RMT grade, stored raw). `returns_start` predates `date_start` so the 252-day
    RMT windows have warm history from the first rebalance.
    """

    # --- 1. SPX IV (the index leg) + master calendar + rebalance schedule -----
    iv_index = get_iv(db, [SPX_SECID], date_start, date_end, days=iv_days)
    iv_index["date"] = pd.to_datetime(iv_index["date"])
    calendar = _master_calendar(db, iv_index, date_start, date_end)
    rebals = _rebalance_dates(calendar)

    iv_index = iv_index[iv_index["date"].isin(calendar)].drop(columns="secid")
    # bounded ffill on the index series; no outlier clipping on the index
    iv_index = (
        iv_index.set_index("date").reindex(calendar).ffill(limit=ffill_limit).reset_index(names="date")
    )

    # --- 2. Per-quarter loop -------------------------------------------------- #
    weights_rows, iv_comp_parts, vol_parts, rho_rows, corr_mat_parts = [], [], [], [], []
    surface_parts, spot_parts = [], []

    # active window per rebalance; the cleaning step reuses this too
    active_by_reb: dict[pd.Timestamp, pd.DatetimeIndex] = {}
    for i, reb in enumerate(rebals):
        nxt = rebals[i + 1] if i + 1 < len(rebals) else calendar[-1] + pd.Timedelta(days=1)
        active_by_reb[reb] = calendar[(calendar >= reb) & (calendar < nxt)]

    for reb in rebals:
        active = active_by_reb[reb]
        if len(active) == 0:
            continue

        univ = get_universe(db, str(reb.date()), n=n)
        univ["permno"] = univ["permno"].astype(int)
        univ["secid"] = univ["secid"].astype("Int64")
        w = univ.set_index("permno")["weight"]
        permnos = univ["permno"].tolist()
        secids = univ["secid"].dropna().astype(int).tolist()
        sec2permno = dict(zip(univ["secid"].astype("Int64"), univ["permno"]))

        # weights (frozen) at this rebalance
        wr = univ[["permno", "secid", "weight", "market_cap", "rnk"]].copy()
        wr.insert(0, "rebalance_date", reb)
        weights_rows.append(wr)

        # component IV over the active window (point-in-time secids)
        ivc = get_iv(db, secids, str(active[0].date()), str(active[-1].date()), days=iv_days)
        if not ivc.empty:
            ivc["date"] = pd.to_datetime(ivc["date"])
            ivc["permno"] = ivc["secid"].astype("Int64").map(sec2permno).astype("Int64")
            ivc["rebalance_date"] = reb
            iv_comp_parts.append(ivc[["rebalance_date", "date", "permno", "secid", "iv_atm"]])

        # multi-pillar surface (marking-grade, stored raw) for this quarter's names
        surf = get_surface(db, secids, str(active[0].date()), str(active[-1].date()), days=surface_days)
        if not surf.empty:
            surf["date"] = pd.to_datetime(surf["date"])
            surf = surf[surf["date"].isin(active)]
            surf["permno"] = surf["secid"].astype("Int64").map(sec2permno).astype("Int64")
            surf.insert(0, "rebalance_date", reb)
            surface_parts.append(surf)

        # daily spots (same split convention as the strikes)
        sp = get_spots(db, secids, str(active[0].date()), str(active[-1].date()))
        if not sp.empty:
            sp["date"] = pd.to_datetime(sp["date"])
            sp = sp[sp["date"].isin(active)]
            sp["permno"] = sp["secid"].astype("Int64").map(sec2permno).astype("Int64")
            sp.insert(0, "rebalance_date", reb)
            spot_parts.append(sp)

        # returns with a trailing buffer for the rolling windows
        buf_start = (active[0] - pd.Timedelta(days=corr_window * 2 + 20)).date()
        ret = get_returns(db, permnos, str(buf_start), str(active[-1].date()))
        if ret.empty:
            continue
        ret.index = pd.to_datetime(ret.index)
        vols = realized_vol(ret)

        # realized vol (long) on active dates
        v21 = vols[21].reindex(active)
        v63 = vols[63].reindex(active)
        vlong = (
            v21.reset_index(names="date").melt("date", var_name="permno", value_name="vol_21")
            .merge(
                v63.reset_index(names="date").melt("date", var_name="permno", value_name="vol_63"),
                on=["date", "permno"],
            )
        )
        vol_parts.append(vlong)

        # daily rho-bar + full matrix at the rebalance date
        for d in active:
            C = realized_corr_matrix(ret, str(d.date()), corr_window, corr_min_obs, corr_method)
            if C is None:
                rho_rows.append((d, np.nan, np.nan))
                continue
            if d not in vols[63].index:  # skip rather than fall back to a stale vol row
                rho_rows.append((d, np.nan, np.nan))
                continue
            vols_d = vols[63].loc[d]
            rho_rows.append(
                (
                    d,
                    average_correlation(C, w, vols_d, rho_method),
                    average_correlation(C, w, vols_d, "equal"),
                )
            )
            if d == reb:  # store the full matrix only at rebalances
                m = C.stack().rename("corr").reset_index()
                m.columns = ["permno_i", "permno_j", "corr"]
                m.insert(0, "rebalance_date", reb)
                corr_mat_parts.append(m)

    # --- 2bis. Marking/RMT-grade extras: SPX surface, zero curve, returns panel  #
    surf_spx = get_surface(db, [SPX_SECID], date_start, date_end, days=surface_days)
    surf_spx["date"] = pd.to_datetime(surf_spx["date"])
    surf_spx = surf_spx[surf_spx["date"].isin(calendar) & (surf_spx["date"] >= rebals[0])].copy()
    reb_idx = pd.DatetimeIndex(rebals)
    surf_spx.insert(
        0, "rebalance_date",
        reb_idx[reb_idx.searchsorted(surf_spx["date"].values, side="right") - 1],
    )
    surf_spx["permno"] = pd.Series(pd.NA, index=surf_spx.index, dtype="Int64")

    surface = pd.concat(surface_parts + [surf_spx], ignore_index=True)
    surface["secid"] = surface["secid"].astype("Int64")
    surface = surface[
        ["rebalance_date", "date", "permno", "secid", "days", "cp_flag", "iv", "premium", "strike"]
    ].sort_values(["rebalance_date", "date", "secid", "days", "cp_flag"]).reset_index(drop=True)

    spx_spots = get_spots(db, [SPX_SECID], date_start, date_end)
    spx_spots["date"] = pd.to_datetime(spx_spots["date"])
    spx_spots = spx_spots[spx_spots["date"].isin(calendar) & (spx_spots["date"] >= rebals[0])].copy()
    spx_spots.insert(
        0, "rebalance_date",
        reb_idx[reb_idx.searchsorted(spx_spots["date"].values, side="right") - 1],
    )
    spx_spots["permno"] = pd.Series(pd.NA, index=spx_spots.index, dtype="Int64")

    spots = pd.concat(spot_parts + [spx_spots], ignore_index=True)
    spots["secid"] = spots["secid"].astype("Int64")
    spots = spots[
        ["rebalance_date", "date", "permno", "secid", "close", "cfadj"]
    ].sort_values(["rebalance_date", "date", "secid"]).reset_index(drop=True)

    rates = db.raw_sql(
        f"SELECT date, days, rate FROM optionm.zerocd WHERE date BETWEEN '{date_start}' AND '{date_end}'"
    )
    rates["date"] = pd.to_datetime(rates["date"])
    rates = rates[rates["date"].isin(calendar)].sort_values(["date", "days"]).reset_index(drop=True)

    # union of all constituents ever held; history from returns_start for 252d RMT windows
    all_permnos = sorted({int(p) for wr in weights_rows for p in wr["permno"]})
    ret_all = get_returns(db, all_permnos, returns_start, date_end)
    ret_all.index = pd.to_datetime(ret_all.index)
    returns_long = (
        ret_all.reset_index(names="date")
        .melt("date", var_name="permno", value_name="ret")
        .dropna(subset=["ret"])
        .sort_values(["date", "permno"])
        .reset_index(drop=True)
    )
    returns_long["permno"] = returns_long["permno"].astype(int)

    # --- 3. Cleaning of component IV (global percentile outliers, then ffill) -- #
    iv_components = pd.concat(iv_comp_parts, ignore_index=True)
    lo, hi = iv_components["iv_atm"].quantile(list(iv_pct))
    iv_components.loc[(iv_components["iv_atm"] < lo) | (iv_components["iv_atm"] > hi), "iv_atm"] = np.nan
    # ffill per (quarter, permno) over the full active window, not just up to the
    # name's last observation — so trailing gaps stay NaN, extended at most ffill_limit days
    cleaned = []
    for (reb, _), g in iv_components.groupby(["rebalance_date", "permno"]):
        active = active_by_reb[reb]
        s = g.set_index("date")["iv_atm"].reindex(active).ffill(limit=ffill_limit)
        cleaned.append(
            pd.DataFrame(
                {"date": s.index, "permno": g["permno"].iloc[0], "secid": g["secid"].iloc[0],
                 "iv_atm": s.values, "rebalance_date": reb}
            )
        )
    iv_components = pd.concat(cleaned, ignore_index=True)[
        ["rebalance_date", "date", "permno", "secid", "iv_atm"]
    ]

    # --- 4. Collect ----------------------------------------------------------- #
    out = {
        "iv_index": iv_index,
        "iv_components": iv_components,
        "weights": pd.concat(weights_rows, ignore_index=True),
        "realized_vol": pd.concat(vol_parts, ignore_index=True),
        "realized_corr": pd.DataFrame(rho_rows, columns=["date", "rho_bar", "rho_bar_equal"]),
        "corr_matrices": pd.concat(corr_mat_parts, ignore_index=True),
        "surface": surface,
        "spots": spots,
        "rates": rates,
        "returns": returns_long,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, df in out.items():
            df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)

    return out
