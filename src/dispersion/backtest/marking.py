"""
Mark aged option positions day by day: interpolate ATM sigma across pillars,
look up rates, and rebase strikes for splits.

    from dispersion.backtest.marking import interp_sigma, RateCurve, adjust_strike
"""
import numpy as np
import pandas as pd

PILLARS = (30, 60, 91)


def interp_sigma(sig30, sig60, sig91, tau_days):
    """
    ATM sigma at residual maturity `tau_days`, interpolated linearly in total
    variance w = sigma^2 * tau. Below 30 days we hold sigma(30) (short-end slope
    dropped, level still tracked). Scalars or aligned arrays; NaN pillars propagate.
    """
    sig30 = np.asarray(sig30, dtype=float)
    sig60 = np.asarray(sig60, dtype=float)
    sig91 = np.asarray(sig91, dtype=float)
    tau = np.asarray(tau_days, dtype=float)

    w30, w60, w91 = sig30**2 * 30.0, sig60**2 * 60.0, sig91**2 * 91.0

    w_lo = w30 + (w60 - w30) * (tau - 30.0) / 30.0        # [30, 60]
    w_hi = w60 + (w91 - w60) * (tau - 60.0) / 31.0        # [60, 91]

    with np.errstate(invalid="ignore", divide="ignore"):
        sig_lo = np.sqrt(w_lo / tau)
        sig_hi = np.sqrt(w_hi / tau)

    out = np.where(
        tau <= 30.0, sig30,
        np.where(tau <= 60.0, sig_lo, np.where(tau <= 91.0, sig_hi, sig91)),
    )
    return out if out.ndim else float(out)


class RateCurve:
    """
    Continuously-compounded zero rate r(date, tau) from the zerocd panel.
    Linear across the maturity grid, clamped at the ends. If `date` is missing
    from zerocd, fall back to the most recent curve within `max_stale_days`
    calendar days; beyond that, raise rather than use a stale rate. Parquet is in
    percent, returned as decimals.
    """

    def __init__(self, rates: pd.DataFrame, max_stale_days: int = 7):
        self._dates = pd.DatetimeIndex(np.sort(rates["date"].unique()))
        self._by_date = {
            d: (g["days"].to_numpy(dtype=float), g["rate"].to_numpy(dtype=float))
            for d, g in rates.sort_values("days").groupby("date")
        }
        self.max_stale_days = int(max_stale_days)

    def rate(self, date, tau_days: float) -> float:
        date = pd.Timestamp(date)
        if date in self._by_date:
            use = date
        else:
            pos = self._dates.searchsorted(date, side="right") - 1
            if pos < 0:
                raise KeyError(f"no zerocd curve on or before {date.date()}")
            use = self._dates[pos]
            if (date - use).days > self.max_stale_days:
                raise KeyError(f"zerocd gap too wide at {date.date()} (last curve {use.date()})")
        days, rate = self._by_date[use]
        return float(np.interp(float(tau_days), days, rate)) / 100.0


def adjust_strike(k_entry: float, cfadj_entry: float, cfadj_now: float) -> float:
    """
    Strike rebased to today's price scale after splits: K_t = K_entry *
    cfadj_entry / cfadj_t. Quantity scales by the inverse, so a split leaves the
    invested wealth unchanged.
    """
    return float(k_entry) * float(cfadj_entry) / float(cfadj_now)
