"""
Daily mark-to-surface utilities for aged option positions (README §7.3).

Frozen rules:
- ATM pillar IVs at 30/60/91 days, interpolated LINEARLY IN TOTAL VARIANCE
  (sigma^2 * tau) between pillars;
- below 30 days of residual maturity: sigma := that day's sigma(30d), re-read daily
  (only the short-end slope is ignored, never the level moves);
- ATM proxy: the interpolated ATM sigma is applied to the position's FIXED entry
  strike via BS (documented approximation, §7.3) — pricing itself lives in
  dispersion.utils.greeks;
- rates: zerocd curve interpolated at tau, with a bounded date-ffill for the few
  calendar days missing from zerocd (§9bis);
- splits: strikes and quantities are rebased with secprd's cfadj
  (K_t = K_entry * cfadj_entry / cfadj_t; quantity scales by the inverse).

Usage:
    from dispersion.backtest.marking import interp_sigma, RateCurve, adjust_strike
"""
import numpy as np
import pandas as pd

PILLARS = (30, 60, 91)


def interp_sigma(sig30, sig60, sig91, tau_days):
    """
    ATM sigma at residual maturity `tau_days`, linear in total variance.

    Piecewise rule on w(tau) = sigma^2 * tau (tau in days — the 1/365 scaling
    cancels in the interpolation):
        tau <= 30           : sigma(30)                (frozen short end, §7.3)
        30 < tau <= 60      : w linear between pillars 30 and 60
        60 < tau <= 91      : w linear between pillars 60 and 91
        tau > 91            : sigma(91)                (never needed: tau <= 91)

    Inputs may be scalars or aligned numpy/pandas arrays; NaN pillars propagate.
    """
    sig30 = np.asarray(sig30, dtype=float)
    sig60 = np.asarray(sig60, dtype=float)
    sig91 = np.asarray(sig91, dtype=float)
    tau = np.asarray(tau_days, dtype=float)

    w30, w60, w91 = sig30**2 * 30.0, sig60**2 * 60.0, sig91**2 * 91.0

    # linear total variance on each segment
    w_lo = w30 + (w60 - w30) * (tau - 30.0) / 30.0        # segment [30, 60]
    w_hi = w60 + (w91 - w60) * (tau - 60.0) / 31.0        # segment [60, 91]

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

    - linear interpolation across the maturity grid (clamped at the ends);
    - bounded date-ffill: if `date` is absent from zerocd (10 of 7,281 calendar
      days, §9bis), the most recent curve within `max_stale_days` CALENDAR days is
      used; beyond that a KeyError is raised (loud failure, no silent staleness).
    Rates come in percent in the parquet and are returned as decimals.
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
    Strike rebased to today's price scale after splits (secprd cfadj convention):
        K_t = K_entry * cfadj_entry / cfadj_t.
    The position quantity scales by the inverse (cfadj_t / cfadj_entry) so the
    wealth invested is unchanged by the split.
    """
    return float(k_entry) * float(cfadj_entry) / float(cfadj_now)
