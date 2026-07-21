"""
Run the whole project end to end: pull WRDS, build the base parquets, fit the
signals and ML features, then backtest v0 / v1 / v1_rmt / v1_rmt+regime. This is
the reproducible path from raw WRDS to the strategy P&L; notebooks only read the
parquets it writes.

    uv run --env-file .env python main.py            # full run
    uv run --env-file .env python main.py --no-data  # skip the WRDS rebuild, reuse base parquets
"""
import argparse
import time

import numpy as np
import pandas as pd

from dispersion.backtest.engine import exante_quantile_threshold, run_backtest
from dispersion.data.assemble import build_dataset
from dispersion.data.wrds_client import get_connection
from dispersion.ml.experiments import _high_veto, _stats
from dispersion.ml.features import build_ml_dataset
from dispersion.ml.regime import build_regime_feature
from dispersion.rmt.daily import build_rmt_daily, build_signal_rmt
from dispersion.signal.implied_corr import build_signal

PROC = "data/processed"


def stage_data():
    db = get_connection()
    out = build_dataset(db)
    db.close()
    return {k: v.shape for k, v in out.items()}


def stage_models():
    build_signal()          # signal.parquet
    build_rmt_daily()       # rmt_daily.parquet: 252d cleaned rho-bar + spectral features
    build_signal_rmt()      # signal_rmt.parquet: the RMT-cleaned signal
    build_ml_dataset()      # ml_dataset.parquet: features + spike label
    build_regime_feature()  # adds f_reg_gmm / f_reg_hmm / f_regime to ml_dataset


def stage_backtests():
    signal = pd.read_parquet(f"{PROC}/signal.parquet")
    signal_rmt = pd.read_parquet(f"{PROC}/signal_rmt.parquet")
    ml = pd.read_parquet(f"{PROC}/ml_dataset.parquet")
    weights = pd.read_parquet(f"{PROC}/weights.parquet")
    for d in (signal, signal_rmt, ml):
        d["date"] = pd.to_datetime(d["date"])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    rebals = sorted(weights["rebalance_date"].unique())
    base = {k: pd.read_parquet(f"{PROC}/{k}.parquet") for k in ("surface", "spots", "rates", "weights")}

    def bt(sig, thr, net, tag):
        r = run_backtest(data={**{k: v.copy() for k, v in base.items()}, "signal": sig.copy()},
                         threshold=thr, costs=({} if net else None))
        suffix = "net" if net else "gross"
        r["quarterly"].to_parquet(f"{PROC}/mainrun_{tag}_{suffix}_quarterly.parquet", index=False)
        return _stats(r["quarterly"], r["daily"])

    # gates: v1 on the base signal, v1_rmt on the RMT-cleaned signal, both ex-ante median
    thr_v1 = exante_quantile_threshold(signal, rebals, q=0.5, warmup=12)
    thr_v1rmt = exante_quantile_threshold(signal_rmt, rebals, q=0.5, warmup=12)
    reg_at = {pd.Timestamp(r): float(ml.loc[ml["date"] == pd.Timestamp(r), "f_regime"].iloc[0])
              for r in rebals if (ml["date"] == pd.Timestamp(r)).any()
              and np.isfinite(ml.loc[ml["date"] == pd.Timestamp(r), "f_regime"].iloc[0])}
    thr_reg = thr_v1rmt.copy()
    for d in _high_veto(reg_at, q=0.67):
        thr_reg[pd.Timestamp(d)] = 1e9

    res = {}
    res["v0_gross"] = bt(signal, None, False, "v0")
    res["v0_net"] = bt(signal, None, True, "v0")
    res["v1_net"] = bt(signal, thr_v1, True, "v1")
    res["v1rmt_gross"] = bt(signal_rmt, thr_v1rmt, False, "v1rmt")
    res["v1rmt_net"] = bt(signal_rmt, thr_v1rmt, True, "v1rmt")
    res["v1rmt_regime_gross"] = bt(signal_rmt, thr_reg, False, "v1rmt_regime")
    res["v1rmt_regime_net"] = bt(signal_rmt, thr_reg, True, "v1rmt_regime")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-data", action="store_true", help="skip the WRDS rebuild")
    args = ap.parse_args()

    t0 = time.time()
    if not args.no_data:
        print("[1/3] data — building base parquets from WRDS ...", flush=True)
        shapes = stage_data()
        print("      " + " | ".join(f"{k} {v}" for k, v in shapes.items()), flush=True)
    else:
        print("[1/3] data — skipped (reusing base parquets)", flush=True)

    print("[2/3] models — signal, RMT, ML features, regime ...", flush=True)
    stage_models()

    print("[3/3] backtests — v0 / v1 / v1_rmt / v1_rmt+regime ...", flush=True)
    res = stage_backtests()
    print(f"\n{'strategy':22s} {'Sharpe':>7s} {'skew':>6s} {'maxDD':>7s} {'cumul':>7s}", flush=True)
    for k, s in res.items():
        print(f"{k:22s} {s['sharpe']:+7.2f} {s['skew']:+6.2f} {s['maxDD']:>7.1%} {s['cumul']:>6.1f}x", flush=True)
    print(f"\ndone in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
