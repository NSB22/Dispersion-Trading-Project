"""
Daily danger-regime probability from a GMM and a filtered HMM on the core stress
features, both refit walk-forward at each rebalance. The danger state is named
from the training window only, and the HMM uses the filtered (forward-pass)
posterior rather than the smoothed one, so nothing looks ahead. The feature is
the mean of the two probabilities. Causality is checked in tests/test_regime.py.

    from dispersion.ml.regime import build_regime_feature
    build_regime_feature()      # adds regime probs to ml_dataset.parquet
"""
import os
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

CORE = ["f_lam1", "f_dlam1_63", "f_turb21", "f_vix"]
# every core feature is oriented "higher = more stress", so we name the danger
# state by the highest composite centroid instead of any single axis (turbulence
# alone doesn't work — EWMA devol flattens it)
WARMUP = 756                      # ~3y of complete core rows before the first fit
N_STATES = 3


def hmm_filtered_posterior(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Filtered posterior P(state_t | obs_1..t) via a forward pass — at each t it
    only uses observations up to t, never later ones. Returns (T, K).
    """
    logB = model._compute_log_likelihood(X)          # (T, K) log emission probs
    log_pi = np.log(model.startprob_ + 1e-300)
    log_A = np.log(model.transmat_ + 1e-300)
    T, K = logB.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + logB[0]
    log_alpha[0] -= logsumexp(log_alpha[0])          # normalise (filtered)
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + logB[t]
        log_alpha[t] -= logsumexp(log_alpha[t])
    return np.exp(log_alpha)


def _danger_index(centroids: np.ndarray) -> int:
    """
    Danger = the component/state whose centroid has the highest mean across the
    core features. Centroids are already in training z-score units, so a plain
    mean is comparable across features.
    """
    return int(np.argmax(centroids.mean(axis=1)))


def regime_probs(feat: pd.DataFrame, rebals, core=CORE, warmup: int = WARMUP,
                 seed: int = 0):
    """
    Walk-forward danger probabilities for a core feature subset. `feat` is
    date-indexed with at least the `core` columns. Returns (p_gmm, p_hmm) aligned
    to feat.index — used for both the full-core feature and the VIX-only variant.
    """
    feat = feat[core]
    p_gmm = pd.Series(np.nan, index=feat.index)
    p_hmm = pd.Series(np.nan, index=feat.index)

    for k, reb in enumerate(rebals):
        reb = pd.Timestamp(reb)
        nxt = pd.Timestamp(rebals[k + 1]) if k + 1 < len(rebals) \
            else feat.index[-1] + pd.Timedelta(days=1)

        train = feat.loc[:reb].dropna()
        if len(train) < warmup:
            continue

        mu, sd = train.mean(), train.std().replace(0, 1.0)
        Ztr = ((train - mu) / sd).to_numpy(dtype="float64")

        gmm = GaussianMixture(N_STATES, covariance_type="full", random_state=seed,
                              reg_covar=1e-4).fit(Ztr)
        gd = _danger_index(gmm.means_)
        hmm = GaussianHMM(N_STATES, covariance_type="full", random_state=seed,
                          n_iter=100, tol=1e-3)
        hmm.fit(Ztr)
        hd = _danger_index(hmm.means_)

        seg = feat.loc[reb:nxt].iloc[:-1] if nxt in feat.index else feat.loc[reb:nxt]
        seg = seg.dropna()
        if seg.empty:
            continue
        Zseg = ((seg - mu) / sd).to_numpy(dtype="float64")
        p_gmm.loc[seg.index] = gmm.predict_proba(Zseg)[:, gd]

        hist = feat.loc[:reb].dropna()
        combo = pd.concat([hist, seg[~seg.index.isin(hist.index)]])
        Zc = ((combo - mu) / sd).to_numpy(dtype="float64")
        post = hmm_filtered_posterior(hmm, Zc)
        p_hmm.loc[seg.index] = pd.Series(post[:, hd], index=combo.index).loc[seg.index].to_numpy()

    return p_gmm, p_hmm


def build_regime_feature(
    processed_dir: str = "data/processed",
    core=CORE,
    warmup: int = WARMUP,
    out_file: str | None = "ml_dataset.parquet",
    seed: int = 0,
) -> pd.DataFrame:
    """Walk-forward causal danger-regime probabilities, merged into ml_dataset."""
    ml = pd.read_parquet(os.path.join(processed_dir, "ml_dataset.parquet"))
    weights = pd.read_parquet(os.path.join(processed_dir, "weights.parquet"))
    ml["date"] = pd.to_datetime(ml["date"])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])

    feat = ml[["date"] + core].set_index("date")
    rebals = sorted(weights["rebalance_date"].unique())
    p_gmm, p_hmm = regime_probs(feat, rebals, core=core, warmup=warmup, seed=seed)

    out = ml.copy()
    out["f_reg_gmm"] = p_gmm.to_numpy()
    out["f_reg_hmm"] = p_hmm.to_numpy()
    with warnings.catch_warnings():
        # warm-up rows are NaN in both models -> nanmean warns on those; ignore it
        warnings.simplefilter("ignore", RuntimeWarning)
        out["f_regime"] = np.nanmean(np.c_[p_gmm.to_numpy(), p_hmm.to_numpy()], axis=1)
    num = [c for c in out.columns if c != "date"]
    out[num] = out[num].astype("float64")

    if out_file:
        out.to_parquet(os.path.join(processed_dir, out_file), index=False)
    return out
