"""
Causality tests for dispersion.ml.regime — the HMM filtered posterior must use
ONLY past+present observations (the user's explicit look-ahead concern).
"""
import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from dispersion.ml.regime import hmm_filtered_posterior


def _fit_hmm(X, seed=0):
    m = GaussianHMM(3, covariance_type="full", random_state=seed, n_iter=50, tol=1e-3)
    m.fit(X)
    return m


def test_filtered_posterior_is_causal():
    # perturbing the FUTURE must leave every past filtered posterior bit-identical
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 4))
    m = _fit_hmm(X)
    post = hmm_filtered_posterior(m, X)

    t0 = 250
    X2 = X.copy()
    X2[t0 + 1:] += 5.0 * rng.standard_normal((len(X) - t0 - 1, 4))   # nuke the future
    post2 = hmm_filtered_posterior(m, X2)

    assert np.allclose(post[:t0 + 1], post2[:t0 + 1], atol=1e-12)     # past unchanged
    assert not np.allclose(post[t0 + 1:], post2[t0 + 1:])             # future did move


def test_filtered_rows_are_probabilities():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 4))
    post = hmm_filtered_posterior(_fit_hmm(X), X)
    assert np.allclose(post.sum(axis=1), 1.0, atol=1e-10)
    assert (post >= 0).all() and (post <= 1).all()


def test_filtered_equals_truncated_smoothed_at_last_step():
    # filtered posterior at time t == smoothed posterior of the sequence TRUNCATED
    # at t (no future obs => smoothing degenerates to filtering) — cross-check vs hmmlearn
    rng = np.random.default_rng(3)
    X = rng.standard_normal((120, 4))
    m = _fit_hmm(X)
    filt = hmm_filtered_posterior(m, X)
    for t in (30, 60, 119):
        smoothed_trunc = m.predict_proba(X[:t + 1])[-1]   # last row, no future
        assert np.allclose(filt[t], smoothed_trunc, atol=1e-8)
