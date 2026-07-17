"""
Unit tests for dispersion.rmt.cleaning (pipeline MP + Laloux, README §8bis).
"""
import numpy as np
import pandas as pd
import pytest

from dispersion.rmt.cleaning import (corr_window, devolatilise, laloux_clip,
                                     spectral_features)


def _corr_df(A):
    n = A.shape[0]
    idx = list(range(n))
    return pd.DataFrame(A, index=idx, columns=idx)


# --------------------------------------------------------------------------- #
# The decisive test: pure noise in -> nothing found, near-identity out
# --------------------------------------------------------------------------- #
def test_iid_noise_cleans_to_near_identity():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((252, 100))
    C = _corr_df(np.corrcoef(X, rowvar=False))
    C2, d = laloux_clip(C, t_win=252)

    off_raw = np.abs(C.values[np.triu_indices(100, 1)]).mean()
    off_clean = np.abs(C2.values[np.triu_indices(100, 1)]).mean()
    assert d["k_signal"] == 0                       # no invented signal
    assert off_clean < 0.05 * off_raw               # bulk collapsed to ~identity
    assert d["trace_prenorm"] == pytest.approx(100.0, rel=1e-10)


def test_diag_psd_trace_on_correlated_data():
    rng = np.random.default_rng(11)
    # one-factor returns: beta * market + idio
    mkt = rng.standard_normal(252)
    X = 0.6 * mkt[:, None] + rng.standard_normal((252, 80))
    C = _corr_df(np.corrcoef(X, rowvar=False))
    C2, d = laloux_clip(C, t_win=252)

    assert np.allclose(np.diag(C2.values), 1.0, atol=1e-12)          # diag = 1
    assert np.linalg.eigvalsh(C2.values).min() > -1e-9               # PSD
    assert d["trace_prenorm"] == pytest.approx(80.0, rel=1e-10)      # trace kept
    assert d["k_signal"] >= 1                                        # market mode found
    assert d["edge_laloux"] < d["edge_naive"]                        # Laloux lowers the edge


def test_equicorrelation_structure_survives_cleaning():
    # exact one-factor world: C = (1-rho) I + rho J -> cleaning must NOT distort it
    n, rho = 100, 0.3
    C = _corr_df((1 - rho) * np.eye(n) + rho * np.ones((n, n)))
    C2, d = laloux_clip(C, t_win=252)
    assert np.abs(C2.values - C.values).max() < 1e-10
    assert d["k_signal"] == 1                                        # the market mode only


def test_devolatilise_scale_invariance():
    # invariance holds in LOG-return space (log1p is nonlinear in simple returns):
    # r2 such that log1p(r2) = k * log1p(r1) must give identical z-correlations
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    r1 = pd.DataFrame(rng.standard_normal((300, 4)) * 0.01, index=dates)
    r2 = np.expm1(np.log1p(r1) * [1.0, 5.0, 0.2, 3.0])   # per-name log-scale rescaling
    c1 = devolatilise(r1).tail(200).dropna().corr().values
    c2 = devolatilise(r2).tail(200).dropna().corr().values
    assert np.abs(c1 - c2).max() < 1e-10                 # correlations scale-free


def test_corr_window_thin_panel_returns_none():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    z = pd.DataFrame(rng.standard_normal((300, 50)), index=dates)
    assert corr_window(z, dates[-1], list(range(50)), min_names=60) is None


def test_corr_window_returns_effective_t():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2019-01-01", periods=400, freq="B")
    z = pd.DataFrame(rng.standard_normal((400, 70)), index=dates)
    z.iloc[-30:-20, 0] = np.nan                       # 10 jours troués sur un nom
    C, t_eff = corr_window(z, dates[-1], list(range(70)))
    assert t_eff == 242 and C.shape == (70, 70)       # complete-case : 252 - 10


def test_effective_t_kills_spurious_signal_on_noise():
    # bruit iid avec T_eff = 201 : le bord doit utiliser q_mp = N/201, pas N/252
    rng = np.random.default_rng(21)
    X = rng.standard_normal((201, 100))
    C = _corr_df(np.corrcoef(X, rowvar=False))
    _, d_wrong = laloux_clip(C, t_win=252)            # T nominal (l'ancien bug)
    _, d_right = laloux_clip(C, t_win=201)            # T effectif
    assert d_right["k_signal"] == 0                   # rien dans du bruit
    assert d_wrong["k_signal"] > 0                    # le bug aurait inventé du signal


def test_rotation_metric():
    from dispersion.rmt.daily import _rotation
    idx = list(range(20))
    v = pd.Series(np.ones(20) / np.sqrt(20), index=idx)
    assert _rotation(v, v) == pytest.approx(0.0, abs=1e-12)          # identique -> 0
    assert _rotation(v, -v) == pytest.approx(0.0, abs=1e-12)         # signe ignoré
    w = pd.Series(np.zeros(20), index=idx); w.iloc[0] = 1.0
    v2 = pd.Series(np.zeros(20), index=idx); v2.iloc[1] = 1.0
    assert _rotation(v2, w) == pytest.approx(1.0, abs=1e-12)         # orthogonal -> 1
    assert np.isnan(_rotation(v, None))                              # pas de veille
    u = pd.Series(np.ones(5), index=list(range(100, 105)))
    assert np.isnan(_rotation(v, u))                                 # recouvrement < 10 noms


def test_spectral_features_consistency():
    rng = np.random.default_rng(13)
    mkt = rng.standard_normal(252)
    X = 0.7 * mkt[:, None] + rng.standard_normal((252, 90))
    C = _corr_df(np.corrcoef(X, rowvar=False))
    feats, v1 = spectral_features(C, t_win=252, top=5)
    _, d = laloux_clip(C, t_win=252)
    assert feats["k_signal"] == d["k_signal"]
    assert 0 < feats["lam1_share"] < 1
    assert feats["absorption_top"] > feats["lam1_share"]             # top-5 ⊇ top-1
    assert v1.sum() > 0                                              # sign convention
    assert np.isclose((v1**2).sum(), 1.0)                            # unit norm