"""
RMT cleaning of realised correlation matrices (README §8bis; validated in
exploration_notebooks/07_explore_spectrum.ipynb).

Pipeline (frozen, Week 4): EWMA(λ=0.94) de-volatilisation → 252-day correlation
of standardised residuals (complete-case) → Marchenko–Pastur with the **Laloux
effective edge** λ₊ᵉᶠᶠ = (1 − λ₁/N)(1 + √q_mp)² → clip the bulk to its mean
(trace-preserving) → renormalise (diag = 1, PSD).

Notation: `q_mp` = N/T is the Bouchaud–Potters MP aspect ratio — NOT the
dividend yield `q` of the pricing modules (utils/greeks, backtest/engine).

Usage:
    from dispersion.rmt.cleaning import devolatilise, corr_window, laloux_clip
    z = devolatilise(returns_wide)
    C = corr_window(z, date, permnos)
    C_clean, diag = laloux_clip(C)
"""
import numpy as np
import pandas as pd

T_WIN = 252
MIN_OBS = 200
EWMA_LAMBDA = 0.94


def devolatilise(returns_wide: pd.DataFrame, lam: float = EWMA_LAMBDA,
                 min_periods: int = 20) -> pd.DataFrame:
    """
    Standardised residuals z_t = log-return_t / sigma_{t-1}, with sigma from an
    EWMA (RiskMetrics decay). The one-day shift keeps sigma PREDICTABLE — no
    look-ahead inside subsequent correlation windows.
    """
    logret = np.log1p(returns_wide.astype("float64"))
    sigma = logret.ewm(alpha=1.0 - lam, min_periods=min_periods).std().shift(1)
    return logret / sigma


def corr_window(z: pd.DataFrame, date, permnos, t_win: int = T_WIN,
                min_obs: int = MIN_OBS, min_names: int = 60):
    """
    Complete-case correlation of the last `t_win` rows of z up to `date`, for
    the given universe. Names with fewer than `min_obs` observations in the
    window are dropped first (listwise philosophy, §6.3). Returns None if the
    panel is too thin, else (C, t_eff) where **t_eff is the EFFECTIVE number of
    complete-case rows** — the MP edge must use q_mp = N/t_eff, not N/t_win
    (nominal-T under-cleans: iid noise at t_eff=201 passed as 252 shows spurious
    k_signal — adversarial audit, 17 Jul).
    """
    cols = [p for p in permnos if p in z.columns]
    win = z.loc[:pd.Timestamp(date), cols].tail(t_win)
    win = win.dropna(axis=1, thresh=min_obs).dropna(axis=0)
    if win.shape[1] < min_names or win.shape[0] < min_obs:
        return None
    return win.corr(), int(win.shape[0])


def laloux_clip(C: pd.DataFrame, t_win: int = T_WIN):
    """
    Clip the noise bulk of a correlation matrix below the Laloux effective edge.

    Returns (C_clean, diagnostics) with diagnostics carrying n, t, q_mp,
    lam1_share, edge_naive, edge_laloux, k_signal and trace_prenorm (the
    trace-preservation witness before the diagonal renormalisation).
    """
    is_df = isinstance(C, pd.DataFrame)
    A = C.to_numpy(dtype="float64") if is_df else np.asarray(C, dtype="float64")
    n = A.shape[0]
    ev, V = np.linalg.eigh(A)                      # ascending
    q_mp = n / t_win
    lam1_share = ev[-1] / n
    edge_naive = (1.0 + np.sqrt(q_mp)) ** 2
    edge = (1.0 - lam1_share) * edge_naive         # Laloux correction
    bulk = ev < edge
    ev2 = ev.copy()
    if bulk.any():
        ev2[bulk] = ev[bulk].mean()                # trace-preserving on the bulk
    A2 = (V * ev2) @ V.T
    d = np.sqrt(np.diag(A2))
    A2 = A2 / np.outer(d, d)
    np.fill_diagonal(A2, 1.0)

    out = pd.DataFrame(A2, index=C.index, columns=C.columns) if is_df else A2
    diagnostics = {
        "n": n, "t": t_win, "q_mp": q_mp, "lam1_share": float(lam1_share),
        "edge_naive": float(edge_naive), "edge_laloux": float(edge),
        "k_signal": int((~bulk).sum()), "trace_prenorm": float(ev2.sum()),
    }
    return out, diagnostics


def spectral_features(C: pd.DataFrame, t_win: int = T_WIN, top: int = 5):
    """
    Regime features of a correlation matrix (role B, README §8bis):
    lam1_share, k_signal (Laloux edge), absorption_top (share of variance in the
    `top` largest eigenvalues, Kritzman-style absorption ratio). Returns
    (features dict, dominant eigenvector as a pd.Series) — the caller computes
    the day-to-day eigenvector rotation 1 − |v1_t · v1_{t-1}|.
    """
    A = C.to_numpy(dtype="float64")
    n = A.shape[0]
    ev, V = np.linalg.eigh(A)
    q_mp = n / t_win
    edge = (1.0 - ev[-1] / n) * (1.0 + np.sqrt(q_mp)) ** 2
    feats = {
        "lam1_share": float(ev[-1] / n),
        "k_signal": int((ev >= edge).sum()),
        "absorption_top": float(ev[-top:].sum() / n),
    }
    v1 = pd.Series(V[:, -1], index=C.index)
    if v1.sum() < 0:                               # sign convention: market mode positive
        v1 = -v1
    return feats, v1
