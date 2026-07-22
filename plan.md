# Work plan — ML-Driven Dispersion Trading Strategy

> **Timeline:** start 25 June 2026, target finish 31 July 2026 → ~5 weeks. Intensive pace. The ML is a bonus layer, not a prerequisite.

---

## WEEK 1 (29 Jun → 5 Jul) — Setup, Data & Universe

**Goal:** infrastructure ready + a clean point-in-time dataset. This is the most critical week.

### Setup
- [✅] WRDS connection operational (`wrds.Connection()`) — account validated, `.pgpass` created, `crsp`/`optionm`/`comp` access confirmed
- [✅] `optionm.vsurfd` explored (notebook 01): annual tables 1996→2025, columns known, 91d/delta ±50 native, SPX = secid 108105
- [✅] CRSP explored (notebook 02): membership `crsp.dsp500list` (1925→2024), realised side `crsp.dsf` (prc/ret/shrout), `secid↔permno` join = `wrdsapps_link_crsp_optionm.opcrsphist`. Modern product `idx_const_*_v2` = access denied → legacy approach retained.
- [✅] GitHub repo + environment (uv + pyproject.toml: wrds, pandas, numpy, scipy, scikit-learn, statsmodels)
- [✅] WRDS connection plumbing written (`src/dispersion/data/wrds_client.py` + `.env`)
- [✅] **Design parameters frozen**: N=100 (top market cap, turnover ~4/quarter, 63–70% coverage), 91d maturity, delta ±50, quarterly rebalancing. SPX = secid 108105.
- [✅] **Backtest period frozen: 1996 → 2024** (~29 years, ~116 rebalances). Coverage diagnostic: 97–100/100 constituents with IV + SPX OK every year from 1996; upper bound = end of `dsf` (31/12/2024).

### Data
- [✅] CRSP: `get_universe(db, date, n=100)` → `src/dispersion/data/universe.py` — tested on 2010/2015/2020, 0 missing secids
- [✅] OptionMetrics `vsurfd`: IV for SPX + constituents, 91d, delta ±50 → `get_iv()` (`src/dispersion/data/iv.py`), tested March 2020: 0 gaps, diversification sanity OK
- [✅] Identifier joins: `secid` ↔ `permno` via `opcrsphist` score=1, integrated into `get_universe`
- [✅] CRSP `dsf` → returns + realised vol + realised correlation: `get_returns`, `realized_vol`, `realized_corr_matrix` (listwise), `average_correlation` (weighted ρ̄, option B) — `src/dispersion/data/returns.py`, tested COVID vs calm OK. Singular matrix (T=63<N=100) → confirms the case for RMT.
- [✅] Cleaning: master calendar = intersection vsurfd(SPX)∩dsf; IV ffill capped at 3d; constituent outliers outside the 0.1/99.9 percentiles → NaN; SPX not clipped. (`src/dispersion/data/assemble.py`)
- [✅] Assembly `build_dataset(db, 1996, 2024)` → 6 `.parquet` files in `data/processed/` (full build in 7 min).
- [✅] Multi-agent adversarial QA (6 dimensions): 4 PASS / 2 WARN. Only 1 genuine issue = OptionMetrics gap 27/07→18/08/2020 (vendor outage, zero IV across the whole surface) → accepted + documented (methodology §9bis). **WEEK 1 DONE.**

**Deliverable:** ✅ `.parquet` files produced (`iv_index`, `iv_components`, `weights`, `realized_vol`, `realized_corr`, `corr_matrices`) aligned on the master calendar. Reproducible via `build_dataset`.

**Pitfalls:** survivorship/look-ahead bias → point-in-time mandatory · the identifier join is the real bottleneck, attack it from day 1.

---

## WEEK 2 (6 Jul → 12 Jul) — Theory & Signal

**Goal:** the signal mechanics + their theoretical justification.

### Mathematical derivation — ✅ acquired via a full reading of DMV (2009), 8 Jul.
- [✅] Dispersion-portfolio P&L via Itô's lemma — eqs. (8)–(11) re-derived: triangular hedge (vega first, vol-of-vol cancels in the relative-vega ratio; residual delta with the index alone). Write-up for the report in W5.
- [✅] Decomposition into a realised-vs-implied correlation bet — eqs. (2)–(3), identification via VRP_indiv ≈ 0 (98/127 names)
- [✅] VRP + CRP formalisation — eq. (3) via Itô/Girsanov (convexities cancel under ℚ−ℙ); MFIV eqs. (4)–(5) Carr–Madan understood

### Methodological forks decided (8 Jul, methodology §5 + §8)
- [✅] **Implied variance = ATM 91d** (instrument consistency: we trade ATM straddles; conservative bias: the index skew makes ρ_implied understated); MFIV = W5 robustness if time
- [✅] **Window-matching = two series**: trailing (ex-ante signal, W3) + forward = trailing shifted by +63 business days (DMV-style premium validation); intra-quarter weights caveat documented
- [✅] **Implementation micro-choices (8 Jul, data-verified)**: missing IV → daily weight renormalisation on available names + 90/100 floor ("drop the days" would have destroyed 54.9% of the sample; median missing weight 1.3%); ρ_implied stored **raw** + violation counters; deliverable `signal.parquet`; `secid` → Int64 (methodology §5.1)

### Implied correlation (implementation)
- [✅] Inversion formula implemented + validated (notebook 03, 8 Jul) → promoted to `src/dispersion/signal/implied_corr.py`
- [✅] ρ_realized from the realised covariance (done in W1: daily weighted ρ̄, option B — same functional form as the inversion ⟹ apples-to-apples spread)
- [✅] **Premium** and **signal** series computed: Π̄ = **+0.079 (t-NW63 = 7.4, 75% of days > 0)**, S̄ = +0.078 (t = 10.6, 80% of days > 0) → **CRP thesis validated on 1996–2024**
- [✅] Bound checks: ρ_implied ∈ [0.12, 0.91] — **0 violations, 0 NaN** over 7,221 days; min coverage 91/100 (floor never hit)
- [✅] Subperiod preview: 2008–12 = 0.135 (t=4.3); **2020–24 = 0.028 (t=1.19, not significant)** → recent premium compression, to dig into in W5 (limits to arbitrage)
- [✅] Promotion: `src/dispersion/signal/implied_corr.py` (`implied_correlation` + `build_signal`) + `signal.parquet` written (7,221 d × 7 cols) — reproduces the notebook exactly.
- [✅] **Full end-of-W2 audit** (4 independent axes: data, signal, mathematics, consistency): 0 critical findings; ρ_implied recomputed independently (gap ~1e-16); documentation corrections applied; latent fragilities → methodology §9bis. **WEEK 2 DONE** (derivation write-up → W5).

**Deliverable:** ✅ `signal.parquet` + validation chart `results/figures/fig_crp_validation.png` (implied above forward realised, window-matched premium).

**Pitfall:** one-factor approximation (a single average correlation) — own it and document it (DMV eq. 6 makes the same assumption).

---

## WEEK 3 (13 Jul → 19 Jul) — Baseline Strategy

**Goal:** a running backtest, no ML = the reference benchmark.

### Engine
- [✅] `backtest/marking.py` (8 Jul): total-variance interpolation 30/60/91 + frozen σ(30d), `RateCurve` (capped ffill, loud failure), `adjust_strike` (cfadj splits) — 7 pytest tests (17/17 overall)
- [✅] `DispersionEngine` class (8 Jul, `backtest/engine.py`): book short index straddle / long constituent straddles, entry at real premiums + q̂ via C−P parity, wealth sizing λ (neutrality to PROPORTIONAL vol shocks — the only convention that reproduces DMV's +101.12%, since ν≈1/σ for an ATM straddle), daily marking, daily index delta hedge (eqs. 10–11), intrinsic settlement, cfadj splits, v1 wired (`threshold`) — **5 closed-form tests** (22/22 overall)
- [✅] **Unconditional v0 EXECUTED (8 Jul, 222 s, 115 quarters) — DMV BENCHMARKS MET**: **Σyᵢ = 99.5%** (theirs +101.12%), **gross Sharpe 0.77** (theirs 0.73), 70% quarters > 0, skew −1.29; mean quarterly ret +7.3% gross; worst quarters = **Q1-2018 Volmageddon (−91%), Q3-2024 yen carry (−41%), summer 2002 (−30%)**; gross maxDD −95.8% → the central argument for v1/ML (cut the tails). Counters: frozen marks 0.21 d/pos/quarter; 621 settlements at the D-1 spot = universe rotation (~5/quarter, fixed at the next rebuild). Outputs: `backtest_v0_{daily,quarterly}.parquet`
- [✅] **Q1-2018 mechanism CONFIRMED via the ledger** (not an artefact): index leg +20% of W, constituents +2%, **delta-hedge −154% of W** = gamma bleed on real SPX closes (independent of the marking model) — the "index realised variance explodes when correlation spikes" channel (DMV eq. 3). Intra-quarter NAV fell to 6% of W → unconditional wealth sizing = near-ruin (consistent with DMV Table VI margins)
- [✅] **Relative** greeks (8 Jul): `src/dispersion/utils/greeks.py` (BS + Black-76 prices/greeks, contract↔relative conversions, forward via parity, implied vol, straddles) + **10 pytest tests** (`tests/test_greeks.py`): closed forms, parity, finite differences, anti-DMV-trap test (wealth vega-neutrality), reproduction of real SPX premiums of 28/06/2024 to ±0.5%
- [✅] **Fork decided (notebook 04, 8 Jul): premiums = `impl_premium`** (vsurfd re-extraction — naive BS biased by ±5%, SPX implied q 1.04% identical call/put, 100% coverage from 1996); **greeks = analytic BS/Black-76** from IV + forward (`fwdprd`/put-call parity) + `zerocd` (methodology §8bis)
- [✅] **Fork decided (8 Jul): DAILY marking by surface interpolation** (pillars < 91d, linear in total variance at fixed delta) — effective daily delta hedge, decomposable P&L (vega/gamma/theta), daily series for W4 ML; DMV holding-period available by aggregation (benchmark); constant-IV marking ruled out (a vol strategy's P&L IS vega×ΔIV)
- [✅] **Bundled dataset rebuild (8 Jul, 2 passes, ~10 min)**: `surface.parquet` (4.9M rows, pillars 10/30/60/91d, 0 NaN, 0 duplicates) + `spots.parquet` (724k secprd closes + cfadj, 169/329 names with splits, SPX verified) + `rates.parquet` (zerocd) + `returns.parquet` (329 permnos from 1995) + §9bis hardening + truncated-tail fix. Non-regression: 5 parquets bit-identical; signal moved on 139 days (≤ 8e-3, documented policy finally applied); headline unchanged (+0.079, t=7.4). QA findings: 10d pillar ~37% populated (sub-30d fallback rule to set at engine design); zerocd missing 10 d/7,281 (ffill at consumption).

- [✅] **Engine forks decided (8 Jul, methodology §7.3)**: sizing = **DMV wealth** (Table II comparability — to justify in the thesis); delta hedge = **index only, daily** (one-factor consistency + costs ~0.5–2 bps vs 100 stock lines — to justify in the thesis); expiry = **intrinsic settlement** at the next rebalance (±3 d documented); sub-30d marking = **that day's σ(30d)** (uniform rule); **ATM-proxy** approximation documented (real premiums + payoff ⇒ second-order on the cumulative)

### Rules & realism
- [✅] **v1 EXECUTED (8 Jul)** — threshold = **rolling ex-ante quantile** (option A: median of the signal over available history, 12-rebalance warm-up, zero look-ahead; `exante_quantile_threshold`): 58/115 quarters traded. Results: premium per TRADED quarter improved (+7.9% vs +7.3% v0) → the signal has information about the premium; **dodges Q3-2024 (+1.4% vs −41%), summer 2002 and the Lehman quarters**; switches the strategy off post-2020 (17/20 last quarters skipped = notebook 03's compression at work); **BUT trades Q1-2018** (signal above its median at end-2017: the spread measures premium richness, NOT danger) → maxDD unchanged, overall Sharpe 0.56 < 0.77 (half the time in cash). **Conclusion: the threshold times the premium, not the risk — the exact quantitative argument for W4 ML (target = spike regimes, vol-level/spectral features).** Sensitivity q ∈ {0.25; 0.75} → W5 robustness. Exit/roll: intrinsic settlement (§7.3)
- [✅] **Costs fork decided (8 Jul, notebook 05, methodology §8bis)**: **parametric grid calibrated on real opprcd quotes** (3 groups SPX / rnk 1-50 / rnk 51-100 × linear interpolation 1996→2024: 1→0.6% / 7→1.2% / 10→3%); **½-spread × premium per leg at entry** (spread paid once, DMV held-to-maturity convention) + **1 bp** on the notional of each hedge trade; settlement spread-free. Raw opprcd rejected: SPX 2012 EOD artefact (quoted ≠ effective, ~30-50% of the quote) + COVID 2020 cell. Conservative by construction. Sensitivity ±50% + "stress ×2" → W5
- [✅] **NET RESULTS EXECUTED (8 Jul) — DMV's Table V replicates on our 29 years**: v0 net = +4.0%/quarter (gross +7.3%, ÷1.8), **Sharpe 0.77 → 0.42 (DMV: 0.73 → 0.41)**, cumulative ×165 → ×2.5 (~3%/yr net, marginal); cost ~3.3% per traded quarter; **v1 dies net (×1.1, Sharpe 0.33)** — the gate selects the wide-spread eras (cost 3.0% vs 2.7%). Full limits-to-arbitrage narrative + a quantified bar for W4 ML. Outputs: `backtest_{v0,v1}_net_*.parquet`

### Metrics
- [✅] Sharpe, Sortino, MaxDD, cumulative P&L, t-stats (notebook 06 → `results/tables/table_s3_metrics.csv`): v0 gross t=4.11; **v0 net t=2.24 (still significant)**; v1 net t=1.79 (marginal)
- [✅] **P&L decomposition by source** (ledger, additive): all quarters = index leg +8.3% / constituents −2.4% / hedge +1.2%; **crises (12/115) = hedge −23.0%** (the realised-variance/gamma channel) + index −14.3%, settlement +8.2% — maps onto (theta+vega = legs, gamma/realised correlation = hedge); fine greek granularity = W5 refinement if useful
- [✅] **DMV sanity** (`table_s3_dmv_sanity.csv`): Σy +99.5% vs +101.12% ✓; Sharpe 0.77/0.42 vs 0.73/0.41 ✓; negative skew ✓; entry hedge near delta-neutral (+0.3%, |.| 5%) — DMV's −32.54% (1-month, static) not directly comparable, documented
- [✅] Stats: plain t on non-overlapping quarters; NW reserved for daily series (W5)

**Deliverable:** ✅ `results/figures/fig_backtest_s3.png` (log NAV, 4 variants, crises annotated) + `results/tables/table_s3_{metrics,dmv_sanity}.csv` + notebook 06. **WEEK 3 DONE** (8 Jul — 11 days ahead of schedule).

**Pitfall:** Delta-neutral is not enough → watch the vega convexity (Volga).

---

## WEEK 4 (20 Jul → 26 Jul) — Random Matrix Theory (+ light ML if time allows)

**Goal:** the quantitative value-added layer.

### RMT (priority) — spec frozen 8 Jul (methodology §8bis), module `src/dispersion/rmt/`
- [✅] `returns.parquet`: done in W3 (bundled rebuild brought forward — 329 permnos, 1995 buffer)
- [✅] **Step-0 forks decided (17 Jul)**: notation `q_mp` = N/T (code + methodology note; `q` keeps meaning the dividend yield in pricing); devol **EWMA λ=0.94** (RiskMetrics, W5 sensitivity); cleaned matrices **daily**; re-injection = **both paths** (full signal_rmt variant re-backtested + spectral features for the ML); Laloux = integral part of the pipeline (it sets the clipping edge), not a separate estimator
- [✅] **Step 1 (17 Jul, notebook 07)**: 252d de-volatilised matrices (EWMA 0.94) at the 116 rebalances — median q_mp 0.397, N_eff ≈ 100; **median λ₁/N 33.5%** [17% end-2017 → 54% mid-2012]; spectrum vs MP: the bulk fits the density **with σ² = 1−λ₁/N** (Laloux); **median K = 7 at the Laloux edge vs 4 at the naive edge** (the naive edge, ~35% too high, MISSES ~3 sector factors per quarter — a quantified demonstration of the correction); **iid test: K = 0**, exact edges, trace 1.0000 (the pipeline invents nothing); clipping preview: ρ̄_clean vs ρ̄_raw corr 0.998, median |Δ| 0.011 (max 0.024) → **the scalar barely moves, as anticipated** (λ₁ preserved) — the RMT effort will pay through features + the parsimonious leg. Figure: `fig_mp_spectrum.png`
- [✅] **Step 2 (17 Jul) — module `src/dispersion/rmt/cleaning.py`**: `devolatilise` (EWMA 0.94, predictable σ shift-1), `corr_window` (252d complete-case, min 200 obs/60 names), `laloux_clip` (effective edge → trace-preserving clipping → diag=1 renorm) with diagnostics (q_mp, λ₁/N, K, edges, trace witness), `spectral_features` (λ₁/N, K, top-5 absorption, signed v1 for the rotation)
- [✅] **Tests (6, suite 31/31)**: iid → K=0 + near-identity (off-diag ÷20+); diag/PSD/trace on correlated data; **exact equicorrelation → the cleaning does NOT distort true structure** (diff < 1e-10); log-scale invariance; too-thin panel → None; features/clip consistency
- [✅] **Step 3 — Role A EXECUTED (17 Jul)**: `rmt/daily.py` (`build_rmt_daily` 61 s → `rmt_daily.parquet` 7,221 d × [ρ̄ raw/clean 252d, λ₁/N, K, absorption, Δλ₁, rotation]; `build_signal_rmt` → `signal_rmt.parquet`); corr(signal_base, signal_rmt) = 0.40. **v1_rmt (ex-ante median gate on signal_rmt): gross Sharpe 0.80 (v1: 0.56), skew +1.28 (POSITIVE vs −1.45), maxDD −51.7% (vs −95.9%), dodges ALL THREE crashes including Volmageddon; net: Sharpe 0.56 > the 0.42 bar, skew +0.80, cumulative ×12.4.** 39/115 gate disagreements vs v1.
- [✅] **ATTRIBUTION ABLATION (the "don't oversell RMT" guardrail)**: the same gate on the RAW 252d trailing does 0.78 / skew +1.31 / same crashes dodged (2 disagreements/115 vs cleaned) → **the driver is the 252d WINDOW (a slow regime anchor); the clipping adds only ~+0.02 Sharpe on this channel**. RMT's value plays out in the features (role B/ML) and the parsimonious leg — to be written up as such in the thesis. Integrity: 0 forced skips from NaN at rebalances (85 NaN days = mid-quarter churn 2001/2009/2016).
- [✅] **Adversarial look-ahead audit of the RMT path (independent agent, 17 Jul): NO leakage** — predictable EWMA (bit-level invariance to future perturbations), windows bounded at t (same convention as the baseline), exact shift(−63), reconstructed gate = 115/115 of the run's decisions, inclusive ex-ante quantile = 0/115 decisions changed. **2 defects found and FIXED**: ① signal_rmt's `premium` column **invalidated** (252d trailing vs its shift-63 = 75% shared observations, signal/premium corr 0.876 — mechanical contamination, NEVER use it for validation; the thesis premium remains the base 63d window-matched one); ② **MP edge at nominal T instead of effective T** (complete-case T_eff ∈ [200,252] on ~⅓ of days → k_signal inflated by ~1; iid noise at T_eff=201 passed off as 252 showed K=2) → fix: `corr_window` returns T_eff, edge computed on it + regression test; + silent vol fallback replaced by loud failure. Suite: 34/34. Recorded: estimator mix (de-volatilised corr × raw vols — an owned variant, to write up), rotation/Δλ₁ features gapped on the 85 churn days (to document for the ML).
- [✅] **Parsimonious leg (21 Jul, notebook 10, `n_leg` in the engine)**: trade the top-K names (by cap) instead of all 100 → cost/quarter **3.68%→3.36%** (~9%, small-cap spreads avoided). **Net: top-20 beats the full basket** (Sharpe 0.60 vs 0.57, skew +1.14 vs +1.00); **gross: full slightly better** (0.78 vs 0.75, small replication error) → **parsimony pays NET not GROSS = the RMT answer to frictions, validated**. Honest caveat: the effect is **NON-monotonic** (n=30/50 below the full basket) → the optimum at 20 is partly noise; robust message = "~15-20 names ≥ 100 net, at lower cost". `fig_parsimonious`, `table_parsimonious`. Vega-neutrality unit test (38/38).
- [✅] **Two-window justification** written (methodology §8bis + §6.3: 63d = window-matching of the priced horizon; 252d = spectral health, q≈0.4) — ready for the write-up.
- [✅] Refs consolidated (methodology §13): BBP 2017, Laloux 1999, Potters–Bouchaud 2020, Kritzman (turbulence/absorption), López de Prado.

### ML (enriched 17 Jul — we have the time: regimes via HMM/GMM on top of the supervised layer)
- [✅] **Spectral** features (role B) built (`rmt_daily.parquet` + `ml/features.py`): λ₁/N, λ₂/N, absorption, K, Δλ₁, rotation, entropy, participation ratio, Mahalanobis turbulence + VIX/VRP/vol.
- [✅] **Correlation-spike** target explored (lever 1: the daily spike is predictable, OOS corr +0.40; meta-label on the trade adopted). Skew/drawdown objective confirmed (the regime-only gate cuts the tail).
- [✅] **GMM + HMM regime = stacking feature (17 Jul, `ml/regime.py`)**: GMM(3) pointwise + HMM(3) **MANUAL forward-filtered** (never smoothed Baum-Welch), walk-forward refits, danger state = composite stress centroid (LABEL-FREE → 0 look-ahead), train-only standardisation. **3 causality tests** (perturb the future → past bit-identical). Economics validated: crises → 1.00 (including **Volmageddon 0.98, which the spread missed**), calm → 0.00–0.07. Bug fixed: turbulence alone flattened by the devol → composite centroid.
- [✅] **Steps ④-⑤ EXECUTED (17 Jul, `ml/metamodel.py`, notebook 09)**: shallow XGBoost meta-model (fixed, not tuned on 12 crises), walk-forward purge 63d + embargo 21d, predicts the trade return, veto at the ex-ante 33rd quantile; central test VIX / VIX+spectral / Full.
  **RESULT = NULL (honest)**: near-zero predictive power (corr ŷ,y: VIX +0.02, **spectral −0.14, Full −0.20**) → the trade's quarterly return is unpredictable at this sample size. The ML veto does NOT improve v1_rmt (net 0.56 → 0.57, noise; VIX-only *hurts*, 0.51). **Spectral features do not beat the VIX** (VIX-only even has better crisis recall, 38% vs 0%). **Key finding: detecting a stress regime ≠ predicting the trade will lose** (stress is often exactly when the premium is richest). Respects the plan's "don't oversell the ML" trap.
- [✅] **The real W4 value = the 252d RMT window** (v1_rmt, step 3), not the ML: net Sharpe 0.42→0.56, skew −1.33→+0.80, maxDD −99%→−58.5%, cumulative ×2.5→×12.4.
- [✅] **ML improvement menu presented + decided (17 Jul)**: architecture = **meta-labeling veto** (v1_rmt proposes, the ML vetoes — *NOT a final choice, to revisit*; direct gate as comparison if time); adopted from W4: **continuous labels** (forward Δρ̄ spike), **cost-aware gate** (predicted edge > era cost), **GMM/HMM/XGB ensembling**, **isotonic calibration**; postponed to W5: CPCV.
- [✅] **PRE-REGISTERED PROTOCOL (frozen 17 Jul BEFORE any result — anti multiple-testing)**:
  - **Features — PROTOCOL v2 (enriched 17 Jul BEFORE any walk-forward result, user request)**: 32 features in 5 blocks — **A. deep spectral** (λ₁/N, λ₂/N, K, top-5 absorption, spectral entropy, v1 participation ratio, cross-sectional correlation dispersion, Δλ₁ 21d, rotation 21d, **Mahalanobis turbulence on the CLEANED inverse** raw + 21d); **B. vol structure** (SPX IV + Δ21d, 30→91d term slope, ±50δ skew, vol-of-vol); **C. realised & VRP** (SPX RV21/63, **VRP = IV−RV63**, 252d drawdown, 63d momentum); **D. IV cross-section** (weighted constituent IV, IV dispersion); **E. correlation/signal** (ρ_imp, trail63, clean252, Δρ 21d, base+rmt signals, anchor gap, expanding ex-ante signal percentile, era cost). **Dimensional discipline**: all 32 for XGBoost; FROZEN core set of 8 for GMM/HMM (f_lam1, f_dlam1_21, f_turb21, f_vrp, f_term_slope, f_anchor_gap, f_iv_spx, f_rot21). `ml_dataset.parquet` = 7,221 × 34.
  - **Continuous label**: y(t) = ρ̄_trail63(t+63) − ρ̄_trail63(t) — the realised correlation spike over the next quarter (TRAINING use only, purged).
  - **Models/grids**: GMM 3 components full-cov; Gaussian HMM 3 states (FILTERED probabilities); XGBoost regression {depth 2-3; n_est 100/300; lr 0.05/0.1; subsample 0.8}, selection by purged CV INTERNAL to the training window.
  - **Validation**: expanding walk-forward refit at every rebalance, purge 63 business days + embargo 21 d; veto threshold = ex-ante quantile of the predictions (median, symmetric with v1).
  - **Frozen metrics**: table v0/v1/v1_rmt/ML × gross/net — Sharpe, skew, maxDD, trades, crashes avoided; **bar = net Sharpe 0.42**. Discipline: no full-sample predictive stat looked at before the walk-forward evaluation.
- [✅] **PROTOCOL v3 (17 Jul) — "compute wide, select narrow" — FULL PIPELINE** (①→⑤ all done, see detailed results below): the final model only ever sees **5–8 features** (never the ~35 directly; HMM/GMM: 3–4 max). **Gated pipeline, validation stop at EACH step**: ① point-in-time audit (table feature|window|source|PIT) → ② de-redundancy (clusters corr > 0.8, keep the most interpretable, document the eliminated) → ③ [LABEL DISCUSSION — **re-opened**, y_spike = candidate not adopted] → ③bis generation of the **GMM/HMM regime probabilities as candidate features** (STACKING, decided 17 Jul — unsupervised fit without the label, danger labelling on training history only, walk-forward anti-leakage; on a 3–4 core set) → ④ selection by purged walk-forward XGBoost importance → ⑤ **CENTRAL TEST: "VIX only" (f_vix + term structure) vs "VIX + retained spectral features"**
- [✅] **Step ② de-redundancy DONE (17 Jul, notebook 08) — user-validated**: 35→26 (9 eliminated, 3 clusters |ρ|>0.8, most-interpretable survivor; 2-half stability min 0.68; 0 look-ahead since feature-feature). Then **tightened to 10 by economic judgement** (user directive "keep 10 and move on"; label-free = more robust than in-sample importance on 12 crises): **f_vix, f_term_slope** (vol baseline) + **f_lam1, f_dlam1_63, f_turb21, f_k, f_rot21** (RMT spectral) + **f_vrp, f_drho_imp_21, f_sig_rmt** (economics/signal). HMM/GMM core = {f_lam1, f_dlam1_63, f_turb21, f_vix}. **Regime probability added ON TOP (11th input, no matter what — user directive, no selection for it).** f_cost_era → cost-aware gate only. `ml_features_final.json`. — do the RMT features add predictive power BEYOND the VIX? (the result to nail; more important than the Sharpe). Metrics: predictive (crisis precision/recall) AND economic (net Sharpe), explicit uncertainty (12 crises). **Additions**: `f_vix` (CBOE `cboe.cboe`, 1996–2024, 2 vendor duplicates deduplicated; **VIX3M UNAVAILABLE in the subscription** → term structure = vsurfd ATM 30/91d pillars = `f_term_slope`, full-sample, slope convention documented); `f_drho_imp_21` (the market starts pricing it); `f_dlam1_63` (HMM spec); HY liquidity proxy **skipped** (not cleanly sourceable). `ml_dataset.parquet` = 7,221 × 37 (35 features + date + y_spike candidate).

**Deliverable:** ✅ `fig_mp_spectrum` + `fig_ml_nav` + `fig_central_test` + `fig_regime_timeline` + `fig_feature_importance` + `fig_pred_scatter` + `fig_return_dist` + `table_ml_metrics/central_test/feature_importance`. **WEEK 4 = DONE (17 Jul)**: RMT = the real value (252d window, net Sharpe 0.56, tails cut); ML = an honest null result (no predictive power over the trade return, spectral doesn't beat VIX). RMT look-ahead audit passed; 3 HMM causality tests.

**Pitfall:** ✅ respected — RMT not oversold (ablation: it's the window, not the clipping); ML not oversold (null owned); 0 data leakage (adversarial audit + causality tests).

- [✅] **Improvement levers 1+3 (17 Jul, `ml/experiments.py`, in parallel)**: **(1) target = daily spike** (7,000 obs) → **OOS corr(ŝ, spike) = +0.40** (vs the return's +0.02/−0.20): the spike IS predictable → the target was the problem. BUT the spike veto does not improve the strategy (net Sharpe 0.53) — the danger is priced (stress≠loss confirmed). **(3) regime alone as the gate** → modest tail gain: net skew +1.00 (vs +0.80), maxDD −52.4% (vs −58.5%), Sharpe 0.57; gross skew +1.51, maxDD −44.8%. The only lever that improves the risk profile. **→ To adopt as a variant in W5.**

---

## WEEK 5 (27 Jul → 2 Aug) — Analysis, Robustness & Write-up

**Goal:** turn the code into a 3,000-word academic report.

- [✅] **Step 1 (21 Jul): final strategy `v1_rmt+regime` adopted** (regime-only gate, lever 3) — methodology §14.1bis.
- [✅] **Step 2 — robustness (21 Jul, notebook 10)**: **premium by subperiod (Newey–West)** significant 1996–2019 (t 3.3–4.9) → **compressed post-2020 (t=1.19)** ✓ the limits-to-arbitrage prediction. **Honest nuance**: the RMT gate's Sharpe advantage is NOT uniform (strong 2004-07; **v0 wins 2008-12 and 2013-19**) → the genuinely robust gain = tail shape (skew/DD), not per-era Sharpe. **Robust sensitivity**: net Sharpe 0.46–0.67 (costs ±50%), 0.57–0.58 (veto q), 0.54–0.56 (gate q). Figures/tables: `fig_subperiods`, `table_{premium_subperiods,strategy_subperiods_sharpe,sensitivity}.csv`.
- [✅] **Vanna/Volga analysis (21 Jul, notebook 10 + `utils/greeks`)**: closed-form Volga (∂vega/∂σ) and Vanna (∂vega/∂S) added + tested (finite differences, vanishing at the d2=0 strike). **Narrative**: at entry the ATM straddle is ~2nd-order-neutral (Volga/Vanna ≈ 0), but that is a first-order property — as spot drifts and the position ages, Volga/Vanna grow → the vega-neutrality erodes. Empirically: the book's vega drifts by ~3,600 over Q1-2018 (the neutrality unwinding exactly in the correlation spike). `fig_vanna_volga`. Methodology §14.4.
- [✅] **MFIV robustness (21 Jul, notebook 11)**: MFIV rebuilt on the index leg (full SPX surface, 34 pillars/date, VIX-style Carr–Madan, 116 rebalances). **MFIV > ATM by +1.2 vol points** (19.7% vs 18.5%, put skew) on **100% of dates** → ρ_implied **understated**, bias **Δρ ≈ +0.066 (always positive)**, ρ moves from 0.43 to ~0.50. **The ATM proxy is therefore CONSERVATIVE**: our premium is a lower bound; the true (MFIV) one would be larger → strengthens the thesis. Nuance: index-leg-only effect = UPPER bound on the bias (constituents also have a flatter skew that would partially offset — but net positive since index skew >> single-name, the DMV mechanism). `fig_mfiv_bias`, `table_mfiv_bias`.
- [ ] Write-up: Intro/Client Spec → Theory → Data & Methodology → Results → Limitations → Conclusion
- [✅] **Narrative** — full arc articulated (methodology §14.1): the premium exists (DMV + our 29 years) → frictions kill it (Table V) → RMT (252d estimator + parsimonious leg) + ML (honest null, regime-only cuts the tail) = responses to frictions. Ready for the write-up.
- [✅] Polished figures (9 paper-grade figures, in English) + reading list integrated (methodology §13).
- [✅] **Full A→Z review (21 Jul)**: `main.py` written (WRDS→backtests pipeline), run — reproduces all headline numbers exactly; non-regression 13/14 parquets bit-identical (1 at 1e-9, regime noise); 39 tests reviewed and green; benign warning fixed. **Comments/docstrings humanised** (16 modules + main + tests, natural tone, 0 FR residue, tests still green).
- [✅] Clean the repo (final: `git add`, .gitignore)

**Deliverable:** final report + clean repo.

**Pitfall:** keep at least 3-4 full days for the write-up, not the night before.

---

## Timeline overview

| Week | Dates | Core |
|---|---|---|
| 1 | 29 Jun → 5 Jul | Setup + point-in-time data |
| 2 | 6 → 12 Jul | Theory + implied correlation |
| 3 | 13 → 19 Jul | Baseline backtest |
| 4 | 20 → 26 Jul | RMT (+ light ML) |
| 5 | 27 Jul → 2 Aug | Robustness + write-up |

---

## Defensive priorities (if things slip)

1. **Weeks 1-3** (Data → Signal → Baseline) = a complete, gradeable project; secure it at all costs
2. **Week 4 RMT** = high value added, fast; aim for it
3. **ML** = pure bonus; start it only once everything else holds. A flawless baseline+RMT project beats a rushed ML one.
