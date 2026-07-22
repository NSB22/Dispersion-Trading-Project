# Dispersion Trading Project

**ML-Driven Dispersion Trading Strategy** — MSc Financial Engineering Applied Project, Imperial College London.

This document is both the technical README and the **methodological reference** for the thesis:
every design choice is justified, with the underlying mathematics, in the sections below.

**Quick links:** [Results](#141-the-four-headline-results-the-spine-of-the-paper) ·
[Setup](#10-setup) · [Project structure](#11-project-structure) ·
[Thesis-writing guide](#14-results-consolidation--thesis-writing-guide)

### Results at a glance

| Strategy | Net Sharpe | Net skew | Net max drawdown | Gross Sharpe |
|---|---|---|---|---|
| v0 — unconditional dispersion trade | 0.42 | −1.33 | −99.0% | 0.77 |
| v1_rmt — gated on the RMT-cleaned signal | 0.56 | +0.80 | −58.5% | 0.80 |
| **v1_rmt + regime — recommended** | **0.57** | **+1.00** | **−52.4%** | **0.78** |

The correlation risk premium is real (+0.079 average, t‑NW(63) = 7.4, 1996–2024) but compresses to
statistical insignificance after 2020. RMT filtering and an unsupervised regime overlay cut the
tail risk of the raw trade without adding a supervised ML layer, which turns out to be a clean null
(spectral features do not beat VIX at predicting the trade's return). Full discussion in
[§14](#14-results-consolidation--thesis-writing-guide).

<details>
<summary><b>Table of contents</b></summary>

1. [Economic rationale](#1-economic-rationale)
2. [Data](#2-data)
3. [Universe construction](#3-universe-construction)
4. [Implied volatility](#4-implied-volatility)
5. [Implied correlation](#5-implied-correlation)
6. [Realised side (returns, volatility, correlation)](#6-realised-side-returns-volatility-correlation)
7. [Trade mechanics](#7-trade-mechanics)
8. [The dispersion signal](#8-the-dispersion-signal--two-window-matched-objects)
9. [Methodological pitfalls](#9-methodological-pitfalls-enforced)
10. [Setup](#10-setup)
11. [Project structure](#11-project-structure)
12. [Status](#12-status)
13. [References](#13-references-thesis-reading-list)
14. [Results consolidation & thesis-writing guide](#14-results-consolidation--thesis-writing-guide)
</details>

---

## 1. Economic rationale

The strategy harvests the **Correlation Risk Premium (CRP)**: the market systematically
**overprices the implied correlation** of the S&P 500 index relative to the correlation that
subsequently realises. Foundational reference: Driessen, Maenhout & Vilkov (2009).

Mechanism: investors pay up for **index** options as portfolio insurance, inflating the index's
implied volatility. Because index variance is a function of constituent variances **and** their
correlations, an inflated index IV — for given constituent IVs — mechanically implies an inflated
**implied correlation**. A dispersion trade **sells index volatility and buys constituent
volatility**, which is equivalent to **shorting correlation**:

- Short the index straddle (sell index vol).
- Long a basket of constituent straddles (buy single-name vol).

If realised correlation comes in below implied (the CRP), the trade profits.

A single-name's volatility is mostly idiosyncratic; the index's volatility is mostly systematic
(correlation-driven). Empirically, **average constituent IV > index IV** — the gap is exactly what
correlation "fills in". We already observe this in the raw data (e.g. 2020-03-31: mean constituent
IV ≈ 46.3% vs SPX IV ≈ 37.6%).

---

## 2. Data

All data from **WRDS** (Wharton Research Data Services). Raw data is **never committed** (WRDS
licensing); reproduce with your own credentials (see §10).

### 2.1 Tables used (and why)

| Purpose | Table | Key columns | Notes |
|---|---|---|---|
| Implied vol surface | `optionm.vsurfd<YYYY>` | `secid, date, days, delta, impl_volatility, cp_flag` | Split **by year** (1996→2025). Standardised surface (fixed maturities/deltas) — no manual interpolation. ~536M rows/year ⇒ **filter at source**. |
| Index identity | `optionm.securd`, `optionm.secnmd` | `secid, ticker, index_flag, issuer` | **SPX = `secid` 108105** (unique, `index_flag=1`). |
| Index membership | `crsp.dsp500list` | `permno, start, ending` | Point-in-time S&P 500 membership, 1925→present. |
| Prices / returns | `crsp.dsf` | `permno, date, ret, prc, shrout, cfacpr` | Daily stock file (~108M rows). `ret` = total return (div + split adjusted). |
| Identifier bridge | `wrdsapps_link_crsp_optionm.opcrsphist` | `secid, permno, sdate, edate, score` | Links OptionMetrics `secid` ↔ CRSP `permno`, date-ranged, with a match `score`. |

**Why the "legacy" approach.** The modern all-in-one CIZ product `crsp.idx_const_*_v2` (membership +
float weights + returns in one table) is on schema `crsp_q_mi_hist` — **access denied** under the
Imperial subscription. We therefore reconstruct the universe from `dsp500list` + `dsf`, which is also
the standard method in the literature and has full history.

### 2.2 Identifier linking — `score` filter

`opcrsphist.score` encodes the reconciliation quality (verified empirically):

| score | meaning | kept? |
|---|---|---|
| 1 | CUSIP + ticker + dates all agree | **✅ yes** |
| 2–4 | probable but ambiguous (ADR/non-ordinary, very short windows, name-only match) | ❌ |
| 5 | CUSIP-only match, tickers diverge (rebranding/merger) | ❌ |
| 6 | no match (`permno` NULL) — ~71% of all rows | ❌ |

**Choice: keep `score = 1` only.** Large-cap S&P 500 names all have a score-1 match; ambiguous
links are exactly the small/ADR/merged names we exclude from the universe anyway. An imperfect link
is a *silent* source of bias, so we refuse it.

**Disambiguation.** A single `permno` can carry several valid score-1 links with overlapping
`[sdate, edate]` windows on a given date (which otherwise duplicates the security in the universe and
double-counts its weight). We keep **one secid per permno** — the link with the latest `edate`
(`DISTINCT ON (permno) ... ORDER BY edate DESC`) — guaranteeing a unique mapping.

---

## 3. Universe construction

Implemented in `src/dispersion/data/universe.py` — `get_universe(db, date, n=100)`.

For a rebalancing date $t$:
1. **Membership (point-in-time):** `dsp500list WHERE start ≤ t ≤ ending` — who was *actually* in the
   index that day.
2. **Market cap:** from `dsf` on date $t$: $\text{cap}_i = |\text{prc}_i| \times \text{shrout}_i \times 1000$
   (CRSP `prc` can be negative — a bid/ask-average convention — so we take $|\cdot|$).
3. **Rank & truncate:** keep the top $N=100$ by market cap.
4. **Link:** join `secid` via `opcrsphist` (score = 1).
5. **Weights:** $\;w_i = \dfrac{\text{cap}_i}{\sum_{j=1}^{N}\text{cap}_j}\;$ (renormalised over the top-$N$).

### 3.1 Choice of $N = 100$

The choice is dictated by three forces (validated on 2010/2015/2020 data):

**(a) Accuracy of the implied-correlation formula (primary).** The one-factor index-variance
identity (§5) assumes the universe *represents* the index. Cap coverage of the top-$N$:

| $N$ | cap coverage (2020 / 2015 / 2010) |
|---|---|
| 30 | 43% / 34% / 41% |
| 50 | 54% / 46% / 51% |
| **100** | **70% / 63% / 66%** |
| 150 | 79% / 73% / 76% |

With $N=30$ we ignore ~60% of the index, badly underestimating $\sum_i w_i^2\sigma_i^2$ and biasing
$\rho_{\text{implied}}$. $N=100$ keeps the truncation error small.

**(b) Random Matrix Theory needs a meaningful $N$.** The RMT layer (§8bis) filters the spectrum of an
$N\times N$ correlation matrix estimated on 252-day windows ($q = N/T \approx 0.4$). A small universe
starves it of **spectral resolution**: $N=30$ yields only 30 eigenvalues, with $O(N^{-2/3})$
edge fluctuations that blur the Marchenko–Pastur boundary and leave few bulk eigenvalues to clean;
$N=100$ gives a well-populated spectrum where bulk and signal separate reliably. (The 63-day matrices
are a different object — degenerate, kept only for the window-matched scalar $\bar\rho$; see §6.3.)

**(c) Statistical robustness of $\bar\rho$.** $N=30$ → 435 pairs; $N=100$ → 4 950 pairs. The average
correlation is far more stable and less sensitive to a single outlier.

**Why not more (125+):** diminishing cap coverage (≈5–9% from 100→150), and we start reaching names
with thinner option liquidity (lower-quality IV). Cost is *not* a constraint: the top 100 have **100%
IV coverage** and low turnover (next point).

### 3.2 Backtest period — 1996 → 2024

Frozen after a year-by-year coverage diagnostic (top-100 universe vs valid 91d/$50\Delta$ IV):
**97–100 of 100 constituents carry a valid IV every year from 1996, and the SPX IV is always
present** — no early-data quality cliff. Bounds: the **lower bound (1996)** is the start of `vsurfd`
(the IV side is the binding constraint); the **upper bound (end-2024)** is where CRSP `dsf` ends under
this subscription (so the realised side cannot extend into 2025 even though OptionMetrics does).
Result: ~29 years, ~116 quarterly rebalances, spanning the dot-com bust, the 2008 GFC, COVID-2020 and
the 2022 rate shock. The 63-day realised windows at the first 1996 rebalance are warmed up from
pre-1996 CRSP returns (available far back), so no start-up truncation.

### 3.3 Rebalancing and turnover

Quarterly rebalancing (quarter-end). Measured turnover of the top-100 universe: **~3–8 names per
quarter** (median ≈ 4; the 8 peak is COVID Q1-2020), and 87/100 names persist after one year. The
universe is stable ⇒ low roll cost and a coherent realised-correlation matrix across periods.

### 3.4 Documented approximations

- **Full-cap, not float-adjusted weights.** The true S&P 500 uses free-float weights (IWF factor),
  which lives in the access-denied `idx_const_*_v2`. We use total market cap. Small effect for
  mega-caps but non-zero.
- **Renormalised over the top-100**, not the full 500 (consistent with the truncation in §5).
- **Weights computed only at the ~116 rebalancing dates** (29 yrs × 4), not daily. Positions are then
  **frozen between rebalances** — see §7.

---

## 4. Implied volatility

Implemented in `src/dispersion/data/iv.py` — `get_iv(db, secids, date_start, date_end, days=91)`.

### 4.1 Maturity = 91 days

`vsurfd` provides standardised maturities $\{10,30,60,\mathbf{91},122,152,182,273,365,547,730\}$ days.
**91 days is native** (no interpolation). It is the sweet spot:

- **Short (10–30d):** huge gamma/theta, jump/earnings-dominated, noisy, expensive to roll, too few
  points to estimate realised vol.
- **Long (1–2y):** little usable vega per unit time, illiquid single-name options, term-structure
  contamination.
- **~3 months:** maximal liquidity (index *and* single names), enough vega to express the view,
  aligns with the **quarterly rebalance** (roll ≈ rebalance), and is the horizon where the CRP is
  best documented.

91 calendar days ≈ **63 trading days**, which pins the realised-vol window (§6.2) for an apples-to-
apples comparison.

### 4.2 Delta = ±50 (ATM straddle)

ATM IV is the most reliable (highest vega, least skew-dependent, most liquid). We extract **both
legs**: the $+50\Delta$ call and the $-50\Delta$ put. ATM volatility:

$$\sigma^{\text{ATM}} = \tfrac12\left(\sigma^{\text{call}}_{50\Delta} + \sigma^{\text{put}}_{-50\Delta}\right)$$

The two legs sit at slightly different strikes (ATM-forward effect — visible in the data), and the
trade actually *holds* both (a straddle), so the average is the natural ATM-forward vol proxy.
**Strict mode:** if a leg is missing, $\sigma^{\text{ATM}} = $ NaN (coverage on the top-100 is ~100%,
so this discards almost nothing while keeping a clean symmetric ATM).

### 4.3 Other extraction choices

- Filter invalid surface points at source: `impl_volatility IS NOT NULL AND > 0`.
- Loop over the yearly `vsurfd<YYYY>` tables (clearer/debuggable vs one `UNION ALL`).
- IV only for now (no `impl_strike`/`impl_premium`): the Week-1 deliverable is the **signal**, which
  needs only IV; strikes/premia are extracted later for the **backtest** (Week 3).
- **IV is already annualised** (Black–Scholes convention) — unlike realised vol (§6.2).

---

## 5. Implied correlation

Under a **one-factor approximation** (a single average correlation $\bar\rho$ for all pairs), index
variance decomposes as

$$\sigma_I^2 \;=\; \sum_{i=1}^N w_i^2\sigma_i^2 \;+\; \bar\rho\sum_{i\neq j} w_i w_j \sigma_i\sigma_j .$$

Solving for the average correlation and feeding **implied** vols gives the implied correlation:

$$\boxed{\;\rho_{\text{implied}} \;=\; \frac{\sigma_I^2 - \sum_i w_i^2\sigma_i^2}{\sum_{i\neq j} w_i w_j \sigma_i\sigma_j} \;=\; \frac{\sigma_I^2 - \sum_i w_i^2\sigma_i^2}{\left(\sum_i w_i\sigma_i\right)^2 - \sum_i w_i^2\sigma_i^2}\;}$$

where $\sigma_I$ = SPX implied vol (secid 108105), $\sigma_i$ = constituent $i$ implied vol, $w_i$ =
weights. (This is the CBOE-style implied-correlation construction; DMV 2009, eq. 2.)

**Vol inputs — decided (Week 2, after full DMV reading): ATM 91-day IVs, not MFIV.** DMV build
implied correlation from **model-free implied variance** (Carr–Madan, zero-rate sketch:
$\sigma^2_{MF} = \frac{2}{T}\int_0^\infty \frac{C(K)-\max(S_0-K,0)}{K^2}\,dK$ — the $1/T$ annualises
the expected total variance $E^{\mathbb Q}\!\int_0^T \sigma_t^2\,dt$; the full version discounts and
splits at the forward), which integrates the whole smile. We feed
**ATM ($\pm50\Delta$) 91-day IVs** instead, for three reasons:

1. **Instrument consistency** — the strategy trades ATM straddles (§7): the correlation premium
   measured in ATM-vol space is exactly the one the book can monetise. An MFIV-based measure partly
   reflects wings we do not trade.
2. **Conservative bias** — the index smile is much steeper than single-name smiles, so ATM understates
   index implied variance *more* than component variance $\Rightarrow \rho_{\text{implied}}$ is
   **understated**. If implied > realised holds in ATM space, it holds a fortiori in MFIV space.
3. **Cost** — MFIV needs the full surface (~17 deltas × 2 sides × 100 names × 29 years), delta→strike
   conversion, tail extrapolation, numerical integration. We ran the index-leg reconstruction as a
   robustness check (§14.4): MFIV exceeds ATM on 100% of dates, so the ATM measure is a **conservative
   lower bound** — using MFIV would only make the premium larger.

**Bounds.** From the boxed formula: $\rho_{\text{implied}} \le 1 \iff \sigma_I \le \sum_i w_i\sigma_i$
(index IV below the weighted-average component IV — diversification), consistent with the QA finding
that component IV exceeds index IV on every one of the 7221 days; and
$\rho_{\text{implied}} \ge 0 \iff \sigma_I^2 \ge \sum_i w_i^2\sigma_i^2$, essentially always true (the
concentration term $\sum_i w_i^2$ is tiny for $N=100$ — measured Herfindahl $\in [0.015, 0.037]$).
Both equivalences hold given the strictly positive denominator: $(\sum_i v_i)^2 - \sum_i v_i^2 =
\sum_{i\neq j} v_iv_j > 0$ automatically for $N\ge2$ with positive weights and vols. Both checks are
run and violation counts reported rather than clipped.

**Index–basket basis (documented).** The numerator uses SPX IV (the full 500-name index) while the
sums run over our top-100 basket — the measure embeds an index-vs-basket basis. This is the *same*
basis the strategy itself trades (short SPX straddle vs a 100-name basket, §3.4), so the signal
remains book-consistent.

**Documented limitation:** the one-factor (single $\bar\rho$) assumption collapses the full pairwise
correlation structure into one number — assumed and stated explicitly (DMV eq. 6 makes the same
equicorrelation-factor assumption). The RMT layer (§8bis) revisits the full matrix.

### 5.1 Implementation choices (decided 8 Jul 2026 — data-verified)

- **Missing component IV on a day → renormalise the weights over the available names** that day (the
  exact analogue of the listwise choice on the realised side, §6.3), with an `n_names` coverage column
  and a **coverage floor**: the day is invalidated below 90/100 names (a free guard — observed minimum
  is 91). *The data check that settled it:* a strict "drop any day with a missing name" rule would
  delete **54.9% of days** (3,962/7,221 — ~3 names have no vsurfd option surface for entire quarters
  despite a valid secid link, §9bis, so
  those quarters never reach 100/100), while the missing weight on affected days is only **1.3%
  (median) / 7.2% (max)** of the basket. Renormalisation is innocuous; dropping is destructive.
- **Bound violations stored raw** — no clipping. Violation counts are reported as diagnostics;
  clipping (if ever) is a strategy-layer decision, not a data-layer one.
- **Deliverable:** `signal.parquet` — `date, rho_implied, rho_trailing, rho_forward, premium, signal,
  n_names` (notebook 03 exploration → promoted to `src/dispersion/signal/`).
- `secid` normalised to `Int64` on load (Week-1 cosmetic leftover).

---

## 6. Realised side (returns, volatility, correlation)

Implemented in `src/dispersion/data/returns.py`.

### 6.1 Dividend-adjusted returns — `get_returns`

We use CRSP's **`ret`** directly: the holding-period **total return**

$$ret_t = \frac{P_t + D_t}{P_{t-1}} - 1,$$

where prices are split-adjusted and $D_t$ includes all distributions. CRSP handles every corporate
action via its adjustment factors — we do **not** recompute from prices.

- Not `retx` (price-only return): it shows a spurious drop on each ex-dividend date → fake vol.
- Not manual price ratios: re-implementing split/dividend logic is error-prone.

Output is a **wide panel** (index = date, columns = permno) — pivoting auto-aligns trading calendars
(NaN where a name did not trade), which is the form `rolling`/`.corr()` need.

### 6.2 Realised volatility — `realized_vol`

Work in **log-returns** $r_t = \ln(1+ret_t)$ (additive, clean variance aggregation). Rolling sample
standard deviation, **annualised**:

$$\sigma^{\text{realised}}_{i}(t;W) = \sqrt{252}\;\times\;\operatorname{std}_{\,\tau\in(t-W,\,t]}\!\big(r_{i,\tau}\big).$$

The $\sqrt{252}$ is essential: realised vol is computed from daily returns (a *daily* number) and
must be annualised to be **comparable to IV**, which is already annualised. Forgetting it would make
$\rho_{\text{implied}}-\rho_{\text{realised}}$ meaningless.

**Windows:** $W=63$ trading days (≈ 91 calendar days — **matches the 91-day IV horizon**, primary)
and $W=21$ (≈ 1 month, reactive secondary / ML feature). **Minimum observations** (~80% coverage):
$W=63 \Rightarrow$ min 50; $W=21 \Rightarrow$ min 17. Below that, the value is NaN — a vol/correlation
from too few points is unreliable, and this keeps the §6.3 correlation objects usable (the RMT layer
itself runs on 252-day windows — §8bis).

### 6.3 Realised correlation

The $N\times N$ realised correlation matrix on a rolling window. **Two design decisions:**

- **`listwise` (complete-case), parameterised.** Pairwise estimation (each pair on its overlapping
  days) can yield a **non-PSD** matrix, which breaks RMT. Listwise keeps only dates where all
  (retained) names have data → a clean **positive-semi-definite** matrix. Exposed as `method=` so it
  can be switched at the end of the project. (For top-100 large caps, NaNs are ~0.1%, so listwise
  discards almost no rows.)
- **Storage:** **not** the full $N\times N$ matrix daily. We store the **daily scalar $\bar\rho$** plus
  the **full matrices only at rebalancing dates**. Fully reversible by recomputation from `data/raw`.

**Average correlation $\bar\rho_{\text{realised}}$ — choice: weighted (formula-consistent).** To make
the spread $\rho_{\text{implied}}-\rho_{\text{realised}}$ meaningful, both legs must be the *same* kind
of average. $\rho_{\text{implied}}$ comes from the weighted index-variance identity (§5), so the
realised counterpart uses the matching weighted average. With $v_i = w_i\sigma_i$ (cap weight ×
realised vol) and off-diagonal sums only ($\rho_{ii}=1$):

$$\bar\rho_{\text{realised}} = \frac{\sum_{i\neq j} w_i w_j \sigma_i\sigma_j\,\rho_{ij}}{\sum_{i\neq j} w_i w_j \sigma_i\sigma_j} = \frac{v^\top R\,v - \sum_i v_i^2}{\left(\sum_i v_i\right)^2 - \sum_i v_i^2}.$$

The equal-weighted mean $\frac{2}{N(N-1)}\sum_{i<j}\rho_{ij}$ is also available (`method='equal'`) as a
robustness check. Weights are the **frozen rebalance weights** (§7.1); vols are the §6.2 realised
vols on the same date.

**Empirical validation** (`realized_corr_matrix` + `average_correlation`): on 2020-03-31 (COVID)
$\bar\rho_{\text{realised}} \approx 0.77$ vs $\approx 0.44$ on 2019-09-30 (calm) — realised correlation
spikes in stress, as expected.

**Rank deficiency — a feature, not a bug.** With $T=63 < N=100$, the matrix is PSD but **singular**
(rank $\le T-1=62$, hence $\ge 38$ zero eigenvalues — exactly 38 in the generic full-panel case). The
**market mode** carries a median $\lambda_1/N \approx 34\%$ of total variance across the 116 stored
matrices, spiking to ~75% at the COVID extreme (2020-03-31; 0.73 on 2011-09-30, 0.68 on 2008-12-31).
In the Marchenko–Pastur convention $q=N/T\approx1.6>1$ ($T<N \Rightarrow$ point
mass at zero): the 63-day spectrum is **degenerate**, so spectral filtering cannot run on it. The RMT
layer therefore operates on **252-day** windows ($q\approx0.4$, §8bis); the 63-day object is kept only
for the window-matched scalar $\bar\rho$ (two windows, two uses — see §8bis).

---

### 6.4 Dataset cleaning & assembly (all series)

`src/dispersion/data/assemble.py` — `build_dataset(db, date_start, date_end, ...)` orchestrates
everything over the full period and writes tidy (long) parquet files.

**Orchestration.** A **master trading calendar** is the *strict intersection* of OptionMetrics SPX-IV
days and CRSP `dsf` days (a date is kept only if both bases have it). Rebalances are the last master
day of each quarter. The loop is **point-in-time per quarter**: the universe/weights are taken at the
rebalance, the constituent IV is pulled only for that quarter's secids over that quarter, and returns
carry a trailing buffer so the 63-day windows are warm.

**Cleaning policy** (parameterised):
- **Calendar alignment:** everything reindexed to the master calendar (the intersection above).
- **IV gaps:** bounded forward-fill, **`ffill_limit = 3`** trading days (IV is persistent; longer gaps
  stay NaN as genuine missing). The SPX series is ffilled but never outlier-clipped.
- **IV outliers:** drop to NaN outside the **0.1st / 99.9th percentiles** of the pooled constituent IV
  distribution (robust to data errors; light enough to preserve true crisis spikes, e.g. COVID).
  Order matters: outlier → NaN **then** ffill, so a removed outlier can be filled if the gap is short.

**Deliverables** (`data/processed/`, long format):

| File | Columns | Grain |
|---|---|---|
| `iv_index` | `date, iv_call_50, iv_put_50, iv_atm` | daily (SPX) |
| `iv_components` | `rebalance_date, date, permno, secid, iv_atm` | daily × active constituent |
| `weights` | `rebalance_date, permno, secid, weight, market_cap, rnk` | per rebalance |
| `realized_vol` | `date, permno, vol_21, vol_63` | daily × constituent |
| `realized_corr` | `date, rho_bar, rho_bar_equal` | daily |
| `corr_matrices` | `rebalance_date, permno_i, permno_j, corr` | per rebalance |
| `signal` | `date, rho_implied, rho_trailing, rho_forward, premium, signal, n_names` | daily (Week 2) |
| `surface` | `rebalance_date, date, permno, secid, days, cp_flag, iv, premium, strike` | daily × name × pillar (10/30/60/91d) × leg — **raw**, marking-grade (Week 3; SPX rows have `permno = NA`) |
| `spots` | `rebalance_date, date, permno, secid, close, cfadj` | daily × name (+ SPX) — `secprd` closes + split factors, same convention as the strikes |
| `rates` | `date, days, rate` | daily zero curve (`zerocd`, %) |
| `returns` | `date, permno, ret` | daily × constituent (union), from 1995 (252d RMT buffer) |

The dataset is fully reproducible from `data/raw` (here, directly from WRDS) by re-running
`build_dataset`; `signal.parquet` is rebuilt from the base files by
`dispersion.signal.implied_corr.build_signal` (§5, §8). The signal-grade series keep the cleaning
policy above; `surface`/`rates`/`returns` are stored **raw** (marking-grade — any guarding happens at
consumption, in the engine).

## 7. Trade mechanics

### 7.1 Vega-neutral entry, frozen positions

At each rebalance the portfolio is set **Vega-neutral** (short index vega = long basket vega) using
ATM straddles, then **held to the next quarter**. Positions are *not* re-weighted daily; the **Vega
drifts** between rebalances. This drift is accepted — it **is** the convexity risk (Volga/Vanna)
analysed in Weeks 3/5. Consequently weights are needed only at rebalancing dates, and the daily
signal uses the **frozen** weights of the last rebalance (consistent with the book actually held).

### 7.2 Delta hedging with the underlying

An ATM-forward straddle starts $\approx$ delta-neutral but accumulates delta via gamma as spot moves.
We neutralise it **daily by trading the underlying** (index future / stock), **not** by re-trading
straddles. Reasons:

1. **It is the mechanism that makes the bet clean.** Daily delta-hedging a straddle is *gamma
   scalping*: the hedge P&L captures **realised variance**. To first order,

   $$d\Pi \approx \tfrac12\,\Gamma\,S^2\big(\sigma_{\text{realised}}^2 - \sigma_{\text{implied}}^2\big)\,dt,$$

   i.e. exactly the realised-vs-implied (and hence correlation) bet. The underlying is pure delta-1
   (no theta/vega), so it removes *only* the unwanted directional risk and preserves vega/gamma/theta.
2. **Cost.** Option bid-ask spreads are wide (single-name 3M ATM: ~1–5%+ of premium); re-striking
   means crossing them twice, on ~100 names, daily. The underlying (e.g. ES future) costs ~0.5–2 bps,
   and only the *net delta* is traded. Re-striking daily would not survive transaction costs.
3. **Structure.** Re-striking daily resets strikes/greeks and churns theta+spread, destroying the
   "hold dispersion to expiry" exposure. Re-striking happens **only at the quarterly rebalance**.

### 7.3 Engine design decisions (Week 3 — frozen, with the rationale to carry into the write-up)

1. **Book sizing: DMV wealth normalisation.** The book is expressed in **fractions of wealth**,
   anchored at −100% of wealth in the index straddle; the component-leg sizes follow from the
   relative-vega-neutral ratios (eq.-10 type), and the residual index delta-hedge plus the
   money-market account complete the balance sheet. *Rationale (thesis point):* (i) returns become
   **directly comparable to DMV Table II** — our only external benchmark (weights
   −100/+101.12/−32.54/+131.42, 10.37%/mo, Sharpe 0.73), which is the whole purpose of the
   unconditional v0; (ii) it is a pure *reporting* convention — the strategy is scale-invariant, so a
   vega-notional view is a re-scaling of the same strategy, not a different one.

2. **Delta hedge: the index alone, daily.** Under the one-factor structure — the *same* assumption
   underlying $\rho_{\text{implied}}$ (§5) and DMV's eq. (6) — eq. (11) shows the residual delta of
   the vega-hedged book loads on the **index alone**, while idiosyncratic deltas across ~100 long
   straddles diversify away. *Rationale (thesis point):* (i) **model consistency** — we assume
   one-factor everywhere, so the index hedge is the hedge implied by our own model; (ii) **costs** —
   one instrument at ~0.5–2 bps (index future/ETF) instead of ~100 daily single-name lines each
   paying spread + impact; DMV Table V shows frictions *halve* the strategy's returns, so minimising
   hedge legs is limits-to-arbitrage-aware design, not cosmetics; (iii) consistency with their
   −32.54% average index position. Per-name hedging is a natural extension, left out of scope (§14.5).

3. **Expiry: intrinsic settlement at the next rebalance.** The 91d straddle bought at rebalance $R$
   expires within ±3 calendar days of the next rebalance (quarters run 89–92 days). The book settles
   at **intrinsic value** $|S-K|$ per leg at the next rebalance (the ±3d gap is a documented
   approximation) — DMV's held-to-maturity convention (spread paid once), and no marking of
   near-expired options where the surface has no pillar.

4. **Sub-30d marking: σ frozen at the daily 30d pillar.** Daily marks interpolate the ATM pillars
   30/60/91 **linearly in total variance**; below 30 days of residual maturity,
   $\sigma(\tau) :=$ that day's $\sigma(30d)$ (re-read **daily** — only the short-end *slope* is
   ignored, never the level moves). One uniform rule across names and dates — the vendor's sparse 10d
   pillar (~37% populated, §9bis) would otherwise make the marking panel heterogeneous. Second-order
   by construction of (3): near-expiry values are never realised through marks, they settle at payoff.

**Documented marking approximation — ATM proxy.** Aged positions are marked by applying the ATM
(±50Δ) IV at the residual maturity to the **fixed entry strike** via BS: as spot drifts, the true
option leaves the ATM point and the mark ignores the smile at the drifted moneyness. This is
second-order for the strategy's *cumulative* return: each quarter's P&L is anchored by real entry
premiums (`impl_premium`) and intrinsic settlement (3), so the ATM proxy shapes only the
intra-quarter path (daily P&L, drawdowns, hedge deltas). Full smile marking (delta-grid extraction)
is a natural refinement, left out of scope.

---

## 8. The dispersion signal — two window-matched objects

**Decided (Week 2, after DMV):** the IV at $t$ prices the window $[t,\,t+63\text{ trading days}]$
(≈ 91 calendar days), while our stored realised correlation at $t$ is **trailing** (past 63 days).
Comparing them mixes two windows, so we build **two distinct objects, one per use**:

**The premium (ex-post, validation).** Compare the price paid at $t$ with what *then* realised over
the exact window that price covered:

$$\Pi_t \;=\; \rho_{\text{implied},t} \;-\; \rho^{\text{fwd}}_{\text{realised},t},
\qquad
\rho^{\text{fwd}}_{\text{realised},t} \;=\; \rho^{\text{trail}}_{\text{realised},\,t+63\text{td}}$$

— the forward series is the trailing series shifted **back by 63 trading days** (no new computation).
$E[\Pi_t]>0$ over the sample = the CRP exists. This is DMV's **window-matching**. It uses future
information relative to $t$: legitimate for ex-post analysis (validation chart, report), **never** as
a trading input. *Documented caveat: the trailing value at $t+63$ uses weights frozen at the rebalance
covering $t+63$, which can differ from those at $t$ across a quarter boundary — second-order for
validation purposes.*

**The signal (ex-ante, tradeable).**

$$S_t \;=\; \rho_{\text{implied},t} \;-\; \rho^{\text{trail}}_{\text{realised},t}$$

uses only information available at $t$. The two are linked by the decomposition

$$S_t \;=\; \Pi_t \;+\; \big(\rho^{\text{fwd}}_t - \rho^{\text{trail}}_t\big),$$

i.e. the signal = the premium + the trailing-forecast error. Correlation persistence makes $S_t$ an
**ex-ante estimator of the premium** — the standard premium/proxy structure of any harvesting strategy.

**Strategy design implication.** DMV's dispersion strategy is **unconditional** (no signal — short
correlation every period). Thresholding $S_t$ (Week 3) and ML timing (Week 4) are *our extensions*.
The backtest therefore runs the **unconditional baseline v0 first** (comparable to DMV's Table II
benchmarks, §8bis), then the signal-conditioned v1 — separating "the premium exists" from "we can
time it".

---

## 8bis. DMV (2009) — benchmarks and conventions adopted (full reading, Week 2)

Driessen, Maenhout & Vilkov (2009), *The Price of Correlation Risk: Evidence from Equity Options*,
read in full with all core derivations reworked: eq. (2) index-variance decomposition; eq. (3) index
VRP = weighted individual VRPs + correlation terms (Itô + Girsanov, convexities cancel in
$\mathbb{Q}-\mathbb{P}$), identification via VRP$_{\text{indiv}} \approx 0$ empirically; eq. (4)–(5)
MFIV (Carr–Madan); eq. (6) equicorrelation factor structure; eq. (8)–(11) straddle returns and the
**triangular hedge** (vega first — the vol-of-vol cancels in the relative-vega ratio — then residual
delta with the index alone).

### Benchmarks (sanity guards for the Week-3 backtest)

Orders of magnitude, **not** replication targets (their sample, 1-month horizon and universe differ):

| Quantity | DMV value |
|---|---|
| Index $\sqrt{RV}$ / $\sqrt{MFIV}$ | 20.80% / 24.69% (implied ≫ realised) |
| Single-name $\sqrt{RV}$ / $\sqrt{MFIV}$ | 41.44% / 38.97% (**sign flips**) |
| Strategy weights | −100% index straddle / +101.12% single-name straddles / −32.54% index (delta) / +131.42% riskless |
| Straddle ratio | 0.58 single-name straddle per index straddle |
| Gross performance | 10.37%/mo, Sharpe 0.73, skew −0.28, α=10.59%/mo (t=1.96), **β=0.028 (t=0.02)** |
| Net of bid-ask (Table V) | return ÷ 2, Sharpe 0.41, α dead (t=0.77); CBOE margins without netting → infeasible for γ≤2 |
| Correlation premium (Table IV) | λ_corr = 17.5%/mo (t=2.56, Shanken), R²=89%; β_corr: index −0.96 vs individual −0.24 |

**Rule:** after the unconditional baseline v0, compare — a large deviation triggers a bug hunt before
any interpretation.

### Conventions adopted

1. **Window-matching** (§8): the premium uses realised correlation measured *forward* over the IV
   horizon.
2. **Costs at the adverse quote**: bid-to-maturity for short legs, ask-to-maturity for long legs;
   spread paid once (held to maturity). Cost realism is **first-order** — DMV Table V halves returns.
3. **Newey–West (~63 lags)** for every t-stat on overlapping daily windows; non-overlapping
   (quarterly) tests as robustness.
4. Moneyness in **BS deltas**, not strike/spot (already our extraction grid ✓).
5. **Historical index composition** at each date (already point-in-time ✓).
6. Hedge-ratio greeks are **relative** (per dollar invested: vega/option-price). OptionMetrics/BS give
   per-contract greeks; mixing conventions produces a silently wrong hedge. **Implemented** in
   `dispersion.utils.greeks` (BS + Black-76 pricing/greeks, per-contract↔relative conversions,
   put-call-parity forward, implied vol, straddle aggregation) with a 10-test pytest suite
   (`tests/test_greeks.py`): closed forms, put-call parity, bump-and-reprice finite differences, the
   **anti-trap test** (a relative-vega hedge is wealth-vega-neutral; a per-contract-vega ratio leaves
   a residual equal to the price ratio), and reproduction of the real 2024-06-28 SPX standardised
   premiums to ±0.5%.

### Decided (Week 3, notebook 04): premiums from the surface, greeks analytic

Evidence (notebook `04_explore_premiums`, test date 2024-06-28 + coverage checks 1996/2005/2015):
`vsurfd` natively carries `impl_premium` and `impl_strike` on our frozen grid (91d, ±50Δ), fully
populated since 1996 (SPX 504/504 rows in each test year; point-in-time universe 198–200/200 at the
test rebalances). Naive BS with $q=0$ misprices the standardised premiums by **±5%** (underprices
puts, overprices calls — the dividend signature), while the dividend yield backed out of
`impl_premium` is **identical across the two SPX legs (1.04%)**: the vendor premiums are
forward-consistent. Replicating them ourselves would require index dividend yields plus
American-exercise / discrete-dividend handling for single names — heavy and fragile.
**Decision:** straddle **prices** come from `impl_premium` (re-extraction alongside the IV);
**greeks** are computed analytically (BS/Black-76 from IV + strike + forward — via `optionm.fwdprd`
or put-call parity — and `zerocd` rates), which we need anyway for the *relative*-greeks conversions
and their unit test (§8bis conventions).

**Ageing-position marking (decided): daily mark-to-surface.** The book is revalued **daily** by
interpolating the standardised surface at each position's residual maturity (maturity pillars below
91d, linear in **total variance** at fixed delta). This yields effective daily delta-hedging (as the
plan specifies), a vega/gamma/theta P&L decomposition, proper drawdown statistics, and the daily
series the ML layer (Week 4) needs as labels. The DMV holding-period version stays available for free
by quarterly aggregation (benchmark comparison). Constant-IV marking was rejected outright: a vol
strategy's P&L *is* vega×ΔIV — freezing the IV assumes away the object being traded. Implies a
multi-pillar re-extraction, bundled into the single dataset rebuild (with the `impl_premium` columns,
the §9bis hardening, and Week 4's `returns.parquet`).

### Decided (Week 3, notebook 05): transaction costs — calibrated parametric grid

**Calibration on real `opprcd` quotes** (notebook `05_explore_costs`): median relative bid-ask spread
(full bid→ask, % of premium) of ~ATM options (|Δ| ∈ [0.35, 0.65]), 60–120 days to expiry, open
interest > 0, pooled over three quarter-ends per era:

| Era | SPX | Mega-caps (rnk 1–10) | Small lines (rnk 90–100) |
|---|---|---|---|
| 1998 | 0.9% | 6.8% | 9.8% |
| 2005 | 1.6% | 6.1% | 5.4% |
| 2012 | ⚠ 7.6% | 1.1% | 2.6% |
| 2020 | 0.7% | 3.0% | 7.9%* |
| 2024 | 0.6% | 1.2% | 3.1% |

**Why parametric rather than raw `opprcd` spreads.** (i) The 2012 SPX cell (tight IQR, n=130) is an
**end-of-day quote artefact** of the pit era — quoted EOD spreads ≠ effective spreads (the
effective-vs-quoted literature puts effective at ~30–50% of quoted for options); a raw extraction
would inherit era-inconsistent measurement, not economics. (ii) The 2020 small-cap cell includes
2020-03-31 — spreads widen in stress, a real property, but one to model as a *scenario* (S5), not to
hard-code from one COVID quarter. (iii) A full per-name/per-day extraction (~2 days of work) would
still face quoted≠effective.

**The grid (frozen):** three groups — SPX / rnk 1–50 / rnk 51–100 — with **linear-in-time
interpolation** between 1996 and 2024 anchors, smoothing the noisy cells:
SPX 1.0% → 0.6%; large 7.0% → 1.2%; small 10.0% → 3.0% (clamped outside). The declining anchors ARE
the friction-compression story of §8bis (limits to arbitrage) embedded in the cost model.

**Application (DMV "spread paid once", held to maturity):** each entry leg pays **½ × relative
spread × premium** (`impl_premium` is mid-like, we cross half the quoted spread); the daily delta
hedge pays **1 bp of traded notional** (|Δn|×S, index future); intrinsic settlement pays no spread.
**Conservative by construction:** calibrated on *quoted* spreads while effective is ~2× tighter —
net results are an honest lower bound. The grid is stress-tested at ±50% in the sensitivity analysis
(§14.4); the net Sharpe stays above the v0 baseline even at ×1.5 costs.

The build also stores a daily-returns panel (`returns.parquet`, long: `date, permno, ret`), which
feeds the 252-day RMT matrices and the spectral ML features.

### RMT specification update (Week 4)

**Notation (decided, Week 4):** in RMT sections and code, the Marchenko–Pastur aspect ratio is
written **`q_mp` = N/T** (Bouchaud–Potters convention); the plain letter $q$ keeps meaning the
dividend yield in all pricing contexts (`utils/greeks.py`, `engine.py`). Design decisions:
EWMA de-volatilisation with **λ = 0.94** (RiskMetrics; λ ∈ {0.90, 0.97} is a natural sensitivity);
cleaned matrices computed **daily** (needed for the daily $\bar\rho_{\text{rmt}}$ variant and for
spectral-dynamics features); re-injection runs **both** paths — a full `signal_rmt` variant
re-backtested against the baseline (role A; the 252-vs-91-day horizon mismatch is documented as an
estimator variant) *and* the spectral features for the ML layer (role B). The Laloux correction is
**part of the cleaning pipeline** (it sets the clipping edge), not a separate estimator — there is
one $\bar\rho_{\text{rmt}}$, Laloux-corrected by construction. The naive-edge comparison is reported
as a diagnostic (K = 7 Laloux-corrected factors vs 4 naïvely, `fig_mp_spectrum.png`).

The stored 63-day matrices ($q_{mp}\approx1.6$, singular) are unusable for spectral filtering. RMT
operates on **252-day** windows ($q_{mp}\approx0.4$):
de-volatilise (EWMA) + standardise → diagonalise → effective MP edge with the **Laloux correction**
$\lambda_+^{\text{eff}} = (1-\lambda_1/N)(1+\sqrt{q})^2$ → clip the bulk to a constant **preserving the
trace** → renormalise (diag = 1, PSD guaranteed). Unit tests: diag/PSD/trace + **iid-simulation test**
(empirical spectrum ≈ MP density; filter ≈ identity). Role A: de-noised $\bar\rho_{\text{realised}}$;
Role B: spectral ML features ($\lambda_1/N$, absorption ratio, $K$ = #eigenvalues > $\lambda_+$,
$\Delta\lambda_1$, dominant-eigenvector rotation). **ML layer (enriched, 17 Jul):** regime
identification via **GMM** (soft calm/stress/crisis probabilities on the spectral+vol features,
upgrading the initially planned K-Means) and a **Gaussian HMM** (temporal regime persistence +
transition matrix; strictly ex-ante **filtered** probabilities — forward algorithm only, never
smoothed, and walk-forward expanding fits to avoid look-ahead), compared against supervised XGBoost
under purged walk-forward. **Model wiring decided (17 Jul): stacking, not a parallel vote** — the
unsupervised regime models (GMM soft-cluster, Gaussian HMM with filtered transition dynamics) are fit
*without the label* on a small feature subset and emit a regime probability; that probability enters
the XGBoost candidate pool and must earn its place through selection like any other feature (danger
labelling of clusters/states uses training-window history only, to avoid leakage). Rationale: in a
~104-quarter / 12-crisis sample this spends no crisis degrees of freedom on regime detection and lets
one supervised model learn how much to trust the regime signal, instead of an arbitrary equal-weight
average. The regime probability fitted on spectral inputs counts as a *spectral-derived* feature, so
the VIX-vs-VIX+spectral central test stays interpretable. **Trade architecture: meta-labeling veto**
— the primary signal (v1_rmt gate) proposes the trade, the ML layer vetoes it when the predicted
correlation spike is too high; with a continuous spike label
$y(t)=\bar\rho^{63}_{t+63}-\bar\rho^{63}_t$ (training only, purged), a cost-aware gate (trade only if
predicted edge > the era's §8bis cost), GMM/HMM/XGB probability ensembling and isotonic calibration.
The protocol (features, label, hyperparameter grids, purge 63d + embargo 21d, filtered-only HMM
probabilities, the net-Sharpe 0.42 bar) was **fixed before any result was computed** — a
pre-registration to guard against selection on the outcome. The **parsimonious leg** (use the spectral
structure to trade fewer names → less spread paid — the RMT answer to frictions) is implemented and
reported in §14.1 (result 3). The **dual-window justification** (63d = window-matching of the priced
horizon; 252d = spectral health) is discussed above. Refs: Bun, Bouchaud & Potters (2017,
arXiv:1610.08104); Potters & Bouchaud (2020).

### Research angle: limits to arbitrage

DMV show the premium exists but is largely killed by frictions (single-name spreads × a 101% leg;
margins) — market-makers earn it (net flows, margin netting). Our RMT (parsimonious leg) and ML
(enter only when predicted edge > cost) are *responses to frictions*. This gave a falsifiable
prediction for our 29-year sample (theirs stops in the early 2000s): the premium should compress over
time as frictions fell. The subperiod analysis of $\Pi_t$ **confirms it** — the premium is strongly
significant through 2019 and compresses to insignificance post-2020 (§14.4).

---

## 9. Methodological pitfalls (enforced)

- **Survivorship / look-ahead bias:** index composition is *strictly point-in-time* (`dsp500list` as
  of each date) — never the current membership.
- **Data leakage (ML phase):** walk-forward validation with purging/embargo (López de Prado).
- **Filter at source (SQL):** never extract everything then prune — the option tables are ~0.5B
  rows/year.
- **Signal comparability:** implied (already annualised) vs realised (annualised by us, $\sqrt{252}$)
  must be on the same footing.

---

## 9bis. Dataset QA & known limitations (documented)

The assembled dataset passed a **6-dimension adversarial QA** (each dimension computed on the real
parquet files, every flagged finding independently re-verified). Verdict:

| Dimension | Status | Key result |
|---|---|---|
| Weights integrity | **PASS** | 116×100 = 11 600 rows; Σweights = 1 ± 4e-16 everywhere; 0 duplicate permno/secid; rnk 1..100 complete |
| Point-in-time / anti-look-ahead | **PASS** | 116 rebalances all quarter-end; WRDS-confirmed membership (6/6); no data before the 1st rebalance |
| Coverage / NaN over time | **WARN** | iv_index 0 NaN; iv_components 0.13% NaN; 1 issue → Aug-2020 vendor gap (below) |
| Value plausibility (29y) | **PASS** | ρ̄ ∈ [0.08, 0.83] (weighted; equal-weighted [0.06, 0.80]); **component IV > index IV on all 7 221 days** (mean ratio 1.60); crisis peaks correctly dated |
| Cross-file consistency | **WARN** | calendars/keys align exactly; minor: 3 names lack option IV some quarters (below) |
| Cleaning correctness | **PASS** | ffill bound respected (max run 4d); SPX peaks un-clipped; strict-intersection calendar exact |

A single external issue was confirmed (the Aug-2020 OptionMetrics gap). All findings below are
documented; **none is a pipeline bug**.

- **OptionMetrics vol-surface outage, 2020-07-27 → 2020-08-18 (17 trading days).** Over this window
  the *entire* `vsurfd` surface (every secid, maturity and delta — SPX and all constituents) has
  `impl_volatility = NULL`: the structural rows exist but OptionMetrics never populated the
  interpolated IV. It is a **vendor computation gap**, not a pipeline error. These dates are therefore
  absent from every daily series (the master calendar is a strict intersection), so 2020 carries 236
  trading days instead of ~253. **Handling: accepted as-is, no fabrication** — the impact is 17 of
  7 221 signal days, inside the held-position interval of the Q3-2020 quarter (positions are frozen
  between rebalances anyway). Raw option prices (`opprcd2020`) do exist on those days, so a bespoke
  surface recomputation is possible but was judged out of scope (it would re-introduce a methodology
  inconsistent with the rest of the dataset). The only other multi-day calendar break, 2001-09-11→14,
  is the legitimate post-9/11 NYSE closure.
- **Listwise rank deficiency at 6 rebalances** (1998-06-30, 2002-12-31, 2004-12-31, 2008-03-31,
  2017-09-29, 2018-12-31): a newly-listed constituent without 63 days of history is dropped by the
  complete-case rule, so the matrix is 99×99 and $\bar\rho$ there averages 99 names while the weights
  assume 100 (~0.4% of rebalances).
- **Missing constituent IV for 3 names** (permno 66157, 25267, 66026) on some quarters — 43
  name-quarters out of 11 600 (0.37%): OptionMetrics has no option surface for them those quarters, so
  the implied-correlation aggregation runs on <100 names there (they remain present in weights and the
  realised series).
- **Realised-vol tail NaN before delistings** (1 624 `vol_63` / 1 923 `vol_21` obs, ~0.2%): a name's
  returns dry up a few days before it leaves the universe, so it keeps its frozen weight but has no
  realised vol for those last days (minor effect on the $w_i\sigma_i$-weighted $\bar\rho$ at the very
  end of a holding interval).

**Week-2 full-project audit (four independent axes: data layer, signal layer, mathematics, doc
consistency).** Zero critical/computational findings; $\rho_{\text{implied}}$ independently recomputed
on 6 test dates (max deviation 8e-16); every quoted statistic reproduced from the parquets. Addenda:

- **The single $\bar\rho$ NaN (2015-07-22) — mechanism identified.** A knife-edge listwise
  interaction: a delisting name (permno 13598) passes the per-name `min_obs=50` filter with exactly 50
  observations, its NaN days then shrink the complete-case row set below `min_obs` for the *whole*
  panel → `realized_corr_matrix` returns `None` for the day. Neighbouring days (10–21 Jul 2015) were
  estimated on 51–62 rows instead of 63 (slightly noisier, accepted). One day in 7 221; kept as NaN.
- **Truncated NaN tails in `iv_components` — FIXED at the Week-3 rebuild (8 Jul).** The cleaning
  reindex now spans each quarter's full active window: +2,183 rows materialised (2,032 NaN tails +
  151 cells filled by the documented ≤3-day ffill extension). Effect on the signal: 139 days moved by
  at most 8e-3 in `rho_implied` (headline stats unchanged: premium +0.079, t-NW = 7.4). All other
  signal-grade parquets reproduced **bit-identical**.
- **Latent code guards — HARDENED at the Week-3 rebuild (8 Jul):** `iv.py` missing-leg guard fixed
  (np.nan, no TypeError path); `universe.py` uses `ROW_NUMBER` with a deterministic permno tiebreak
  (an exact cap tie at rank 100 can no longer admit >N rows); `assemble.py` no longer has a
  stale-row fallback for the rho-bar vols; `implied_corr.py` hard-fails on index-IV/spine calendar
  mismatches. Still latent (unfixable without WRDS-side data): a constituent suspended (no `dsf`
  row) on the exact rebalance date would silently drop to the 101st name.
- **Marking-grade data limitations (Week-3 rebuild QA):** the 10-day surface pillar is only
  partially populated by the vendor (~37% of component rows vs the full 30/60/91 pillars; also
  sparse for SPX) → the engine needs a sub-30d fallback rule (flat extrapolation from 30d or
  near-expiry settlement), posed at the engine-design step. `zerocd` is missing on 10 of the 7,281
  calendar days → rates need a bounded ffill at consumption. Universe **rotation at settlement**:
  ~5 positions per quarter (621 over the 115-quarter backtest) settle on the *previous day's* close —
  a name leaving the top-100 has no settlement-day row in its quarter's `spots` partition. One-day
  staleness on ~1% of the book; fix at the next rebuild by extending each partition by one day.

## 10. Setup

### 10.1 Prerequisites
- [uv](https://docs.astral.sh/uv/) (package manager); Python 3.12 (uv installs it if missing).
- A WRDS account with OptionMetrics + CRSP access.

### 10.2 Install
```bash
uv sync
```

### 10.3 Configure access & imports (`.env`)
```bash
cp .env.example .env
```
Edit `.env`:
- `WRDS_USERNAME` — your WRDS username.
- `PYTHONPATH` — **absolute** path to this repo's `src/` (quote it if it contains spaces). Makes the
  `dispersion` package importable.

The password is never stored; WRDS prompts once and offers to create a `.pgpass` (outside the repo).

> **Why PYTHONPATH and not an editable install?** Under macOS, when the repo is in a TCC-protected
> location (e.g. `~/Desktop`), the `.pth` written by an editable install gets a `com.apple.provenance`
> xattr that Python's startup `site`/`io.open_code` silently refuses to read — breaking
> `import dispersion`. `PYTHONPATH` (read from the environment, not a file) sidesteps this. VS Code
> reads `.env` automatically (`.vscode/settings.json`); in the terminal pass `--env-file .env`
> (uv does **not** auto-load it).

### 10.4 Run
```bash
# Test the WRDS connection (crsp / optionm / comp access)
uv run --env-file .env python -m dispersion.data.wrds_client

# Run a notebook headless
uv run --env-file .env jupyter nbconvert --to notebook --execute --inplace exploration_notebooks/01_explore_vsurfd.ipynb
```
In VS Code, just open a notebook — `.venv` + `.env` are picked up automatically.

---

## 11. Project structure

```
main.py                     end-to-end pipeline: WRDS → base parquets → signals/RMT/ML → backtests

src/dispersion/
  data/                     WRDS access: wrds_client, universe, iv, returns, spots, assemble
  signal/                   implied correlation, premium & tradeable signal (implied_corr.py)
  backtest/                 marking.py (surface interpolation) + engine.py (DispersionEngine)
  rmt/                      Marchenko-Pastur + Laloux cleaning (cleaning.py), daily pipeline (daily.py)
  ml/                       features.py, regime.py (GMM/HMM), metamodel.py, experiments.py
  utils/                    greeks.py — BS / Black-76 + relative-greek conversions

tests/                      pytest suite: greeks, marking, engine, RMT, regime causality (39 tests)

exploration_notebooks/      01-11, numbered in pipeline order: WRDS exploration → implied
                            correlation → premiums/costs → backtest → RMT spectrum → ML
                            feature selection/results → robustness/MFIV

results/
  figures/                  13 PNGs referenced throughout this README
  tables/                   10 CSVs backing the headline numbers in §14

data/                       raw + processed (git-ignored — WRDS-licensed data)
config/                     reserved (parameters currently live as module function defaults)
```

**Workflow:** explore in `exploration_notebooks/`, promote validated logic into `src/`. The final
backtest runs from the `.py` modules (reproducibility).

---

## 12. Status

**Week 1 (data) — complete.** WRDS connection; table exploration; design parameters and backtest
period (1996–2024) frozen; full data pipeline (`get_universe`, `get_iv`, `get_returns`,
`realized_vol`, `realized_corr_matrix`, `average_correlation`, `build_dataset`); the six aligned
`.parquet` deliverables built over 1996–2024; and a 6-dimension adversarial QA passed (§9bis).

**Week 2 (theory & signal) — complete.** DMV (2009) read in full; all core derivations reworked
(eq. 2; eq. 3 via Itô/Girsanov; eq. 4–5 Carr–Madan; eq. 8–11 hedge structure). Decisions frozen:
**ATM 91-day IVs** for $\rho_{\text{implied}}$ (§5); **two realised series** — trailing (signal) and
forward window-matched (premium validation) (§8). DMV benchmarks and conventions recorded (§8bis);
downstream plan amended (unconditional baseline first, relative greeks + unit test, RMT on 252-day
windows, post-2003 premium-compression test, Newey–West). **Empirical validation done (notebook 03):**
$\rho_{\text{implied}} \in [0.12, 0.91]$, zero bound violations and zero NaN over all 7,221 days;
window-matched premium $\bar\Pi = +0.079$ (t-NW(63) = 7.4, positive on 75% of days) and signal
$\bar S = +0.078$ (t = 10.6) — **the CRP exists on 1996–2024**. Subperiod preview: 2008–12 = 0.135 vs
**2020–24 = 0.028 (t = 1.19, not significant)** — a marked recent compression, feeding the Week-5
limits-to-arbitrage analysis. Chart: `results/figures/fig_crp_validation.png`. Promoted to
`src/dispersion/signal/implied_corr.py` (reproduces the notebook exactly) and `signal.parquet`
written. The week closed with a **four-axis adversarial audit** (data, signal, mathematics, doc
consistency): zero critical findings, documentation corrections applied, latent guards recorded in
§9bis.

**Week 3 (baseline backtest) — in progress.** Step 1 done (notebook 04): `vsurfd` carries
`impl_premium`/`impl_strike` on our frozen grid, fully populated since 1996; naive BS(q=0) misprices
the standardised premiums by ±5% while the SPX implied dividend yield backed out of `impl_premium` is
leg-consistent (1.04%). **Fork decided (§8bis):** prices from `impl_premium` (re-extraction), greeks
analytic (BS/Black-76 from IV + forward via `fwdprd`/put-call parity + `zerocd`). Step 2 done:
`dispersion.utils.greeks` + 10-test pytest suite (§8bis item 6). Marking fork decided: **daily
mark-to-surface** (§8bis). Step 3 done (8 Jul): the **bundled dataset rebuild** — `surface.parquet`
(4.9M rows, pillars 10/30/60/91d, zero NaN), `spots.parquet` (724k `secprd` closes + `cfadj`;
169/329 names have mid-history splits — settlement prices share the strikes' convention),
`rates.parquet`, `returns.parquet` (329 permnos from 1995), §9bis hardening and tail fix;
non-regression: prior parquets reproduced identically, signal moved on 139 days by ≤8e-3 (documented
policy now correctly applied), headline stats unchanged. Step 4 partly done: engine design frozen
(§7.3) and `backtest/marking.py` implemented (total-variance interpolation, `RateCurve`,
split-adjusted strikes) — full pytest suite 17/17. Step 5 done: `DispersionEngine`
(`backtest/engine.py`, +5 closed-form tests, 22/22) and the **unconditional v0 executed** over 115
quarters: **the DMV guards hold** — Σy = 99.5% (theirs +101.12%), gross Sharpe 0.77 (theirs 0.73),
70% positive quarters, skew −1.29, worst quarters = Q1-2018 Volmageddon (−91%), Q3-2024 (−41%),
summer 2002 — gross maxDD −95.8%, the central motivation for v1/ML tail-cutting. A key convention was
pinned down: DMV's vega-neutrality is to **proportional** vol shocks (ATM straddle ν ≈ 1/σ ⇒ Σy ≈ 1
— the only convention reproducing their +101.12%). The Q1-2018 quarter was dissected via the daily
ledger: the −91% is the **index gamma-hedge bleed** (−154% of wealth on real SPX closes; both option
legs ended positive) — the realised-correlation loss channel, not a marking artefact. **v1**
(ex-ante median gate, 58/115 quarters): better premium per traded quarter (+7.9% vs +7.3%), dodges
2024/2002/Lehman, switches the strategy off post-2020 (the compression at work), but trades Q1-2018 —
the spread measures premium richness, not danger → the quantified case for Week-4 ML regime timing.
**Net of the §8bis cost grid: DMV Table V replicates on our 29 years** — v0 Sharpe 0.77 → 0.42
(theirs 0.73 → 0.41), return ÷1.8, cumulative ×165 → ×2.5 (~3%/yr net, marginal); v1 dies net (×1.1)
as its traded quarters cluster in the wide-spread era. Deliverables shipped (notebook 06):
`results/figures/fig_backtest_s3.png` (log-NAV, four variants, crises annotated),
`results/tables/table_s3_metrics.csv` (v0 net t-stat = 2.24 — the premium survives costs,
marginally) and `table_s3_dmv_sanity.csv`; P&L attribution from the daily ledger — calm quarters
earn on the short index leg (+10.9%) while **crisis quarters lose through the delta-hedge (−23.0%,
the realised-variance/gamma channel)**. **Week 3 complete** (Jul 8 — 11 days ahead of schedule).

**Week 4 (RMT + ML) — complete (17 Jul).** RMT cleaning pipeline (`rmt/`, MP + Laloux, 252-day
windows) with an adversarial look-ahead audit and 3 HMM causality tests passed. **v1_rmt** (gate on
the RMT-cleaned signal) is the week's real result: net Sharpe **0.56**, skew **+0.80**, maxDD
**−58.5%** (vs v0 −99%), dodging all three crashes — though an ablation attributes it to the 252-day
window, not the clipping. The **ML layer is a clean null**: a purged walk-forward meta-model has no
predictive power over the trade's quarterly return, spectral features do not beat VIX, and the veto
does not improve v1_rmt — while the unsupervised regime detector still lights up in every crisis
(*detecting stress ≠ predicting the loss*). Full results, figures and the thesis-writing guide in
**§14**; nine figures + five tables in `results/`. Test suite 37/37.

**Week 5 (robustness + write-up) — complete.** Subperiod analysis (Newey–West), cost/threshold
sensitivity, the parsimonious leg and the MFIV bias check all done — see §14.4. All source and
notebooks were then translated to English, given a full end-to-end reproducibility pass
(`main.py`, 39/39 tests), and the codebase comments were rewritten in a plain, human tone.

---

## 13. References (thesis reading list)

**Foundational — the economics of the trade.**
- Driessen, Maenhout & Vilkov (2009), *The Price of Correlation Risk: Evidence from Equity Options*,
  Journal of Finance 64(3). The founding reference: implied-correlation construction (eq. 2), the
  index-vs-individual variance-risk-premium identification (eq. 3), MFIV (eq. 4–5, Carr–Madan), the
  equicorrelation factor structure (eq. 6), the triangular vega→delta hedge (eq. 8–11), and the
  Table II/V benchmarks we replicate (§8bis).
- Carr & Madan (1998/2001), model-free implied variance (the MFIV integral, §5).
- Bakshi & Kapadia (2003); Bakshi, Kapadia & Madan (2003) — variance risk premium background.

**Random Matrix Theory (Week 4).**
- Bun, Bouchaud & Potters (2017), *Cleaning large correlation matrices: tools from Random Matrix
  Theory*, Physics Reports (arXiv:1610.08104). The pivot reference for the cleaning pipeline.
- Laloux, Cizeau, Bouchaud & Potters (1999), *Noise dressing of the financial correlation matrix*,
  PRL. The effective-edge correction $\lambda_+^{\text{eff}}=(1-\lambda_1/N)(1+\sqrt{q})^2$ (§8bis).
- Marchenko & Pastur (1967), the eigenvalue-density law.
- Potters & Bouchaud (2020), *A First Course in Random Matrix Theory* (textbook).
- Kritzman, Li, Page & Rigobon (2011), *Principal Components as a Measure of Systemic Risk* — the
  absorption-ratio feature; Kritzman & Li (2010), *Skulls, Financial Turbulence…* — the Mahalanobis
  turbulence feature.

**Machine learning & backtest hygiene (Week 4).**
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 (purged K-fold + embargo),
  ch. 3 (meta-labelling), ch. 8 (feature importance). The validation discipline throughout the ML
  layer; the meta-labelling veto architecture (§8bis).
- Rabiner (1989), the HMM tutorial (forward filtering vs smoothing — our causal-filter choice).

**Data.** WRDS: OptionMetrics IvyDB (`vsurfd`, `opprcd`, `zerocd`, `secprd`, `cboe`); CRSP (`dsf`,
`dsp500list`); the CRSP–OptionMetrics linking table (`opcrsphist`).

---

## 14. Results consolidation & thesis-writing guide

Everything needed to draft the 3,000-word paper, in one place. Numbers are net of the §8bis cost
grid unless stated; the full table is `results/tables/table_ml_metrics.csv`.

### 14.1 The four headline results (the spine of the paper)

1. **The correlation risk premium exists** (Week 2). Implied correlation exceeds subsequently-realised
   correlation by **+0.079 on average** (window-matched, t-NW(63) = 7.4, positive on 75% of the 7,221
   days), 1996–2024. Chart `fig_crp_validation.png`. Subperiod: strong 1996–2012, **compressed to
   +0.028 (t = 1.2, insignificant) in 2020–2024** — a falsifiable limits-to-arbitrage prediction
   confirmed.
2. **The premium is real but frictions nearly kill it** (Week 3). The unconditional dispersion trade
   (v0) reproduces DMV Table II out-of-the-box: gross Sharpe **0.77** (theirs 0.73), component weights
   Σy = **+99.5%** (theirs +101.12%), skew −1.29. Net of costs, DMV Table V replicates on our 29
   years: Sharpe **0.42** (theirs 0.41), ~3%/yr, cumulative ×2.5. **But gross maxDD −95.8%** — one
   quarter (Q1-2018 Volmageddon) loses 91%, the delta-hedge bleeding −154% of wealth through the
   realised-variance/gamma channel (P&L attribution, `table_s3_metrics.csv`). Chart
   `fig_backtest_s3.png`.
3. **RMT's value is the 252-day estimator, not the clipping** (Week 4). Gating the trade on the signal
   built from the RMT-cleaned 252-day correlation (v1_rmt) transforms the risk profile: net Sharpe
   **0.56**, **skew +0.80** (from negative), **maxDD −58.5%** (from −99%), cumulative ×12.4 — it dodges
   all three crashes including Volmageddon. **An ablation is decisive and honest: the raw (un-clipped)
   252-day window does the same (Sharpe 0.78 gross, 2/115 gate disagreements)** — the value is the slow
   estimation window, the Laloux clipping adds ~+0.02. The MP spectrum with the Laloux-corrected edge
   (K = 7 sector+market factors vs 4 naïvely) is `fig_mp_spectrum.png`. **Parsimonious leg — RMT's
   frictions answer (Week 5, `fig_parsimonious.png`):** trading only the **top-20** constituent names
   (by cap) instead of all 100 cuts the cost from 3.68% to 3.36% per traded quarter (~9%, the wide
   small-cap spreads are avoided) and **beats the full basket net of costs** (net Sharpe 0.60 vs 0.57,
   skew +1.14 vs +1.00) — although *gross* it is slightly worse (0.75 vs 0.78, a small
   dispersion-replication error), so parsimony pays **net but not gross**, exactly the friction-saving
   mechanism. Honest caveat: the effect is **non-monotonic** (n=30, 50 fall below the full basket), so
   the K=20 optimum is partly noise on ~40 traded quarters — the robust claim is “~15–20 names do at
   least as well as 100, net, at lower cost,” not that 20 is uniquely optimal.
4. **The ML timing layer does not add value — a clean null** (Week 4). A purged walk-forward
   meta-model (predict the trade's quarterly return, veto the bad tail) has **essentially no predictive
   power** (corr(ŷ, y): VIX +0.02, VIX+spectral −0.14, Full −0.20) and **does not improve** v1_rmt (net
   0.56 → 0.57, noise; a VIX-only veto slightly hurts, 0.51). **The central test answer: spectral
   features do NOT beat VIX** for predicting trade outcomes — because neither predicts. Yet the
   unsupervised regime detector (`fig_regime_timeline.png`) lights up cleanly in every crisis: the deep
   finding is **detecting a stress regime ≠ predicting the trade will lose** (stress is often exactly
   when the premium is richest). Charts `fig_ml_nav.png`, `fig_central_test.png`,
   `fig_feature_importance.png`, `fig_pred_scatter.png`, `fig_return_dist.png`.
5. **Two improvement levers (`ml/experiments.py`) — one sharp scientific finding, one modest gain.**
   *(a) The target was the problem, not the features.* Predicting the DAILY forward correlation
   **spike** ($\bar\rho_{t+63}-\bar\rho_t$, ~7,000 obs) instead of the quarterly trade return lifts
   out-of-sample predictability from ≈0 to **corr(ŝ, realised spike) = +0.40** — the spike *is*
   predictable (correlation is persistent). **Yet vetoing on it does not improve the strategy**
   (spike-gate net Sharpe 0.53 < 0.56): even a predictable danger signal doesn't help, because the
   danger is *priced* — a deep confirmation that correlation risk is efficiently compensated
   (stress≠loss, now quantified). *(b) The unsupervised regime detector used directly as the gate*
   (no supervised layer) *modestly trims the tail*: net skew **+1.00** (from +0.80), maxDD **−52.4%**
   (from −58.5%), same Sharpe; gross skew +1.51, maxDD −44.8%. It is the cleanest overlay (no
   overfitting) and the only ML-adjacent lever that improves the risk profile — a candidate to adopt.
   Outputs `backtest_regonly_*`, `backtest_spike_*`, `ml_levers_summary.json`.

### 14.1bis Recommended final strategy (Week 5, adopted)

**`v1_rmt + regime`** — short a correlation basket only when (i) the RMT-cleaned 252-day signal is
above its ex-ante median (v1_rmt primary gate) **and** (ii) the unsupervised causal regime detector
is not in its high (dangerous) tail (ex-ante 67th-percentile veto). This is the cleanest strategy that
improves the *risk profile* without a supervised, overfitting-prone layer:

| Strategy | Net Sharpe | Net skew | Net maxDD | Gross Sharpe | Gross skew |
|---|---|---|---|---|---|
| v0 (unconditional) | 0.42 | −1.33 | −99.0% | 0.77 | −1.29 |
| v1_rmt | 0.56 | +0.80 | −58.5% | 0.80 | +1.28 |
| **v1_rmt + regime** | **0.57** | **+1.00** | **−52.4%** | **0.78** | **+1.51** |

Honest caveat: the return improvement is within noise (Sharpe 0.56→0.57); the genuine gain is
tail-shape (skew, drawdown), which is exactly the stated objective (cut the tails, not the mean). The
tail improvement survives every veto-quantile choice tested (§14.4). The supervised ML timing layer
is **not** part of the recommended strategy — it added nothing (§14.1 result 4).

### 14.2 Suggested thesis structure → where each piece lives

| Thesis section | Source material |
|---|---|
| Intro / client spec | §1 economic rationale; result 1 |
| Theory | §5 (implied corr), §8bis (DMV eq. 2–11 derivations), §8 (premium/signal); references §13 |
| Data & methodology | §2–§4 (WRDS, universe, IV), §6 (realised side), §6.4/§7.3 (cleaning, engine design), §8bis (RMT pipeline, cost calibration); the ML protocol (walk-forward, purge 63d + embargo 21d, label choice) lives in the `ml/` module docstrings and §14.1 results 4–5 |
| Results | §14.1; all figures/tables in `results/` |
| Limitations | §9, §9bis, §14.3 |
| Conclusion | result 3+4: RMT estimation helps, supervised timing doesn't — limits-to-arbitrage + hard-timing |

### 14.3 Honest limitations (state them explicitly in the paper)

- **Full-cap weights** (not float-adjusted; IWF access denied), top-100 renormalisation, **ATM proxy**
  for implied vol (not MFIV) — all documented biases, mostly conservative (§3.4, §5, §7.3).
- **Costs are a calibrated parametric grid** on quoted (not effective) spreads — conservative, but a
  model, not per-trade quotes (§8bis).
- **ML on a tiny sample** (~90 quarterly predictions, ~12 crises): the null result is honest but
  low-powered. Switching to a daily label makes the target predictable (corr +0.40, §14.1 result 5a),
  which is why the null is best read as *the quarterly trade return is unpredictable at this sample*,
  not as a defect of the features.
- **RMT scalar effect is small** by construction (the clipping preserves the market mode that
  dominates ρ̄); most of the value comes through the 252-day estimation window (the ablation, §14.1
  result 3). The parsimonious leg confirms where RMT pays off against frictions — it improves the
  strategy *net* of costs but not *gross* (§14.1 result 3).
- **One provider gap** (OptionMetrics Aug-2020, 17 days) accepted, not fabricated (§9bis).

### 14.4 Robustness — results (Week 5, notebook 10)

**Premium by subperiod (Newey–West).** The CRP is strongly significant early and compresses to
insignificance post-2020 — the limits-to-arbitrage prediction, quantified:

| Period | Mean premium | t-NW(63) | % days > 0 |
|---|---|---|---|
| 1996–2003 | 0.085 | 4.87 | 73% |
| 2004–2007 | 0.078 | 4.06 | 81% |
| 2008–2012 | 0.135 | 4.25 | 83% |
| 2013–2019 | 0.069 | 3.29 | 75% |
| **2020–2024** | **0.028** | **1.19** | 63% |

**Strategy Sharpe by subperiod — the honest nuance (`fig_subperiods.png`).** The RMT gate's Sharpe
advantage is **regime-dependent, not uniform**: it shines in 2004–2007 (v1_rmt 1.01 vs v0 0.67) but
**v0 wins in 2008–2012 (0.39 vs 0.03) and 2013–2019 (0.62 vs 0.45)** — in the premium-rich crisis era,
always-trading did fine on Sharpe and gating merely cut trades. The gate's genuine, robust benefit is
**tail-shape** (skew, drawdown over the full sample), driven by avoiding specific crashes (2018, 2024),
**not** a uniform per-era Sharpe lift. State this plainly in the paper.

**Cost and threshold sensitivity of `v1_rmt+regime` (`table_sensitivity.csv`).** Robust, not
knife-edge: net Sharpe **0.46–0.67** across cost ±50% (still above v0's 0.42 even at ×1.5), **0.57–0.58**
across the regime veto-quantile (0.50/0.67/0.80 — a tighter veto lifts skew to +1.31), and **0.54–0.56**
across the v1_rmt gate quantile (0.40/0.60). The tail improvement survives every threshold choice.

**Vanna/Volga — the convexity risk of the frozen book (`fig_vanna_volga.png`).** Closed-form Volga
($\partial\text{vega}/\partial\sigma$) and Vanna ($\partial\text{vega}/\partial S$) are added to
`utils.greeks` (bump-tested). The point: at entry an ATM straddle sits near the second-order-neutral
strike ($d_2=0$), where **both Volga and Vanna vanish** — so the DMV proportional-vega neutrality is
clean *at inception*. But it is a **first-order** property: as positions are held frozen for the
quarter and the spot drifts while implied vol moves, the straddles leave that point, Volga and Vanna
grow (sharpening as they age — the 28-day curve), and the vega-neutrality **erodes**. Empirically the
v0 book's total vega drifts by **~3,600** over Q1-2018 (Volmageddon) — the frozen neutrality unwinding
precisely in the correlation spike. This is the accepted convexity risk noted at design time (§7.3);
the daily delta-hedge removes only the first-order directional risk, not this second-order channel,
which is part of why crisis quarters bleed (§14.1 result 2).

**MFIV robustness — the ATM proxy is conservative, quantified (`fig_mfiv_bias.png`).** We rebuild the
**model-free implied variance** on the index leg from the full SPX delta grid (17 call + 17 put
pillars per date, VIX-style Carr–Madan replication) over the 116 rebalances. The index MFIV exceeds
ATM by **+1.2 vol points on average** (19.7% vs 18.5% — the steep put skew), on **100% of dates**.
Propagated through the inversion, substituting MFIV for ATM on the index raises $\rho_{\text{implied}}$
by **Δρ ≈ +0.066 on average (always positive)**, i.e. from 0.43 to ~0.50. So the ATM-based
$\rho_{\text{implied}}$ — and hence the headline premium — is a **conservative lower bound**; with MFIV
the correlation-risk premium would be *larger*, strengthening the thesis. *Honest nuance:* this is the
**index-leg-only** effect (an upper bound on the true bias) — applying MFIV to the flatter single-name
smiles too would partially offset it, but the net stays positive because the index skew is much
steeper than single-name skews (exactly DMV's mechanism). Table `table_mfiv_bias.csv`.

### 14.5 Possible extensions

The robustness programme in §14.4 is complete. One natural extension was left out of scope by design:
a **per-name delta hedge** (hedging each constituent straddle with its own underlying, rather than
hedging only the net index exposure). It would sharpen the isolation of the correlation P&L from
single-name direction, at the cost of much heavier trading and a larger cost drag — a trade-off the
parsimonious-leg result (§14.1, result 3) already speaks to. It is a clean next step, not a gap in the
current results.
