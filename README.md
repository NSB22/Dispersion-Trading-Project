# Dispersion Trading Project

**ML-Driven Dispersion Trading Strategy** — MSc Financial Engineering Applied Project, Imperial College London.

This document is both the technical README and the **methodological reference** for the thesis:
every design choice is justified, with the underlying mathematics, in the sections below.

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

**(b) Random Matrix Theory needs a meaningful $N$.** Marchenko–Pastur separates noise from signal in
an $N\times N$ correlation matrix via the ratio $q = T/N$ ($T$ = observations). With a 63-day window:
$N=30 \Rightarrow q\approx 2.1$ (almost no noise bulk to clean — RMT pointless); $N=100 \Rightarrow
q\approx 0.63$ (a genuine noise band to filter — RMT informative).

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
- **Weights computed only at the ~60 rebalancing dates** (15 yrs × 4), not daily. Positions are then
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
weights. (This is the CBOE-style implied-correlation construction.)

**Documented limitation:** the one-factor (single $\bar\rho$) assumption collapses the full pairwise
correlation structure into one number — assumed and stated explicitly. The RMT layer (§8) revisits
the full matrix. Validity check to apply: $\rho_{\text{implied}} \in [0,1]$.

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
from too few points is unreliable, and this also protects the RMT ($q=T/N$) in §8.

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
(rank $\le T-1=62$, so ~38 zero eigenvalues). One eigenvalue dominates (~75% of total variance) — the
**market factor**. This $q=T/N\approx0.63<1$ regime (Marchenko–Pastur has a point mass at zero) is
exactly where eigenvalue clipping (§8) adds value, confirming the $N=100$ choice.

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

The dataset is fully reproducible from `data/raw` (here, directly from WRDS) by re-running
`build_dataset`.

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

---

## 8. The dispersion signal

The raw signal is the **dispersion spread**

$$S_t \;=\; \rho_{\text{implied},t} \;-\; \rho_{\text{realised},t},$$

expected to be **positive on average** (the CRP). Entry/exit rules threshold $S_t$ (Week 3). The
**RMT layer** (Week 4) cleans the realised correlation matrix — superimposing the Marchenko–Pastur
density to separate the noise bulk from genuine signal eigenvalues, eigenvalue-clipping, then
re-injecting the cleaned matrix and re-backtesting vs the baseline. An optional **ML layer** (Week 4,
bonus) predicts the spread (walk-forward, purging/embargo to avoid leakage).

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
| Value plausibility (29y) | **PASS** | ρ̄ ∈ [0.06, 0.83]; **component IV > index IV on all 7 221 days** (mean ratio 1.60); crisis peaks correctly dated |
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
- **Realised-vol tail NaN before delistings** (~1 455 obs, 0.23%): a name's returns dry up a few days
  before it leaves the universe, so it keeps its frozen weight but has no realised vol for those last
  days (minor effect on the $w_i\sigma_i$-weighted $\bar\rho$ at the very end of a holding interval).

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
cd notebooks && uv run --env-file ../.env jupyter nbconvert --to notebook --execute --inplace 01_explore_vsurfd.ipynb
```
In VS Code, just open a notebook — `.venv` + `.env` are picked up automatically.

---

## 11. Project structure

```
src/dispersion/
  data/        WRDS access (wrds_client), universe, iv, returns
  signal/      implied correlation, dispersion spread        (Week 2)
  backtest/    strategy simulation engine                    (Week 3)
  rmt/         Random Matrix Theory filtering                (Week 4)
  ml/          predictive models                             (Week 4, bonus)
  utils/       shared helpers
notebooks/     exploration (01_explore_vsurfd, 02_explore_crsp)
config/        project settings
data/          raw + processed (git-ignored)
results/       generated figures and tables
```

**Workflow:** explore in `notebooks/`, promote validated logic into `src/`. The final backtest runs
from the `.py` modules (reproducibility).

---

## 12. Status

**Week 1 (data) — complete.** WRDS connection; table exploration; design parameters and backtest
period (1996–2024) frozen; full data pipeline (`get_universe`, `get_iv`, `get_returns`,
`realized_vol`, `realized_corr_matrix`, `average_correlation`, `build_dataset`); the six aligned
`.parquet` deliverables built over 1996–2024; and a 6-dimension adversarial QA passed (§9bis).

**Next — Week 2 (signal):** implied correlation via the one-factor formula (§5), the dispersion
spread $\rho_{\text{implied}}-\rho_{\text{realised}}$, and the empirical validation chart (implied
above realised on average). See `plan.md` for the full roadmap.
