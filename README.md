# Dispersion Trading Project

**ML-Driven Dispersion Trading Strategy** — MSc Financial Engineering Applied Project, Imperial College London.

## Objective

This project exploits the systematic overpricing of S&P 500 implied correlation relative to
realised correlation through a delta/vega-neutral dispersion strategy. The strategy is
enhanced with Random Matrix Theory (RMT) filtering of the correlation matrix and
machine-learning regime prediction.

## Data

Data is sourced from **WRDS** (Wharton Research Data Services), available to Imperial College
students:
- **OptionMetrics (Ivy DB US)** — historical implied volatility surfaces for the S&P 500 index and its constituents
- **CRSP** — point-in-time index constituents, weights, adjusted prices

Raw data is **never committed** to this repository (WRDS licensing). Anyone reproducing this
work must use their own WRDS credentials (see Setup below).

## Setup

### 1. Prerequisites
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.12 (uv will install it automatically if missing)
- A valid WRDS account with access to OptionMetrics and CRSP

### 2. Install dependencies
```bash
uv sync
source .venv/bin/activate
```

### 3. Configure access & imports (`.env`)
Copy the example environment file:
```bash
cp .env.example .env
```
Then edit `.env`:
- `WRDS_USERNAME` — your WRDS username.
- `PYTHONPATH` — **absolute** path to this repo's `src/` (quote it if it contains spaces).
  This makes the `dispersion` package importable.

Your **password is never stored in this project**. On the first connection, WRDS will prompt
for it in the terminal and offer to create a secure `.pgpass` file (outside this repo) for
future logins.

> **Why PYTHONPATH instead of an editable install?** Under macOS, when the repo lives in a
> TCC-protected location (e.g. `~/Desktop`), the `.pth` file written by an editable install is
> tagged with a `com.apple.provenance` xattr that Python's startup `site`/`io.open_code` refuses
> to read — silently breaking `import dispersion`. Setting `PYTHONPATH` (read from the
> environment, not a file) sidesteps this. VS Code reads `.env` automatically (see
> `.vscode/settings.json`); in the terminal, pass `--env-file .env` (uv does **not** auto-load it).

### 4. Run anything (scripts, notebooks)
Always pass `--env-file .env` so `PYTHONPATH` is set:
```bash
# test the WRDS connection (checks crsp / optionm / comp access)
uv run --env-file .env python -m dispersion.data.wrds_client

# run a notebook headless
cd notebooks && uv run --env-file ../.env jupyter nbconvert --to notebook --execute --inplace 01_explore_vsurfd.ipynb
```
In VS Code, just open a notebook — the `.venv` interpreter + `.env` are picked up automatically.

## Project structure

```
src/dispersion/     core code
  data/             WRDS access, extraction, cleaning
  signal/           implied correlation, dispersion spread
  backtest/         strategy simulation engine
  rmt/              Random Matrix Theory filtering
  ml/               predictive models (later phase)
  utils/            shared helpers
notebooks/          exploratory notebooks
config/             project settings
data/               raw + processed data (git-ignored)
results/            generated figures and tables
```

## Workflow

Exploration happens in `notebooks/`; validated logic is promoted into clean, reusable
functions under `src/dispersion/`. The final backtest runs from the `.py` modules, not the
notebooks, for reproducibility.

## Status

Work in progress. See `plan.md` for the full roadmap.

**Current stage:** infrastructure complete (environment, repo, WRDS connection module).
Awaiting WRDS account validation to begin data extraction.