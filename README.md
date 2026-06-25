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

### 3. Configure WRDS access
Copy the example environment file and add your own WRDS username:
```bash
cp .env.example .env
```
Then open `.env` and replace the placeholder with your WRDS username:
```
WRDS_USERNAME=your_wrds_username
```
Your **password is never stored in this project**. On the first connection, WRDS will prompt
for it in the terminal and offer to create a secure `.pgpass` file (outside this repo) for
future logins.

### 4. Test the connection
```bash
python -m src.dispersion.data.wrds_client
```
This checks that your account can reach the `crsp`, `optionm`, and `comp` libraries.

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