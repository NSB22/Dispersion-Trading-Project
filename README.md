# Dispersion Trading Project

**ML-Driven Dispersion Trading Strategy** — MSc Financial Engineering Applied Project, Imperial College London.

## Objective

Exploit the systematic overpricing of S&P 500 implied correlation relative to realised
correlation through a delta/vega-neutral dispersion strategy, enhanced with Random Matrix
Theory filtering and machine-learning regime prediction.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
source .venv/bin/activate
```

Data is sourced from WRDS (OptionMetrics + CRSP). Credentials go in a local `.env` file
(see `.env.example`); data itself is never committed (WRDS licence).

## Project structure
