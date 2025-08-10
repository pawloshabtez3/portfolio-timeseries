#!/usr/bin/env python3
# scripts/fetch_data.py

import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = ["TSLA", "BND", "SPY"]
START = "2015-07-01"
# make end the day after to ensure inclusive up to 2025-07-31
END = "2025-08-01"

OUTDIR = Path("../data/raw").resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

def fetch_and_save(ticker):
    print(f"Fetching {ticker} ...")
    df = yf.download(ticker, start=START, end=END, progress=False)
    if df.empty:
        raise RuntimeError(f"No data fetched for {ticker}. Check internet or ticker.")
    df.index.name = "Date"
    # save full DF
    df.to_csv(OUTDIR / f"{ticker}.csv", index=True)
    print(f"Saved {OUTDIR / f'{ticker}.csv'} (rows={len(df)})")

if __name__ == "__main__":
    for t in TICKERS:
        fetch_and_save(t)
    print("Done.")
