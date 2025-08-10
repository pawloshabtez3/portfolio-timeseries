# Portfolio Time-Series Preprocessing & EDA
Task 1: fetch historical price data for TSLA, SPY, BND (2015-07-01 → 2025-07-31), preprocess, calculate returns, rolling volatility, ADF stationarity tests, VaR and Sharpe metrics, and produce EDA plots.

Usage:
1. Create venv and install dependencies: `pip install -r requirements.txt`
2. Fetch raw data: `python scripts/fetch_data.py`
3. Run preprocessing & EDA: `python scripts/preprocess_and_eda.py`
4. Open notebook for interactive analysis: `jupyter lab notebooks/01_preprocess_and_eda.ipynb`
