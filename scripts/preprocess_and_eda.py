import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def rolling_stats(returns, window=20):
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return rolling_mean, rolling_std

def detect_outliers(returns, threshold=3):
    mean = returns.mean()
    std = returns.std()
    outliers = returns[(returns - mean).abs() > threshold * std]
    return outliers


DATA_DIR = Path("data")
TICKERS = ["TSLA", "BND", "SPY"]

def load_data(ticker):
    df = pd.read_csv(
        DATA_DIR / f"{ticker}.csv",
        skiprows=2,
        names=["Date", "Open", "High", "Low", "Close", "Volume"],  # rename columns explicitly
        parse_dates=["Date"],
        index_col="Date"
    )
    return df


def compute_daily_returns(df):
    return df['Close'].pct_change().dropna()

def historical_var(returns, confidence_level=0.05):
    return returns.quantile(confidence_level)

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / 252  # daily risk-free rate approx
    return excess_returns.mean() / excess_returns.std() * (252 ** 0.5)



def plot_price(df, ticker):
    df['Close'].plot(title=f"{ticker} Adjusted Close Price")
    plt.show()

def plot_daily_returns(returns, ticker):
    returns.plot(title=f"{ticker} Daily Returns")
    plt.show()

def adf_test(series, ticker):
    result = adfuller(series)
    print(f"ADF Statistic for {ticker}: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print(f"The series {ticker} is stationary.")
    else:
        print(f"The series {ticker} is non-stationary.")
    print()

def main():
    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")
        df = load_data(ticker)
        plot_price(df, ticker)

        returns = compute_daily_returns(df)
        plot_daily_returns(returns, ticker)

        adf_test(returns, ticker)

        rolling_mean, rolling_std = rolling_stats(returns)
        rolling_mean.plot(title=f"{ticker} Rolling Mean of Returns (20 days)")
        plt.show()
        rolling_std.plot(title=f"{ticker} Rolling Std Dev of Returns (Volatility, 20 days)")
        plt.show()

        outliers = detect_outliers(returns)
        print(f"Outliers detected in {ticker} returns:\n{outliers}\n")

        var_95 = historical_var(returns)
        print(f"{ticker} Historical VaR at 95% confidence level: {var_95:.4f}")

        sr = sharpe_ratio(returns)
        print(f"{ticker} Sharpe Ratio: {sr:.4f}")



if __name__ == "__main__":
    main()

