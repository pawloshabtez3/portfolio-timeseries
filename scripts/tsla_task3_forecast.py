# If running in Colab and you want to save this as a file, put at top of cell:
# %%writefile tsla_task3_forecast.py

import os
import sys
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# USER CONFIG
# ---------------------------
CSV_PATH = "TSLA.csv"          # path to your CSV (uploaded in Colab or in VS Code folder)
FORECAST_MONTHS = 12          # change to 6 or 12 (6-12 months requested)
TRADING_DAYS_PER_MONTH = 21   # approximate trading days in one month
ARIMA_P_RANGE = range(0,4)    # try p=0..3
ARIMA_D_RANGE = range(0,3)    # try d=0..2
ARIMA_Q_RANGE = range(0,4)    # try q=0..3
LSTM_RUN = True               # Set False to skip LSTM (TensorFlow required if True)
LSTM_SIMULATIONS = 150        # Monte Carlo runs to produce LSTM intervals (if LSTM_RUN True)
LOOK_BACK = 60                # LSTM lookback window (days)
LSTM_EPOCHS = 30
LSTM_BATCH = 32
OUT_PREFIX = "tsla_future_forecast"
# ---------------------------

def load_and_clean(path):
    # handle your weird CSV where first row is 'Ticker' and second row has headers like "Date"
    # read raw, detect if first column header is "Price" and second row contains "Date"
    raw = pd.read_csv(path, header=None, dtype=str)
    # If standard CSV (has header with 'Date'), just read normally:
    first_row = raw.iloc[0].tolist()
    if 'Date' in first_row or 'date' in [c.lower() for c in first_row]:
        df = pd.read_csv(path)
    else:
        # skip the first line (row 0) and use second line as header
        df = pd.read_csv(path, skiprows=1)
    # ensure a 'Date' column exists (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    if 'date' in cols_lower:
        if cols_lower['date'] != 'Date':
            df = df.rename(columns={cols_lower['date']: 'Date'})
    else:
        raise ValueError("Could not find a Date column in CSV. Columns: {}".format(df.columns.tolist()))
    # Convert date and numeric columns
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Some CSVs have the price column named 'Close' (we'll also keep 'Price' if available)
    price_col = None
    for cand in ['Close','close','Price','price','Adj Close','Adj Close*']:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        # try case-insensitive
        for c in df.columns:
            if c.lower() in ['close','price','adj close']:
                price_col = c
                break
    if price_col is None:
        raise ValueError("Could not find a price column (Close/Price). Columns: {}".format(df.columns.tolist()))
    if price_col != 'Close':
        df = df.rename(columns={price_col: 'Close'})
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df

def chronological_split(df, cutoff_date_str="2023-12-31"):
    try:
        cutoff = pd.to_datetime(cutoff_date_str)
        train = df[df['Date'] <= cutoff].copy()
        test = df[df['Date'] > cutoff].copy()
        # if no rows after cutoff, fallback to 80/20 chronological split
        if len(test) < 10:
            cut = int(len(df) * 0.8)
            train = df.iloc[:cut].copy()
            test = df.iloc[cut:].copy()
    except Exception:
        cut = int(len(df) * 0.8)
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:].copy()
    return train, test

def arima_grid_search(train_series, p_range, d_range, q_range):
    best_aic = np.inf
    best_order = None
    best_res = None
    # keep things fast: we use limited ranges
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    m = ARIMA(train_series, order=(p,d,q))
                    r = m.fit(method_kwargs={"warn_convergence": False})
                    if r.aic < best_aic:
                        best_aic = r.aic
                        best_order = (p,d,q)
                        best_res = r
                    print(f"Tested ARIMA{(p,d,q)} AIC={r.aic:.2f}")
                except Exception:
                    continue
    return best_order, best_aic, best_res

def arima_forecast_with_ci(fitted_model, steps, alpha=0.05):
    # statsmodels' get_forecast produces mean and conf_int
    fc = fitted_model.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    return mean, ci

# Simple LSTM implementation with Monte Carlo for intervals
def train_lstm_and_simulate(train_series, horizon_steps, look_back=60, epochs=30, batch_size=32, sims=100):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except Exception as e:
        print("TensorFlow not available or failed to import:", e)
        return None

    arr = train_series.values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(arr).flatten()
    # create sequences
    def create_seq(data, look_back):
        X,y = [],[]
        for i in range(len(data)-look_back):
            X.append(data[i:i+look_back])
            y.append(data[i+look_back])
        return np.array(X), np.array(y)
    X, y = create_seq(scaled, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # small model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back,1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    # prepare seed
    seed = scaled[-look_back:].tolist()
    # compute residual std in scaled domain
    preds_train = model.predict(X, verbose=0).flatten()
    resid = y - preds_train
    resid_std = np.std(resid)
    print("LSTM residual std (scaled):", resid_std)
    # Monte Carlo sims: iteratively predict adding gaussian noise in scaled space
    sims_preds = []
    for s in range(sims):
        seq = seed.copy()
        out = []
        for i in range(horizon_steps):
            x = np.array(seq[-look_back:]).reshape((1, look_back, 1))
            yhat = model.predict(x, verbose=0)[0][0]
            # add noise in scaled space
            noisy = yhat + np.random.normal(0, resid_std)
            out.append(noisy)
            seq.append(noisy)
        sims_preds.append(out)
    sims_preds = np.array(sims_preds)  # shape (sims, horizon)
    # invert scale -> real prices
    sims_preds_real = scaler.inverse_transform(sims_preds.reshape(-1,1)).reshape(sims_preds.shape)
    # point forecast = mean across sims (per horizon step)
    mean_forecast = sims_preds_real.mean(axis=0)
    lower = np.percentile(sims_preds_real, 2.5, axis=0)
    upper = np.percentile(sims_preds_real, 97.5, axis=0)
    return mean_forecast, lower, upper, model

# Put everything together
def forecast_future(df, forecast_months=12, arima_ranges=(ARIMA_P_RANGE, ARIMA_D_RANGE, ARIMA_Q_RANGE),
                    do_lstm=True, lstm_sims=100, look_back=LOOK_BACK, lstm_epochs=LSTM_EPOCHS, lstm_batch=LSTM_BATCH):
    # split data
    train_df, test_df = chronological_split(df)
    train_series = train_df.set_index('Date')['Close']
    test_series = test_df.set_index('Date')['Close'] if len(test_df)>0 else None
    print(f"Train has {len(train_series)} rows; Test has {len(test_df) if test_df is not None else 0} rows.")
    # fit ARIMA grid search on train_series
    p_range, d_range, q_range = arima_ranges
    print("Running ARIMA grid search (may take a minute)...")
    best_order, best_aic, best_res = arima_grid_search(train_series, p_range, d_range, q_range)
    print("Best ARIMA order:", best_order, "AIC:", best_aic)
    # forecast horizon in trading days
    horizon = int(round(forecast_months * TRADING_DAYS_PER_MONTH))
    # ARIMA forecast with CI
    # Use the fitted best_res if available; if not, fit with best_order
    if best_res is None and best_order is not None:
        arima_model = ARIMA(train_series, order=best_order).fit()
        best_res = arima_model
    arima_mean, arima_ci = arima_forecast_with_ci(best_res, steps=horizon)
    # build future index of business days starting after last date
    last_date = df['Date'].max()
    future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    arima_mean.index = future_index
    arima_ci.index = future_index

    # LSTM forecasts via Monte Carlo (optional)
    if do_lstm:
        print("Training LSTM and running Monte Carlo simulations for intervals...")
        try:
            lstm_mean, lstm_lower, lstm_upper, lstm_model = train_lstm_and_simulate(train_series, horizon, look_back=look_back, epochs=lstm_epochs, batch_size=lstm_batch, sims=lstm_sims)
            lstm_index = future_index
        except Exception as e:
            print("LSTM failed:", e)
            lstm_mean = lstm_lower = lstm_upper = None
            lstm_index = None
    else:
        lstm_mean = lstm_lower = lstm_upper = None
        lstm_index = None

    # Save results into DataFrame
    out = pd.DataFrame(index=future_index)
    out['ARIMA_Mean'] = arima_mean.values
    out['ARIMA_Lower'] = arima_ci.iloc[:,0].values
    out['ARIMA_Upper'] = arima_ci.iloc[:,1].values
    if lstm_mean is not None:
        out['LSTM_Mean'] = lstm_mean
        out['LSTM_Lower'] = lstm_lower
        out['LSTM_Upper'] = lstm_upper

    return out, best_order, best_aic, train_series, test_series

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("CSV not found at", CSV_PATH)
        sys.exit(1)
    df = load_and_clean(CSV_PATH)
    print("Data loaded. Date range:", df['Date'].min().date(), "to", df['Date'].max().date(), "| rows:", len(df))

    results_df, best_order, best_aic, train_ser, test_ser = forecast_future(
        df,
        forecast_months=FORECAST_MONTHS,
        arima_ranges=(ARIMA_P_RANGE, ARIMA_D_RANGE, ARIMA_Q_RANGE),
        do_lstm=LSTM_RUN,
        lstm_sims=LSTM_SIMULATIONS,
        look_back=LOOK_BACK,
        lstm_epochs=LSTM_EPOCHS,
        lstm_batch=LSTM_BATCH
    )

    # Save results and plot
    csv_out = OUT_PREFIX + ".csv"
    results_df.to_csv(csv_out)
    print("Saved forecast CSV:", csv_out)

    # Plot historical + forecast
    plt.figure(figsize=(14,6))
    plt.plot(train_ser.index, train_ser.values, label="Historical (train)", color='black', linewidth=1)
    if test_ser is not None and len(test_ser)>0:
        plt.plot(test_ser.index, test_ser.values, label="Historical (test)", color='gray', linewidth=1)
    # ARIMA mean and CI
    plt.plot(results_df.index, results_df['ARIMA_Mean'], label="ARIMA Mean", linestyle='-', linewidth=1.5)
    plt.fill_between(results_df.index, results_df['ARIMA_Lower'], results_df['ARIMA_Upper'], alpha=0.2, label="ARIMA 95% CI")
    # LSTM if present
    if 'LSTM_Mean' in results_df.columns:
        plt.plot(results_df.index, results_df['LSTM_Mean'], label="LSTM Mean", linestyle='--', linewidth=1.2)
        plt.fill_between(results_df.index, results_df['LSTM_Lower'], results_df['LSTM_Upper'], alpha=0.15, label="LSTM 95% Empirical Interval")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"TSLA Forecast for next {FORECAST_MONTHS} months")
    plt.legend()
    plt.tight_layout()
    plot_path = OUT_PREFIX + "_plot.png"
    plt.savefig(plot_path)
    print("Saved plot:", plot_path)
    plt.show()

    # Basic interpretation auto-summary
    # Trend detection on ARIMA mean
    arima_mean = results_df['ARIMA_Mean'].values
    first, last = arima_mean[0], arima_mean[-1]
    pct_change = (last - first)/first * 100
    print("\n--- Automatic Summary ---")
    print(f"ARIMA best order: {best_order} (AIC={best_aic:.2f})")
    print(f"ARIMA forecast start: {results_df.index[0].date()}, end: {results_df.index[-1].date()}")
    print(f"ARIMA mean change over horizon: {pct_change:.2f}% (start {first:.2f} -> end {last:.2f})")
    # CI width behaviour
    ci_widths = (results_df['ARIMA_Upper'] - results_df['ARIMA_Lower']).values
    width_start, width_end = ci_widths[0], ci_widths[-1]
    print(f"ARIMA 95% CI width at start: {width_start:.2f}, at end: {width_end:.2f}")
    if width_end > width_start * 1.5:
        print("Confidence intervals widen notably across the horizon — uncertainty increases for longer-term forecasts.")
    elif width_end < width_start * 1.1:
        print("CI width remains similar — model implies relatively stable uncertainty across horizon.")
    else:
        print("CI width increases moderately across horizon.")

    if 'LSTM_Mean' in results_df.columns:
        lm = results_df['LSTM_Mean'].values
        l_first, l_last = lm[0], lm[-1]
        l_pct = (l_last - l_first)/l_first * 100
        print(f"\nLSTM mean change over horizon: {l_pct:.2f}% (start {l_first:.2f} -> end {l_last:.2f})")
        # compare ARIMA vs LSTM means
        diff_pct = (np.mean(lm) - np.mean(arima_mean)) / np.mean(arima_mean) * 100
        print(f"Average LSTM mean is {diff_pct:.2f}% different vs ARIMA mean over the horizon.")

    # Save a short text summary
    summary_text = []
    summary_text.append(f"Forecast months: {FORECAST_MONTHS}")
    summary_text.append(f"ARIMA order: {best_order} AIC: {best_aic:.2f}")
    summary_text.append(f"ARIMA pct change (start->end): {pct_change:.2f}%")
    summary_text.append(f"ARIMA CI width start/end: {width_start:.2f}/{width_end:.2f}")
    if 'LSTM_Mean' in results_df.columns:
        summary_text.append(f"LSTM pct change (start->end): {l_pct:.2f}%")
    with open(OUT_PREFIX + "_summary.txt","w") as f:
        f.write("\n".join(summary_text))
    print("Saved summary:", OUT_PREFIX + "_summary.txt")
