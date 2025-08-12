import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def load_and_clean(file_path):
    # Skip the first row, use second row as header
    df = pd.read_csv(file_path, skiprows=1)
    # Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # Sort by date
    df.sort_values('Date', inplace=True)
    # Keep only Date and Close price
    df = df[['Date', 'Close']].copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(inplace=True)
    return df

def train_arima(train, test):
    # Find best ARIMA params
    model = auto_arima(train, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
    order = model.order
    arima_model = ARIMA(train, order=order)
    arima_model_fit = arima_model.fit()
    forecast = arima_model_fit.forecast(steps=len(test))
    return forecast

def train_lstm(train, test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))

    def create_dataset(dataset, look_back=60):
        X, y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i-look_back:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    look_back = 60
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, y_test = create_dataset(np.concatenate((train_scaled[-look_back:], test_scaled)), look_back)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[early_stop])

    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    return predictions.flatten()

def evaluate(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mae, rmse, mape

def main(file_path):
    df = load_and_clean(file_path)

    # Train/test split
    train = df[df['Date'] < '2024-01-01']['Close']
    test = df[df['Date'] >= '2024-01-01']['Close']

    # ARIMA
    arima_pred = train_arima(train, test)
    arima_mae, arima_rmse, arima_mape = evaluate(test, arima_pred)

    # LSTM
    lstm_pred = train_lstm(train, test)
    lstm_mae, lstm_rmse, lstm_mape = evaluate(test, lstm_pred)

    # Save results
    results = pd.DataFrame({
        'Date': df[df['Date'] >= '2024-01-01']['Date'][:len(test)].values,
        'Actual': test.values,
        'ARIMA_Pred': arima_pred,
        'LSTM_Pred': lstm_pred[:len(test)]
    })
    results.to_csv('tsla_forecast.csv', index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['Date'], results['Actual'], label='Actual')
    plt.plot(results['Date'], results['ARIMA_Pred'], label='ARIMA')
    plt.plot(results['Date'], results['LSTM_Pred'], label='LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('TSLA Forecast')
    plt.legend()
    plt.savefig('tsla_forecast_plot.png')

    print("\nModel Performance:")
    print(f"ARIMA  - MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.2f}%")
    print(f"LSTM   - MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}, MAPE: {lstm_mape:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tsla_forecasting_full.py <path_to_csv>")
    else:
        main(sys.argv[1])
