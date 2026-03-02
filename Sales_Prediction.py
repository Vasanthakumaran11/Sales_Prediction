import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Preprocessing
# ==========================================

def load_and_preprocess(file_path, store_id=1):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df[df['Store'] == store_id].sort_values('Date')
    df.ffill(inplace=True)
    return df

# ==========================================
# 2. Feature Engineering
# ==========================================

def create_features(df):
    df_feat = df[['Date', 'Weekly_Sales']].copy()

    df_feat['Lag1'] = df_feat['Weekly_Sales'].shift(1)
    df_feat['Lag2'] = df_feat['Weekly_Sales'].shift(2)

    df_feat['Rolling_Mean_4'] = df_feat['Weekly_Sales'].shift(1).rolling(4).mean()
    df_feat['Rolling_Std_4'] = df_feat['Weekly_Sales'].shift(1).rolling(4).std()

    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['Year'] = df_feat['Date'].dt.year

    df_feat.dropna(inplace=True)
    return df_feat

# ==========================================
# 3A. ARIMA
# ==========================================

def run_arima(train_series, test_series):
    model = ARIMA(train_series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_series))
    mae = mean_absolute_error(test_series, forecast)
    return model_fit, forecast, mae

# ==========================================
# 3B. Random Forest
# ==========================================

def run_rf(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return model, preds, mae, r2

# ==========================================
# 3C. LSTM
# ==========================================

class SalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SalesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def prepare_lstm_data(X, y, lookback=4):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)

def run_lstm(X_train, y_train, X_test, y_test):
    lookback = 4
    X_seq, y_seq = prepare_lstm_data(X_train, y_train, lookback)

    X_seq_t = torch.tensor(X_seq, dtype=torch.float32)
    y_seq_t = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

    model = SalesLSTM(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_seq_t)
        loss = criterion(outputs, y_seq_t)
        loss.backward()
        optimizer.step()

    # Evaluate on test
    X_test_seq, y_test_seq = prepare_lstm_data(X_test, y_test, lookback)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
    preds = model(X_test_t).detach().numpy().flatten()
    mae = mean_absolute_error(y_test_seq, preds)

    return model, mae

# ==========================================
# 4. Recursive Forecast
# ==========================================

def recursive_forecast(model, last_features, n_weeks=4):
    forecasts = []
    current = last_features.copy()

    for _ in range(n_weeks):
        pred = model.predict(current.reshape(1, -1))[0]
        forecasts.append(pred)

        # Update Lag Features
        lag1 = pred
        lag2 = current[0]

        # Update Rolling Mean/Std
        roll_mean = (lag1 + lag2 + current[0] + current[1]) / 4
        roll_std = np.std([lag1, lag2, current[0], current[1]])

        current[0] = lag1
        current[1] = lag2
        current[2] = roll_mean
        current[3] = roll_std

    return forecasts

# ==========================================
# 5. Inventory Optimization
# ==========================================

def optimize_inventory(predicted_demand, sales_std, lead_time, current_stock):
    z = 1.65
    safety_stock = z * sales_std * np.sqrt(lead_time)
    reorder_qty = max(0, predicted_demand + safety_stock - current_stock)

    if current_stock < safety_stock:
        risk = "HIGH"
    elif current_stock < predicted_demand:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return safety_stock, reorder_qty, risk

# ==========================================
# 6. MAIN PIPELINE
# ==========================================

def run_pipeline(store_id, stock, lead_time):
    df = load_and_preprocess("Walmart.csv", store_id)
    df_feat = create_features(df)

    train = df_feat[:-8]
    test = df_feat[-8:]

    features = ['Lag1','Lag2','Rolling_Mean_4','Rolling_Std_4','Month','Year']

    X_train = train[features].values
    y_train = train['Weekly_Sales'].values
    X_test = test[features].values
    y_test = test['Weekly_Sales'].values

    # ARIMA
    arima_model, arima_forecast, arima_mae = run_arima(train['Weekly_Sales'], test['Weekly_Sales'])

    # RF
    rf_model, rf_preds, rf_mae, rf_r2 = run_rf(X_train, y_train, X_test, y_test)

    # LSTM
    lstm_model, lstm_mae = run_lstm(X_train, y_train, X_test, y_test)

    # Compare
    results = pd.DataFrame({
        "Model":["ARIMA","Random Forest","LSTM"],
        "MAE":[arima_mae, rf_mae, lstm_mae]
    })

    print("\nMODEL COMPARISON")
    print(results)

    best_model_name = results.loc[results['MAE'].idxmin(), 'Model']
    print(f"\nBest Model Selected: {best_model_name}")

    # Use RF for recursive forecast (stable for tabular features)
    last_features = df_feat.iloc[-1][features].values
    future = recursive_forecast(rf_model, last_features, 4)
    total_demand = sum(future)

    print("\nNEXT 4 WEEK FORECAST")
    for i,val in enumerate(future):
        print(f"Week {i+1}: {val:,.2f}")

    print(f"Total 30-Day Demand: {total_demand:,.2f}")

    # Inventory
    sales_std = df_feat['Weekly_Sales'].std()
    safety, reorder, risk = optimize_inventory(total_demand, sales_std, lead_time, stock)

    print("\nINVENTORY DECISION")
    print("Safety Stock:", round(safety,2))
    print("Recommended Reorder:", round(reorder,2))
    print("Risk Level:", risk)

# ==========================================
# USER INPUT
# ==========================================

if __name__ == "__main__":
    try:
        s = int(input("Enter Store ID: ") or 1)
        stock = float(input("Enter Current Stock: ") or 1500000)
        lead = int(input("Enter Lead Time (weeks): ") or 2)
        run_pipeline(s, stock, lead)
    except Exception as e:
        print("Error:", e)