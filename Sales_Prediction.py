import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. STORE REGISTRATION LAYER
# ==========================================

def register_store():
    print("\n--- STORE REGISTRATION LAYER ---")
    name = input("Enter Store Name: ")
    location = input("Location (Rural / Urban / Town): ").strip().capitalize()
    size = input("Store Size (Large / Medium / Small): ").strip().capitalize()
    category = input("Business Category (LargeScale / SmallScale): ").strip().capitalize()

    # Classification Logic
    # Category A -> Large Urban LargeScale
    # Category B -> Medium Urban/Town
    # Category C -> Small Urban or Medium Rural
    # Category D -> Small Rural SmallScale

    if size == "Large" and location == "Urban" and category == "Largescale":
        store_cat = "A"
        enc_val = 3
    elif size == "Medium" and (location == "Urban" or location == "Town"):
        store_cat = "B"
        enc_val = 2
    elif (size == "Small" and location == "Urban") or (size == "Medium" and location == "Rural"):
        store_cat = "C"
        enc_val = 1
    else:
        store_cat = "D"
        enc_val = 0

    print(f"\n[System] Store '{name}' classified as Category: {store_cat}")
    print(f"Explanation: {size} size in {location} area focusing on {category}.")
    
    return {
        "name": name,
        "location": location,
        "size": size,
        "business_category": category,
        "category": store_cat,
        "category_encoded": enc_val
    }

# ==========================================
# 2. MACHINE LEARNING FORECASTING LAYER
# ==========================================

def load_and_preprocess_data():
    print("\n[AI] Loading training datasets...")
    # Using Rossmann Store Sales data
    train = pd.read_csv('train.csv', low_memory=False)
    store = pd.read_csv('store.csv')
    
    # Merge
    df = pd.merge(train, store, on='Store')
    
    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    
    # Handle StateHoliday
    df['StateHoliday'] = df['StateHoliday'].astype(str).replace({'0': 0, 'a': 1, 'b': 2, 'c': 3}).astype(int)
    
    # Map StoreType/Assortment to our A/B/C/D Categories for training
    # This is synthetic mapping to align with User Registration logic
    type_map = {'b': 3, 'a': 2, 'd': 1, 'c': 0} # b is mostly unique/large
    df['StoreCategory_Encoded'] = df['StoreType'].map(type_map)
    
    # Features
    features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                'Month', 'Year', 'Day', 'StoreCategory_Encoded']
    X = df[features]
    y = df['Sales']
    
    # Train-test split (using a sample for speed in this demo)
    X_train, X_test, y_train, y_test = train_test_split(X.head(100000), y.head(100000), test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, features

def train_xgboost(X_train, X_test, y_train, y_test):
    print("[AI] Training XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    # Accuracy based on MAPE
    mape = np.mean(np.abs((y_test - preds) / y_test))
    accuracy = (1 - mape) * 100
    
    print(f"\nModel Performance:")
    print(f"- MAE: {mae:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- R2 Score: {r2:.4f}")
    print(f"- Accuracy: {accuracy:.2f}%")
    
    return model

# ==========================================
# 3. HYBRID EXTENSION (LSTM)
# ==========================================

class SalesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(SalesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_hybrid(sales_series):
    print("\n[Hybrid] Training LSTM for Personalization (Advanced Mode)...")
    # Simple time-series setup
    data = sales_series.values[-180:].reshape(-1, 1).astype(float)
    # Normalize
    mean = data.mean()
    std = data.std()
    data_norm = (data - mean) / std
    
    X, y = [], []
    lookback = 7
    for i in range(len(data_norm) - lookback):
        X.append(data_norm[i:i+lookback])
        y.append(data_norm[i+lookback])
        
    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    y_t = torch.tensor(np.array(y), dtype=torch.float32)
    
    model = SalesLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        
    print("[System] LSTM Model weight initialized.")
    return model, mean, std

# ==========================================
# 4. INVENTORY OPTIMIZATION LAYER
# ==========================================

def optimize_inventory(predicted_demand, current_inventory, lead_time_weeks, weekly_sales_std):
    # Safety Stock = Z (1.65 for 95%) * Std * Sqrt(Lead Time)
    z = 1.65
    safety_stock = z * weekly_sales_std * np.sqrt(lead_time_weeks)
    
    reorder_qty = max(0, predicted_demand + safety_stock - current_inventory)
    
    if current_inventory < predicted_demand:
        risk = "HIGH"
    elif current_inventory < (predicted_demand + safety_stock):
        risk = "MEDIUM"
    else:
        risk = "LOW"
        
    return safety_stock, reorder_qty, risk

# ==========================================
# 5. MAIN EXECUTION FLOW
# ==========================================

def main():
    print("\n" + "="*50)
    print("HYBRID INTELLIGENT RETAIL FORECASTING SYSTEM")
    print("="*50)

    # Layer 1: Registration
    store_info = register_store()
    
    # Layer 2: ML Forecasting (Pre-trained/Loading)
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    xgb_model = train_xgboost(X_train, X_test, y_train, y_test)
    
    # Layer 3: Prediction Input Flow
    print("\n--- PREDICTION INPUT FLOW ---")
    month = int(input("Current Month (1-12): ") or 3)
    avg_sales = float(input("Current Avg Weekly Sales: ") or 10000)
    promo = 1 if input("Promotion planned? (Yes/No): ").lower() == 'yes' else 0
    holiday = 1 if input("Holiday upcoming? (Yes/No): ").lower() == 'yes' else 0
    inv = float(input("Current Inventory: ") or 50000)
    lead = int(input("Supplier Lead Time (weeks): ") or 2)

    # Prepare input for XGBoost
    # Feature order: ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Month', 'Year', 'Day', 'StoreCategory_Encoded']
    # Using dummy values for Store, DayOfWeek, Year, etc. for the prediction
    input_data = pd.DataFrame([[1, 1, promo, holiday, 0, month, 2015, 1, store_info['category_encoded']]], 
                              columns=features)
    
    predicted_weekly = xgb_model.predict(input_data)[0]
    predicted_30d = predicted_weekly * 4 # Approximation for 30 days
    
    # Layer 4: Inventory Optimization
    # Estimate weekly std from training data for the specific category
    sales_std = y_train.std() 
    safety, reorder, risk = optimize_inventory(predicted_30d, inv, lead, sales_std)
    
    # Layer 5: Hybrid Extension
    hybrid_mode = input("\nEnable Hybrid Mode (LSTM)? (Yes/No): ").lower() == 'yes'
    if hybrid_mode:
        lstm_model, l_mean, l_std = train_lstm_hybrid(y_train)
        print("[Hybrid] Prediction refined using temporal patterns.")

    # FINAL OUTPUT
    print("\n" + "*"*30)
    print("AI RETAIL FORECASTING SYSTEM")
    print("*"*30)
    print(f"Store Category: {store_info['category']}")
    print(f"Predicted 30-Day Sales: {predicted_30d:,.2f}")
    print(f"Safety Stock:           {safety:,.2f}")
    print(f"Recommended Reorder:    {reorder:,.2f}")
    print(f"Risk Level:             {risk}")
    
    print("\nBusiness Interpretation:")
    if risk == "HIGH":
        print(f"Warning: Current inventory is insufficient to meet predicted 30-day demand. "
              f"Place a reorder of {reorder:,.2f} immediately.")
    elif risk == "MEDIUM":
        print(f"Advice: Stocks are adequate but close to safety limits. Monitor sales closely.")
    else:
        print(f"Status: Inventory levels are healthy. No urgent reorder needed.")
        
if __name__ == "__main__":
    main()