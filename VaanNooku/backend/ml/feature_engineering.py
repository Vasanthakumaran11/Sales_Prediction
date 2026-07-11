"""
RetailAI Feature Engineering Module
=====================================
CRITICAL: This exact module must be used for BOTH:
  1. Training the models (batch mode on historical CSV)
  2. Live prediction in FastAPI (inference mode on new user data)

If these two paths compute features differently, predictions will be
wrong even if the models are perfect. This is the #1 cause of "works
in notebook, fails in production" bugs.

Usage:
    # Training (batch):
    df = pd.read_csv("retailai_finalized_dataset.csv")
    df_features = engineer_features_batch(df)

    # Live inference (single new transaction + recent history from DB):
    row_features = engineer_features_live(new_transaction, history_df, encoders)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ============================================================
# CONSTANTS — must match generate_finalized_dataset.py exactly
# ============================================================
STOCK_BUFFER = {'Small': 1.15, 'Medium': 1.35, 'Supermarket': 1.65}
COLD_START_SCHEDULE = [(1, 0.40), (3, 0.70), (float('inf'), 1.00)]

CATEGORICAL_COLS = ['Store_Type', 'Location_Type', 'Category', 'Day_Type', 'Demand_Level', 'Supplier_ID', 'Item_Name']

FEATURE_COLUMNS = [
    'Day', 'Month', 'Day_of_Week', 'Is_Weekend', 'Is_Festival', 'Is_Salary_Day',
    'Store_Age_Months', 'Lead_Time_Days', 'Unit_Price', 'Discount',
    'Sell_Through_Ratio', 'Stock_Remaining_Ratio',
    'Lag_1', 'Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7',
    'Store_Type_enc', 'Location_Type_enc', 'Category_enc', 'Day_Type_enc',
    'Supplier_ID_enc', 'Item_Name_enc'
]

CAT_FEATURE_INDICES = [FEATURE_COLUMNS.index(c) for c in
                        ['Store_Type_enc', 'Location_Type_enc', 'Category_enc',
                         'Day_Type_enc', 'Supplier_ID_enc', 'Item_Name_enc']]


# ============================================================
# ENCODER MANAGEMENT — fit once on training data, reuse forever
# ============================================================
def fit_and_save_encoders(train_df: pd.DataFrame, save_dir: str = "encoders"):
    """
    Run ONCE during initial model training.
    Saves LabelEncoders so live inference uses the EXACT same category mapping.
    """
    os.makedirs(save_dir, exist_ok=True)
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        encoders[col] = le
        joblib.dump(le, f"{save_dir}/{col}_encoder.pkl")
    return encoders


def load_encoders(save_dir: str = "encoders"):
    """Load saved encoders for inference. Call once at FastAPI startup."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        encoders[col] = joblib.load(f"{save_dir}/{col}_encoder.pkl")
    return encoders


def _safe_encode(series: pd.Series, encoder: LabelEncoder) -> pd.Series:
    """
    Encode values, falling back to the first known class for anything
    unseen at training time (e.g. a brand-new Supplier_ID a user adds).
    NEVER let an unseen category crash the pipeline in production.
    """
    known = set(encoder.classes_)
    safe_vals = series.astype(str).apply(lambda v: v if v in known else encoder.classes_[0])
    return encoder.transform(safe_vals)


# ============================================================
# COLD-START FACTOR (shared logic — used in features AND business logic)
# ============================================================
def cold_start_factor(store_age_months: float) -> float:
    for threshold, factor in COLD_START_SCHEDULE:
        if store_age_months <= threshold:
            return factor
    return 1.00


# ============================================================
# BATCH MODE — for training on the full historical CSV
# ============================================================
def engineer_features_batch(df: pd.DataFrame, encoders: dict = None,
                              fit_encoders: bool = False,
                              save_dir: str = "encoders") -> tuple:
    """
    Computes Lag/Rolling features chronologically per Store x Item group
    (NO shuffling — this prevents data leakage).

    Returns: (df_with_features, encoders_used)
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store_ID', 'Item_ID', 'Date']).reset_index(drop=True)

    # --- Business ratios ---
    df['Sell_Through_Ratio'] = (df['Units_Sold'] / df['Units_Stocked']).round(4)
    df['Stock_Remaining_Ratio'] = (df['Units_Remaining'] / df['Units_Stocked']).round(4)

    # --- Lag / Rolling (shift(1) BEFORE rolling = no leakage from current day) ---
    grp = df.groupby(['Store_ID', 'Item_ID'])['Units_Sold']
    df['Lag_1'] = grp.shift(1)
    df['Lag_7'] = grp.shift(7)
    df['Rolling_Mean_7'] = grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df['Rolling_Std_7'] = grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())

    fallback = df['Units_Sold'].median()
    df['Lag_1'] = df['Lag_1'].fillna(fallback)
    df['Lag_7'] = df['Lag_7'].fillna(fallback)
    df['Rolling_Mean_7'] = df['Rolling_Mean_7'].fillna(fallback)
    df['Rolling_Std_7'] = df['Rolling_Std_7'].fillna(0)

    # --- Categorical encoding ---
    if fit_encoders:
        encoders = fit_and_save_encoders(df, save_dir)
    elif encoders is None:
        encoders = load_encoders(save_dir)

    for col in CATEGORICAL_COLS:
        df[col + '_enc'] = _safe_encode(df[col], encoders[col])

    return df, encoders


# ============================================================
# LIVE MODE — for a single new transaction from a store owner
# ============================================================
def engineer_features_live(new_row: dict, history_df: pd.DataFrame,
                             encoders: dict) -> pd.DataFrame:
    """
    new_row: dict with the MINIMUM fields the user actually enters
        {
          'Date', 'Store_ID', 'Store_Type', 'Location_Type', 'Store_Age_Months',
          'Item_ID', 'Category', 'Supplier_ID', 'Lead_Time_Days',
          'Units_Stocked', 'Units_Sold', 'Unit_Price', 'Discount',
          'Day_of_Week', 'Is_Weekend', 'Is_Festival', 'Is_Salary_Day', 'Day_Type'
        }
    history_df: last 7+ days of this Store_ID + Item_ID pulled from PostgreSQL
                (must contain 'Date' and 'Units_Sold' at minimum)
    encoders: loaded via load_encoders() ONCE at FastAPI app startup

    Returns a single-row DataFrame with FEATURE_COLUMNS ready for model.predict()
    """
    row = new_row.copy()

    row['Units_Remaining'] = row['Units_Stocked'] - row['Units_Sold']
    row['Sell_Through_Ratio'] = round(row['Units_Sold'] / row['Units_Stocked'], 4) if row['Units_Stocked'] else 0
    row['Stock_Remaining_Ratio'] = round(row['Units_Remaining'] / row['Units_Stocked'], 4) if row['Units_Stocked'] else 0

    # --- Pull lag/rolling from actual DB history (NOT from the new row itself) ---
    hist = history_df.sort_values('Date')
    if len(hist) >= 1:
        row['Lag_1'] = hist['Units_Sold'].iloc[-1]
    else:
        row['Lag_1'] = row['Units_Sold']  # true cold start, no history yet

    if len(hist) >= 7:
        row['Lag_7'] = hist['Units_Sold'].iloc[-7]
    else:
        row['Lag_7'] = row['Lag_1']

    recent7 = hist['Units_Sold'].tail(7)
    row['Rolling_Mean_7'] = recent7.mean() if len(recent7) > 0 else row['Units_Sold']
    row['Rolling_Std_7'] = recent7.std() if len(recent7) > 1 else 0.0
    if pd.isna(row['Rolling_Std_7']):
        row['Rolling_Std_7'] = 0.0

    df_row = pd.DataFrame([row])
    for col in CATEGORICAL_COLS:
        if col in df_row.columns:
            df_row[col + '_enc'] = _safe_encode(df_row[col], encoders[col])

    for c in FEATURE_COLUMNS:
        if c not in df_row.columns:
            raise ValueError(f"Missing required field for prediction: {c}")

    return df_row[FEATURE_COLUMNS]


# ============================================================
# INVENTORY / BUSINESS LOGIC (applied AFTER ensemble prediction)
# ============================================================
def compute_business_metrics(predicted_demand: float, current_stock: float,
                               lead_time_days: int, demand_std: float,
                               store_type: str, unit_cost: float,
                               ordering_cost: float = 50.0,
                               holding_cost_rate: float = 0.20,
                               service_level_z: float = 1.65) -> dict:
    """
    predicted_demand : daily demand forecast from ensemble
    demand_std       : Rolling_Std_7 (or better: std over longer validated window)
    unit_cost        : Cost_Price of the item
    ordering_cost    : fixed cost per purchase order (₹)
    holding_cost_rate: annual holding cost as fraction of unit cost (typ. 15-25%)
    """
    lead_time_weeks = lead_time_days / 7
    safety_stock = service_level_z * demand_std * np.sqrt(lead_time_weeks) if demand_std > 0 else 0
    reorder_point = predicted_demand * lead_time_days + safety_stock

    annual_demand = predicted_demand * 365
    holding_cost = unit_cost * holding_cost_rate
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost) if holding_cost > 0 else 0

    if current_stock < predicted_demand:
        risk = "HIGH"
    elif current_stock < reorder_point:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "safety_stock": round(safety_stock, 1),
        "reorder_point": round(reorder_point, 1),
        "economic_order_quantity": round(eoq, 1),
        "risk_level": risk,
        "recommended_order_qty": round(max(0, reorder_point + eoq - current_stock), 1)
    }