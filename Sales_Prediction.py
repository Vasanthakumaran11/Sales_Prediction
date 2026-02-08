# ===================== IMPORTS =====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# ===================== CONSTANTS =====================
CURRENT_YEAR = 2025
DATA_PATH = "SuperKart.csv"

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH)

# ===================== DATA CLEANING & MAPPING =====================
# Fill missing weights
df['Product_Weight'] = df['Product_Weight'].fillna(df['Product_Weight'].median())

# Map Store_Type to user-specified categories (normalized to lowercase)
store_type_map = {
    'Supermarket Type1': 'supermarket',
    'Supermarket Type2': 'supermarket',
    'Departmental Store': 'departmentstore',
    'Food Mart': 'food mart'
}
df['Store_Type'] = df['Store_Type'].map(store_type_map)

# Map Store_Location_City_Type to user-specified categories (normalized to lowercase)
location_map = {
    'Tier 1': 'tier1',
    'Tier 2': 'tier2',
    'Tier 3': 'tier3'
}
df['Store_Location_City_Type'] = df['Store_Location_City_Type'].map(location_map)

# Rename column for convenience
df.rename(columns={'Store_Location_City_Type': 'store_location_type', 'Store_Type': 'store_type', 'Store_Establishment_Year': 'Store_year'}, inplace=True)

# ===================== FEATURE SELECTION =====================
# Only keep the features requested by the user
features = ['Product_Weight', 'Product_Type', 'Store_year', 'store_location_type', 'store_type']
target = 'Product_Store_Sales_Total'

df_model = df[features + [target]]

# Calculate Store_Age internally for better prediction performance (still based on Store_year)
df_model['Store_Age'] = CURRENT_YEAR - df_model['Store_year']

# ===================== ENCODING =====================
df_encoded = pd.get_dummies(df_model)

# ===================== SPLIT =====================
X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

train_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== MODEL =====================
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ===================== EVALUATION =====================
y_pred = model.predict(X_test)
print("\n===== MODEL PERFORMANCE (Updated Features) =====")
print("R2 Score :", r2_score(y_test, y_pred))
print("MAE      :", mean_absolute_error(y_test, y_pred))

# ===================== USER INPUT =====================
print("\n===== ENTER PRODUCT & STORE DETAILS =====")

product_weight = float(input("Product Weight (kg): "))
product_type = input(f"Product Type (options: {', '.join(df['Product_Type'].unique()[:5])}...): ")
store_year = int(input("Store Year (Establishment Year): "))
location_type = input("Store Location Type (tier1, tier2, tier3): ").strip().lower()
st_type = input("Store Type (supermarket, departmentstore, food mart): ").strip().lower()

# ===================== BUILD INPUT DATAFRAME =====================
new_data = {
    'Product_Weight': product_weight,
    'Product_Type': product_type,
    'Store_year': store_year,
    'Store_Age': CURRENT_YEAR - store_year,
    'store_location_type': location_type,
    'store_type': st_type
}

new_df = pd.DataFrame([new_data])

# ===================== ENCODING =====================
new_df = pd.get_dummies(new_df)

# Align columns with training data
new_df = new_df.reindex(columns=train_columns, fill_value=0)

# ===================== PREDICTION =====================
predicted_sales = model.predict(new_df)

print("\n===== PREDICTION RESULT =====")
print(f"Predicted Product Store Sales: {predicted_sales[0]:.2f}")

