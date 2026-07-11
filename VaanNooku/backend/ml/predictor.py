"""
Prediction logic: single-day ensemble prediction + recursive next-month forecast.
"""
import pandas as pd
from ml.model_loader import get_models
from ml.feature_engineering import (
    FEATURE_COLUMNS, CATEGORICAL_COLS, _safe_encode,
    engineer_features_live, compute_business_metrics
)

def predict_next_day(new_transaction: dict, history_df: pd.DataFrame) -> dict:
    models, weights, encoders = get_models()
    feature_row = engineer_features_live(new_transaction, history_df, encoders)

    preds = {name: float(m.predict(feature_row)[0]) for name, m in models.items()}
    ensemble_pred = sum(weights[name] * preds[name] for name in preds)

    return {
        "predicted_units": round(max(0, ensemble_pred), 2),
        "individual_predictions": {k: round(v, 2) for k, v in preds.items()}
    }


def predict_next_month(store_id: str, item_id: str, static_info: dict,
                         history_df: pd.DataFrame, forecast_days: int = 30) -> dict:
    """
    Recursive multi-step forecast. static_info must contain:
    Store_Type, Location_Type, Category, Supplier_ID, Store_Age_Months,
    Lead_Time_Days, Unit_Price, Cost_Price, Sell_Through_Ratio,
    Stock_Remaining_Ratio, Units_Remaining
    """
    models, weights, encoders = get_models()
    rolling_history = history_df[["Date", "Units_Sold"]].copy()
    rolling_history["Date"] = pd.to_datetime(rolling_history["Date"])
    last_date = rolling_history["Date"].max()

    forecast_rows = []
    for step in range(forecast_days):
        forecast_date = last_date + pd.Timedelta(days=step + 1)
        lag_1 = rolling_history["Units_Sold"].iloc[-1]
        lag_7 = rolling_history["Units_Sold"].iloc[-7] if len(rolling_history) >= 7 else lag_1
        recent7 = rolling_history["Units_Sold"].tail(7)
        rolling_mean_7 = recent7.mean()
        rolling_std_7 = recent7.std() if len(recent7) > 1 else 0.0
        if pd.isna(rolling_std_7):
            rolling_std_7 = 0.0

        is_weekend = 1 if forecast_date.dayofweek >= 5 else 0
        is_salary_day = 1 if forecast_date.day in (5, 20) else 0

        row = {
            "Day": forecast_date.day, "Month": forecast_date.month,
            "Day_of_Week": forecast_date.dayofweek, "Is_Weekend": is_weekend,
            "Is_Festival": 0, "Is_Salary_Day": is_salary_day,
            "Store_Age_Months": static_info["Store_Age_Months"],
            "Lead_Time_Days": static_info["Lead_Time_Days"],
            "Unit_Price": static_info["Unit_Price"], "Discount": 0.0,
            "Sell_Through_Ratio": static_info["Sell_Through_Ratio"],
            "Stock_Remaining_Ratio": static_info["Stock_Remaining_Ratio"],
            "Lag_1": lag_1, "Lag_7": lag_7,
            "Rolling_Mean_7": rolling_mean_7, "Rolling_Std_7": rolling_std_7,
            "Store_Type": static_info["Store_Type"], "Location_Type": static_info["Location_Type"],
            "Category": static_info["Category"], "Day_Type": "Weekend" if is_weekend else "Weekday",
            "Supplier_ID": static_info["Supplier_ID"],
        }
        row_df = pd.DataFrame([row])
        for col in CATEGORICAL_COLS:
            row_df[col + "_enc"] = _safe_encode(row_df[col], encoders[col]) if col in row_df.columns else 0

        feature_row = row_df[FEATURE_COLUMNS]
        preds = {name: float(m.predict(feature_row)[0]) for name, m in models.items()}
        predicted_units = max(0, sum(weights[name] * preds[name] for name in preds))

        forecast_rows.append({"date": forecast_date.strftime("%Y-%m-%d"),
                               "predicted_units": round(predicted_units, 2)})
        rolling_history = pd.concat([rolling_history,
            pd.DataFrame([{"Date": forecast_date, "Units_Sold": predicted_units}])], ignore_index=True)

    monthly_total = sum(r["predicted_units"] for r in forecast_rows)
    avg_daily = monthly_total / forecast_days

    biz = compute_business_metrics(
        predicted_demand=avg_daily,
        current_stock=static_info["Units_Remaining"],
        lead_time_days=int(static_info["Lead_Time_Days"]),
        demand_std=pd.Series([r["predicted_units"] for r in forecast_rows]).std(),
        store_type=static_info["Store_Type"],
        unit_cost=static_info["Cost_Price"],
    )

    return {
        "store_id": store_id, "item_id": item_id,
        "daily_forecast": forecast_rows,
        "monthly_total_units": round(monthly_total, 1),
        "monthly_revenue": round(monthly_total * static_info["Unit_Price"], 2),
        "avg_daily_units": round(avg_daily, 2),
        "business_metrics": biz,
    }