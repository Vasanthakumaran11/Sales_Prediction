from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import HTTPException
from src.models import models
import pandas as pd

# ML predictor — only imported when needed for next-month forecast
def _get_predictor():
    from ml.predictor import predict_next_month
    return predict_next_month


def _get_product_history(db: Session, store_id: str, item_id: str):
    """Last 90 days of daily transaction history for this store + product.
    Returns None (not an exception) when there's no history, so callers can
    decide whether to 400 (single-item forecast) or skip (bulk aggregate)."""
    history = (
        db.query(
            models.TransactionDaily.date,
            func.sum(models.TransactionItem.qty).label("units_sold")
        )
        .join(models.TransactionItem, models.TransactionDaily.id == models.TransactionItem.transaction_id)
        .join(models.Product, models.TransactionItem.product_id == models.Product.id)
        .filter(
            models.TransactionDaily.store_id == store_id,
            models.Product.id == item_id
        )
        .group_by(models.TransactionDaily.date)
        .order_by(models.TransactionDaily.date.desc())
        .limit(90)
        .all()
    )
    if len(history) < 1:
        return None
    return pd.DataFrame([{"Date": h.date, "Units_Sold": h.units_sold or 0} for h in history])


def _build_static_info(store: models.Store, product: models.Product, current_stock: float, history_df: pd.DataFrame) -> dict:
    """Maps a Store + Product DB row pair onto the categorical/static feature
    values the ML ensemble was trained on. Shared by both the single-item and
    bulk-aggregate forecast paths so this mapping logic never diverges."""
    # Map friendly frontend category names to model-trained categorical values
    db_cat = product.category or "Non-Perishable"
    ml_category = "Perishable" if db_cat.lower() in ["dairy & bakery", "perishable", "milk", "curd", "paneer", "bread", "eggs"] else "Non-Perishable"

    # Map supplier ID to trained categories (SUP_FRESH, SUP_STAPLE, SUP_PACKAGED)
    raw_sup = (product.supplier_id or "SUP_STAPLE").upper()
    ml_supplier = (
        "SUP_FRESH" if raw_sup in ["SUP_FRESH", "BALAJI-AGRO"] or ml_category == "Perishable"
        else "SUP_STAPLE" if raw_sup in ["SUP_STAPLE", "BALAJI-STAPLE"]
        else "SUP_PACKAGED"
    )

    # Lead times based on dataset defaults per item ID
    lead_time_days = 2
    if product.id in ["ITM01", "ITM02", "ITM04"]:
        lead_time_days = 1
    elif product.id in ["ITM06", "ITM11", "ITM17", "ITM18"]:
        lead_time_days = 5
    elif product.id == "ITM08":
        lead_time_days = 6
    elif product.id == "ITM12":
        lead_time_days = 7

    return {
        "Store_Type":            store.type or "Supermarket",
        "Location_Type":         store.location or "Urban",
        "Store_Age_Months":      store.months_active or 12,
        "Category":              ml_category,
        "Supplier_ID":           ml_supplier,
        "Lead_Time_Days":        lead_time_days,
        "Unit_Price":            product.selling_price,
        "Cost_Price":            product.cost_price,
        "Sell_Through_Ratio":    min(1.0, history_df["Units_Sold"].mean() / max(1, current_stock)),
        "Stock_Remaining_Ratio": current_stock / max(1, current_stock),
        "Units_Remaining":       current_stock,
    }


def get_stock_summary(db: Session, store_id: str):
    """Query product count and stock counts grouped by category."""
    results = db.query(
        models.Product.category,
        func.count(models.Product.id).label("skuCount"),
        func.sum(models.StockLevel.current_stock).label("totalStock")
    ).join(
        models.StockLevel, models.Product.id == models.StockLevel.product_id
    ).filter(
        models.StockLevel.store_id == store_id
    ).group_by(
        models.Product.category
    ).all()

    return [
        {
            "category": row.category,
            "skuCount": row.skuCount,
            "totalStock": row.totalStock or 0
        }
        for row in results
    ]


def get_forecast(db: Session, store_id: str):
    """
    Aggregate next-month forecast for every product in the store, built by
    running the real ML ensemble per-product (same model as the single-item
    endpoint below) and summing the results server-side. Replaces the old
    investment-heuristic placeholder so the summary tiles, weekly trend, and
    per-SKU table are always mathematically consistent with each other.
    """
    store = db.query(models.Store).filter(models.Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    rows = db.query(models.Product, models.StockLevel).join(
        models.StockLevel, models.Product.id == models.StockLevel.product_id
    ).filter(models.StockLevel.store_id == store_id).all()

    predict_next_month = _get_predictor()

    product_breakdown = []
    total_revenue = 0.0
    total_units = 0.0
    daily_unit_totals = [0.0] * 30

    for product, stock in rows:
        history_df = _get_product_history(db, store_id, product.id)
        if history_df is None:
            continue  # skip products with no sales history yet, don't fail the whole request

        static_info = _build_static_info(store, product, stock.current_stock, history_df)
        result = predict_next_month(store_id, product.id, static_info, history_df)

        total_revenue += result["monthly_revenue"]
        total_units += result["monthly_total_units"]
        for i, day in enumerate(result["daily_forecast"][:30]):
            daily_unit_totals[i] += day["predicted_units"]

        product_breakdown.append({
            "id": product.id,
            "name": product.name,
            "sku": product.sku,
            "category": product.category,
            "currentStock": stock.current_stock,
            "projectedDemand": result["monthly_total_units"],
            "projectedRevenue": result["monthly_revenue"],
            "reorderPoint": result["business_metrics"]["reorder_point"],
            "recommendedOrder": result["business_metrics"]["recommended_order_qty"],
        })

    if not product_breakdown:
        raise HTTPException(
            status_code=400,
            detail="No sales history for any product in this store — cannot forecast yet. Please add daily sales data first."
        )

    # Revenue isn't tracked per-day in predict_next_month's output, so weekly
    # revenue is derived from the real weekly unit share using the store's
    # actual average revenue-per-unit across this forecast — not a fabricated split.
    avg_revenue_per_unit = (total_revenue / total_units) if total_units > 0 else 0.0
    week_bounds = [(0, 7), (7, 14), (14, 21), (21, 30)]
    weekly_breakdown = [
        {
            "week": f"Week {idx + 1}",
            "projectedSales": round(sum(daily_unit_totals[start:end]) * avg_revenue_per_unit, 2)
        }
        for idx, (start, end) in enumerate(week_bounds)
    ]

    from ml.model_loader import get_ensemble_r2

    return {
        "summary": {
            "projectedRevenue": round(total_revenue, 2),
            "projectedUnitsDemand": round(total_units, 1),
            "confidenceMetricR2": get_ensemble_r2()
        },
        "weeklyBreakdown": weekly_breakdown,
        "productBreakdown": product_breakdown
    }


def get_next_month_prediction(db: Session, store_id: str, item_id: str):
    """
    ML ensemble next-month demand forecast for a given store + product.
    Pulls last 90 days from TransactionDaily table.
    """
    store = db.query(models.Store).filter(models.Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail=f"Store '{store_id}' not found.")

    product = db.query(models.Product).filter(models.Product.id == item_id).first()
    if not product:
        raise HTTPException(status_code=404, detail=f"Product '{item_id}' not found.")

    history_df = _get_product_history(db, store_id, item_id)
    if history_df is None:
        raise HTTPException(
            status_code=400,
            detail="No sales history for this store/item — cannot forecast yet. Please add daily sales data first."
        )

    stock = db.query(models.StockLevel).filter(
        models.StockLevel.store_id == store_id,
        models.StockLevel.product_id == item_id
    ).first()
    current_stock = stock.current_stock if stock else 0

    static_info = _build_static_info(store, product, current_stock, history_df)

    predict_next_month = _get_predictor()
    result = predict_next_month(store_id, item_id, static_info, history_df)
    return result
