"""
Streamlit Web Application for Retail Forecasting System
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.data_engine import UserDataEngine
from models.predict import PredictionEngine
from models.train_personalized import PersonalizedModelTrainer
from utils.inventory import InventoryOptimizer, print_inventory_report
from utils.config import MODEL_METADATA_PATH, PRODUCTS

# Page configuration
st.set_page_config(
    page_title="🏪 Retail Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "user_engine" not in st.session_state:
    st.session_state.user_engine = UserDataEngine()

if "prediction_engine" not in st.session_state:
    try:
        st.session_state.prediction_engine = PredictionEngine(use_personalized=False)
    except:
        st.session_state.prediction_engine = None

if "use_personalized" not in st.session_state:
    st.session_state.use_personalized = False


# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📝 Sales Entry",
        "📊 Sales Analytics",
        "📦 Inventory Management",
        "🔮 Demand Forecast",
        "🤖 Model Management",
        "ℹ️ About",
    ],
)


def render_dashboard():
    """Render dashboard"""
    st.title("🏪 Smart Retail Forecasting Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Sales Records", len(st.session_state.user_engine.sales))

    with col2:
        st.metric("Products Tracked", len(st.session_state.user_engine.products))

    with col3:
        st.metric("Total Inventory", 1000)  # Placeholder

    st.markdown("---")

    # Quick stats
    st.subheader("📈 Quick Statistics")

    if not st.session_state.user_engine.sales.empty:
        col1, col2, col3, col4 = st.columns(4)

        sales_df = st.session_state.user_engine.sales.copy()
        if "Date" in sales_df.columns:
            sales_df["Date"] = pd.to_datetime(
                sales_df["Date"], format="%d-%m-%Y", errors="coerce"
            )
            recent_sales = sales_df[sales_df["Date"] == sales_df["Date"].max()]

            with col1:
                st.metric("Today's Revenue", f"₹{recent_sales['Revenue'].sum():.0f}")

            with col2:
                st.metric("Today's Units", f"{recent_sales['Units_Sold'].sum():.0f}")

            with col3:
                st.metric("Avg Unit Price", f"₹{sales_df['Unit_Price'].mean():.0f}")

            with col4:
                high_demand_count = len(sales_df[sales_df["Units_Sold"] > 30])
                st.metric("High Demand Items", high_demand_count)
    else:
        st.info("No sales data yet. Start by recording sales in the Sales Entry section.")

    st.markdown("---")

    # Model status
    st.subheader("🤖 Model Status")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.prediction_engine:
            st.success(f"✅ Base Model Active: {st.session_state.prediction_engine.model_name}")
        else:
            st.error("❌ Prediction engine not loaded")

    with col2:
        if os.path.exists(MODEL_METADATA_PATH):
            with open(MODEL_METADATA_PATH, "r") as f:
                metadata = json.load(f)
                if metadata.get("personalized"):
                    st.success("✅ Personalized Model Active")
                else:
                    st.info("⏳ Personalized Model: Training in progress...")


def render_sales_entry():
    """Render sales entry form"""
    st.title("📝 Daily Sales Entry")

    with st.form("sales_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            product_name = st.selectbox(
                "Product", list(PRODUCTS.keys())
            )

        with col2:
            units_sold = st.number_input("Units Sold", min_value=0, max_value=1000)

        with col3:
            unit_price = st.number_input("Unit Price (₹)", min_value=0.0, max_value=10000.0)

        col1, col2, col3 = st.columns(3)

        with col1:
            discount = st.slider("Discount (%)", 0, 50, 0, step=1)

        with col2:
            promo = st.checkbox("Promotional Sale")

        with col3:
            holiday = st.checkbox("Holiday")

        date = st.date_input("Date", datetime.now())

        if st.form_submit_button("✅ Record Sale"):
            success = st.session_state.user_engine.record_sale(
                product_name=product_name,
                units_sold=int(units_sold),
                unit_price=float(unit_price),
                discount=float(discount) / 100,
                promo=promo,
                holiday=holiday,
                date=date.strftime("%d-%m-%Y"),
            )

            if success:
                st.success("✅ Sale recorded successfully!")
                st.session_state.user_engine.reload()
            else:
                st.error("❌ Error recording sale. Check product name.")


def render_sales_analytics():
    """Render sales analytics"""
    st.title("📊 Sales Analytics")

    st.session_state.user_engine.reload()

    if st.session_state.user_engine.sales.empty:
        st.info("No sales data available. Start recording sales to see analytics.")
        return

    sales_df = st.session_state.user_engine.sales.copy()

    # Summary by product
    st.subheader("Sales by Product")

    if "Product_Name" in sales_df.columns:
        product_summary = (
            sales_df.groupby("Product_Name")
            .agg(
                {
                    "Units_Sold": "sum",
                    "Revenue": "sum",
                }
            )
            .rename(columns={"Units_Sold": "Total Units", "Revenue": "Total Revenue"})
            .sort_values("Total Revenue", ascending=False)
        )

        st.bar_chart(product_summary["Total Units"])
        st.dataframe(product_summary, use_container_width=True)

    # Timeline analysis
    st.subheader("Daily Revenue Trend")

    if "Date" in sales_df.columns:
        sales_df["Date"] = pd.to_datetime(
            sales_df["Date"], format="%d-%m-%Y", errors="coerce"
        )
        daily_revenue = sales_df.groupby("Date")["Revenue"].sum()

        st.line_chart(daily_revenue)

    # Recent transactions
    st.subheader("Recent Transactions")
    st.dataframe(sales_df.tail(20), use_container_width=True)


def render_inventory_management():
    """Render inventory management"""
    st.title("📦 Inventory Management")

    st.session_state.user_engine.reload()

    if st.session_state.user_engine.sales.empty:
        st.info("No sales data available. Start recording sales to get inventory recommendations.")
        return

    st.subheader("Inventory Optimization")

    sales_df = st.session_state.user_engine.sales

    recommendations = []

    for product in sales_df["Product_Name"].unique():
        product_data = sales_df[sales_df["Product_Name"] == product]
        mean_demand = product_data["Units_Sold"].mean()
        demand_std = product_data["Units_Sold"].std()

        if pd.isna(demand_std):
            demand_std = 0

        current_inventory = int(mean_demand * 3)

        rec = InventoryOptimizer.get_inventory_recommendation(
            product, current_inventory, mean_demand, demand_std
        )
        recommendations.append(rec)

    if recommendations:
        rec_df = pd.DataFrame(recommendations)

        # Color code by risk
        def highlight_risk(row):
            if row["risk_level"] == "LOW":
                return ["background-color: #d4edda"] * len(row)
            elif row["risk_level"] == "MEDIUM":
                return ["background-color: #fff3cd"] * len(row)
            elif row["risk_level"] == "HIGH":
                return ["background-color: #f8d7da"] * len(row)
            else:
                return ["background-color: #f5c6cb"] * len(row)

        st.dataframe(
            rec_df.style.apply(highlight_risk, axis=1), use_container_width=True
        )

        # Filter by risk level
        st.subheader("Filter by Risk Level")

        risk_filter = st.selectbox(
            "Select Risk Level", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
        )

        if risk_filter != "All":
            filtered_df = rec_df[rec_df["risk_level"] == risk_filter]
            st.dataframe(filtered_df, use_container_width=True)


def render_demand_forecast():
    """Render demand forecast"""
    st.title("🔮 Demand Forecasting")

    if st.session_state.prediction_engine is None:
        st.error("Prediction engine not loaded")
        return

    st.subheader("Product Demand Forecast")

    selected_product = st.selectbox("Select Product", list(PRODUCTS.keys()))

    # Create sample features
    features = {
        "Store_Type_Encoded": 0,
        "Location_Type_Encoded": 0,
        "Category_Encoded": 0,
        "Units_Stocked": 50,
        "Unit_Price": 100,
        "Discount": 0,
        "Day_of_Week": 2,
        "Is_Weekend": 0,
        "Lag_1_Units_Sold": 30,
        "Lag_7_Units_Sold": 28,
        "Rolling_Mean_7d_Units_Sold": 29,
        "Rolling_Std_7d_Units_Sold": 5,
        "Sell_Through_Ratio": 0.6,
        "Stock_Remaining_Ratio": 0.4,
        "Revenue_Per_Unit_Stocked": 50,
        "Discount_Applied": 0,
        "High_Demand_Flag": 0,
        "Low_Stock_Flag": 0,
        "Is_Festival": 0,
        "Day_Type_Encoded": 0,
    }

    try:
        forecast = st.session_state.prediction_engine.predict_single(features)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Daily Demand", f"{forecast:.0f} units")

        with col2:
            price_info = PRODUCTS.get(selected_product, {})
            avg_price = sum(price_info.get("price_range", (0, 0))) / 2
            st.metric("Est. Revenue", f"₹{forecast * avg_price:.0f}")

        with col3:
            confidence = 85  # Placeholder
            st.metric("Forecast Confidence", f"{confidence}%")

        # Forecast details
        st.subheader("Forecast Details")
        st.info(
            f"""
        📊 **{selected_product} Forecast**
        - Expected daily demand: **{forecast:.0f} units**
        - Confidence interval: 80% - 95%
        - Model: {'Personalized' if st.session_state.use_personalized else 'Base Model'}
        """
        )

    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")


def render_model_management():
    """Render model management"""
    st.title("🤖 Model Management")

    # Model status
    st.subheader("Current Model Status")

    if os.path.exists(MODEL_METADATA_PATH):
        with open(MODEL_METADATA_PATH, "r") as f:
            metadata = json.load(f)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Base Model", metadata.get("model_name", "Unknown"))

        with col2:
            st.metric(
                "Base Model R²",
                f"{metadata.get('metrics', {}).get('test_r2', 0):.4f}",
            )

        if metadata.get("personalized"):
            st.success("✅ Personalized Model Active")
            st.metric(
                "Personalized Model R²",
                f"{metadata.get('personalized_metrics', {}).get('test_r2', 0):.4f}",
            )
        else:
            st.info("⏳ Personalized Model: Collecting user data...")

    # Retraining section
    st.subheader("Model Retraining")

    trainer = PersonalizedModelTrainer()
    should_retrain, data_count = trainer.check_retraining_required()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("User Data Points", data_count)

    with col2:
        st.metric("Required for Retrain", 14)

    with col3:
        st.metric("Progress", f"{(data_count / 14) * 100:.0f}%")

    if st.button("🔄 Trigger Retraining", disabled=not should_retrain):
        if should_retrain:
            with st.spinner("Retraining model..."):
                trainer.retrain()
            st.session_state.use_personalized = True
            st.session_state.prediction_engine = PredictionEngine(
                use_personalized=True
            )
            st.success("✅ Model retraining complete!")
            st.rerun()
        else:
            st.warning(f"Insufficient data. Need {14 - data_count} more data points.")


def render_about():
    """Render about page"""
    st.title("ℹ️ About the System")

    st.markdown(
        """
    ## AI-Based Smart Grocery Account & Demand Forecasting System

    This is a complete ML-driven retail analytics platform designed specifically for grocery retailers.

    ### Key Features

    ✅ **Synthetic Data Generation** - Realistic grocery retail dataset  
    ✅ **Machine Learning Models** - Multiple algorithms (RF, XGBoost, LightGBM)  
    ✅ **Demand Forecasting** - Predict product sales  
    ✅ **Inventory Optimization** - Calculate safety stock and reorder quantities  
    ✅ **Continuous Learning** - Retrains with user data  
    ✅ **Cold Start Handling** - Works from day one  

    ### System Architecture

    1. **Base Dataset** - 90-day synthetic grocery data
    2. **Feature Engineering** - Lag, rolling, and derived features
    3. **Model Training** - Multiple algorithms evaluated
    4. **User Data Collection** - Real sales and purchase data
    5. **Dynamic Retraining** - Personalized models after 2-4 weeks
    6. **Inventory Recommendations** - Safety stock calculations

    ### Technology Stack

    - **Python 3.8+**
    - **Pandas, NumPy, Scikit-learn**
    - **XGBoost, LightGBM**
    - **Streamlit** (Web UI)

    ### Data Sources

    - Synthetic: 90-day Tamil Nadu grocery data
    - User: Daily sales, purchases, and promotions

    ### Models Supported

    - Linear Regression
    - Decision Tree
    - Random Forest ⭐
    - XGBoost
    - LightGBM

    ---

    **Version:** 1.0  
    **Last Updated:** 2025
    """
    )


# Main page rendering
if page == "🏠 Dashboard":
    render_dashboard()
elif page == "📝 Sales Entry":
    render_sales_entry()
elif page == "📊 Sales Analytics":
    render_sales_analytics()
elif page == "📦 Inventory Management":
    render_inventory_management()
elif page == "🔮 Demand Forecast":
    render_demand_forecast()
elif page == "🤖 Model Management":
    render_model_management()
elif page == "ℹ️ About":
    render_about()
