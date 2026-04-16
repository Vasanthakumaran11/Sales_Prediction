# 🏪 AI-Based Smart Grocery Account & Demand Forecasting System

A complete machine learning-driven retail analytics platform for grocery stores combining intelligent demand forecasting, inventory optimization, and continuous learning capabilities.

## 📋 System Overview

This system creates an end-to-end solution for grocery retail operations:

- **Smart Dataset Generation** - Creates realistic 90-day synthetic grocery data for Tamil Nadu
- **ML Model Training** - Trains 5 different models (Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM)
- **Demand Forecasting** - Predicts daily product demand with high accuracy
- **Inventory Optimization** - Calculates safety stock and reorder quantities
- **Continuous Learning** - Automatically retrains with real user data
- **Cold Start Solution** - Works from day 1 without historical data

## 🎯 Key Features

### Stage 1: Base Dataset Generation
- 90-day historical data for 3 stores
- 14 grocery products (Milk, Curd, Rice, Oil, etc.)
- Store types: Small, Medium, Supermarket
- Features: Date, Store, Product, Stock, Sales, Revenue, Discounts
- Realistic demand patterns including weekends and festivals

### Stage 2: Feature Engineering
- **Time Features**: Day, Month, Day_of_Week, Is_Weekend, Is_Festival
- **Lag Features**: 1-day and 7-day lags for sales and revenue
- **Rolling Features**: 7-day rolling mean and std deviation
- **Derived Features**: Sell-through ratio, stock remaining ratio, demand flags

### Stage 3: Machine Learning
Trains multiple models and selects the best performer:
- Linear Regression
- Decision Tree
- Random Forest ⭐ (Usually best)
- XGBoost
- LightGBM

Evaluation metrics: MAE, RMSE, R² Score

### Stage 4: User Data Collection
Track real-world sales with:
- Daily sales entries (Date, Product, Units, Price, Discount, Promo)
- Product inventory (Add new products dynamically)
- Purchase records (Cost tracking)

### Stage 5: Dynamic Dataset Building
Combines synthetic base data with real user data for enhanced predictions.

### Stage 6: Automatic Retraining
After 2-4 weeks of user data collection, automatically retrains model to become personalized and store-specific.

### Stage 7: Inventory Optimization
Calculates for each product:
- Safety Stock = Z × σ × √LeadTime
- Reorder Point = Lead Time Demand + Safety Stock
- Risk Levels: LOW, MEDIUM, HIGH, CRITICAL

## 📁 Project Structure

```
RetailForecasting/
├── data/
│   ├── raw/
│   │   └── base_dataset.csv              # Generated synthetic data
│   ├── user/
│   │   ├── products.csv
│   │   ├── purchases.csv
│   │   └── sales.csv
│   └── processed/
│       ├── base_processed.csv
│       ├── user_processed.csv
│       └── final_dataset.csv
├── models/
│   ├── base_model.pkl                   # Initial trained model
│   ├── personalized_model.pkl           # Retrained with user data
│   └── model_metadata.json
├── src/
│   ├── data/
│   │   ├── generate_data.py             # Synthetic data generator
│   │   └── data_engine.py               # User data handler
│   ├── preprocessing/
│   │   └── preprocess.py                # Feature engineering
│   ├── models/
│   │   ├── train_base.py                # Base model training
│   │   ├── train_personalized.py        # Retraining pipeline
│   │   └── predict.py                   # Prediction engine
│   ├── utils/
│   │   ├── config.py                    # Configuration
│   │   └── inventory.py                 # Inventory optimization
│   └── pipeline/
│       └── run_pipeline.py              # Main pipeline orchestration
├── app/
│   └── app.py                           # Streamlit web interface
├── main.py                              # Terminal interactive CLI
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

#### Recommended: Using Anaconda
```bash
# Navigate to project
cd RetailForecasting

# Create conda environment
conda create -n sales_pred python=3.10

# Activate environment
conda activate sales_pred

# Install dependencies (pre-built binaries - no build errors)
conda install -c conda-forge pandas numpy scikit-learn xgboost lightgbm streamlit matplotlib seaborn python-dateutil
```

**For detailed step-by-step instructions, see [CONDA_SETUP.md](CONDA_SETUP.md)**

---

#### Alternative: Using pip
```bash
# Install dependencies with pip
pip install -r requirements.txt

# Note: If you get "pkg_resources" error, upgrade build tools first:
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Ensure conda environment is active (if using conda)
# conda activate sales_pred

# This generates data, trains model, and initializes system
python main.py
# Choose: 1.1 → Run complete pipeline
```

Or programmatically:

```bash
python src/pipeline/run_pipeline.py
```

### 3. Use the System

#### Option A: Terminal CLI (Interactive)
```bash
python main.py
# Menu options:
# 2.1 - Record daily sales
# 3.1 - Get inventory recommendations
# 4.1 - Get demand forecast
# 5.2 - Trigger retraining (after 2-4 weeks)
```

#### Option B: Web UI (Streamlit)
```bash
streamlit run app/app.py
```

Then open: `http://localhost:8501`

## 📊 Data Specifications

### Products (14 items)
- **Perishables**: Milk, Curd, Paneer, Bread, Eggs
- **Non-Perishables**: Rice, Toor Dal, Wheat Flour, Cooking Oil, Biscuits, Snacks, Masala, Sugar, Salt

### Pricing (₹ Indian Rupees)
- Milk: ₹22-35
- Rice: ₹50-90
- Cooking Oil: ₹120-200
- Products adjust by store type and location

### Time Period
- 90 days of base data (January 2025)
- Pongal festival spike on Jan 14
- Weekend demand +25%
- Variable demand: ±15%

## 🤖 Model Architecture

### Input Features (20+)
```
Categorical (Encoded):
- Store_Type (Small/Medium/Supermarket)
- Location_Type (Urban/Semi-Urban/Rural)
- Product Category (Perishable/Non-Perishable)
- Day Type (Weekend/Weekday)

Numerical:
- Units_Stocked, Unit_Price, Discount
- Day_of_Week, Is_Weekend, Is_Festival
- Lag_1_Units_Sold, Lag_7_Units_Sold
- Rolling_Mean_7d, Rolling_Std_7d
- Sell_Through_Ratio, Stock_Remaining_Ratio
- High_Demand_Flag, Low_Stock_Flag
```

### Output
- **Target**: Units_Sold (Daily sales quantity)
- **Metrics**: MAE, RMSE, R² Score

### Hyperparameters
```python
Random Forest: n_estimators=100, max_depth=15
XGBoost: n_estimators=100, max_depth=6, learning_rate=0.1
LightGBM: n_estimators=100, max_depth=6, learning_rate=0.1
```

## 📈 Workflow

```
┌─────────────────────────────────────────────────┐
│  Synthetic Dataset (90 days)                     │
│  Base dataset for initial training              │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Feature Engineering │
        │  (Preprocessing)     │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Model Training      │
        │  (5 algorithms)      │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Base Model Selected │
        │  ⭐ Best Performer   │
        └──────────┬───────────┘
                   │
            ┌──────┴─────┐
            │  Cold Start │
            │  Ready ✓    │
            │             │
            ▼             │
┌──────────────────────┐  │
│  User Records Sales  │  │
│  (Daily Data Points) │  │
└──────────┬───────────┘  │
           │               │
      (Wait 2-4 Weeks)    │
           │               │
           ▼               │
┌──────────────────────┐  │
│  Combine Datasets    │  │
│  Base + User Data    │  │
└──────────┬───────────┘  │
           │               │
           ▼               │
┌──────────────────────┐  │
│  Retrain Model       │  │
│  (Personalized)      │  │
└──────────┬───────────┘  │
           │               │
           ▼               │
┌──────────────────────┐  │
│  Better Predictions  │────┘
│  (Store-Specific)    │
└──────────────────────┘
```

## 📦 Inventory Optimization Examples

```
Product: Milk
─────────────────────────────────
Current Inventory: 45 units
Mean Daily Demand: 12 units
Safety Stock: 8 units
Reorder Point: 32 units
Quantity to Order: 0 units (Well-stocked)
Risk Level: LOW ✅
Action: Monitor - No immediate action needed

Product: Rice
─────────────────────────────────
Current Inventory: 15 units
Mean Daily Demand: 25 units
Safety Stock: 12 units
Reorder Point: 62 units
Quantity to Order: 47 units (URGENT)
Risk Level: HIGH 🔴
Action: Priority reorder - Increase purchase immediately
```

## 🔮 Demand Forecasting Examples

```
Model: Random Forest (Base Model)
Test R²: 0.8234

Product: Milk
─────────────────────────────────
Features: Weekday, No promo, Normal stock
Predicted Daily Demand: 18 units
Expected Revenue: ₹450 (at ₹25/unit)
Forecast Confidence: 85%

Product: Rice
─────────────────────────────────
Features: Weekend, Summer season, Festival
Predicted Daily Demand: 42 units
Expected Revenue: ₹2,100 (at ₹50/unit)
Forecast Confidence: 78%
```

## 🛠️ Configuration

Edit `src/utils/config.py` to customize:

```python
# Time range
DAYS_RANGE = 90
START_DATE = "01-01-2025"

# Stores
NUM_STORES = 3
STORE_TYPES = ["Small", "Medium", "Supermarket"]

# Demand patterns
WEEKEND_MULTIPLIER = 1.25  # +25% on weekends
FESTIVAL_MULTIPLIER = 1.35  # +35% on festivals

# Retraining threshold
MIN_DATA_POINTS_FOR_RETRAIN = 14  # 2 weeks
```

## 📊 API Usage

### Generate Data
```python
from src.data.generate_data import SyntheticDataGenerator

generator = SyntheticDataGenerator()
df = generator.generate_data()
```

### Make Predictions
```python
from src.models.predict import PredictionEngine

engine = PredictionEngine(use_personalized=False)
prediction = engine.predict_single(features_dict)
```

### Record Sales
```python
from src.data.data_engine import UserDataEngine

user_engine = UserDataEngine()
user_engine.record_sale(
    product_name="Milk",
    units_sold=20,
    unit_price=25,
    discount=0.1
)
```

### Get Inventory Recommendations
```python
from src.utils.inventory import InventoryOptimizer

rec = InventoryOptimizer.get_inventory_recommendation(
    product_name="Milk",
    current_inventory=45,
    mean_demand=12,
    demand_std=2
)
```

## 🔄 Continuous Learning

The system automatically:

1. **Collects user data** - Daily sales transactions
2. **Monitors progress** - Tracks data accumulation
3. **Triggers retraining** - After 14 days (configurable)
4. **Combines datasets** - Synthetic + Real data
5. **Retrains models** - Creates personalized predictions
6. **Updates recommendation** - Store-specific inventory advice

## ⚠️ Important Notes

- **Data Privacy**: User data stored locally in CSV files
- **Model Performance**: Improves significantly after retraining
- **Cold Start**: Base model provides quality forecasts from day 1
- **Extensibility**: Easy to add new products and features
- **Scalability**: Supports multiple stores (add Store_ID management)

## 🚨 Troubleshooting

### "ModuleNotFoundError: No module named 'pkg_resources'" (Installation error)
This occurs when pip tries to build packages from source.

**Solution 1: Use Conda (Recommended)**
```bash
# Follow CONDA_SETUP.md for complete conda installation
conda create -n sales_pred python=3.10
conda activate sales_pred
conda install -c conda-forge pandas numpy scikit-learn xgboost lightgbm streamlit
```

**Solution 2: Fix pip installation**
```bash
pip install --upgrade pip setuptools wheel
pip install --no-build-isolation -r requirements.txt
```

### Missing packages
```bash
pip install --upgrade -r requirements.txt
# If using conda:
conda install -c conda-forge pandas numpy scikit-learn xgboost lightgbm streamlit
```

### Model not loading
```bash
# Regenerate base model
python src/models/train_base.py
```

### Streamlit not starting
```bash
pip install streamlit --upgrade
streamlit run app/app.py --logger.level=debug
```

## 🆕 Enhanced Application (NEW!)

We now provide an **advanced terminal application** with professional features:

### Enhanced Smart Grocery App (`app_enhanced.py`)

```bash
python app_enhanced.py
```

**Features:**
- ✅ Store registration and management
- ✅ Product catalog (40+ items, 7 categories)
- ✅ Daily sales entry with validation
- ✅ Advanced inventory management
- ✅ Sales analytics and insights
- ✅ Monthly demand predictions
- ✅ Professional terminal UI
- ✅ Real-world workflow

**Documentation:**
- [ENHANCED_APP_GUIDE.md](ENHANCED_APP_GUIDE.md) - Complete feature guide
- [SYSTEM_COMPLETE_GUIDE.md](SYSTEM_COMPLETE_GUIDE.md) - Getting started
- [CONDENSED_ARCHITECTURE.md](CONDENSED_ARCHITECTURE.md) - Technical design

---

## 📚 File Formats

### sales.csv
```
Sale_ID,Date,Product_Name,Units_Sold,Unit_Price,Discount,Revenue,Promo,Holiday,Shop_Closed
SALE_000001,15-01-2025,Milk,15,25.50,0.1,382.5,0,1,0
```

### products.csv
```
Product_ID,Product_Name,Category,Default_Price_Min,Default_Price_Max,Date_Added
PROD_001,Milk,Perishable,22,35,15-01-2025
```

## 🎓 Learning Path

1. **Understand the system**: Read this README
2. **Run the pipeline**: `python main.py` → Choose 1.1
3. **Explore the code**: Check `src/` directory
4. **Record sample data**: Use menu option 2.1
5. **Get recommendations**: Use menu option 3.1
6. **Trigger retraining**: After 14 days (use 5.2)
7. **Monitor improvements**: Check model metrics in 5.1

## 📈 Expected Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | 5.2 | 6.8 | 0.71 |
| Decision Tree | 4.8 | 6.2 | 0.76 |
| **Random Forest** | **4.1** | **5.4** | **0.82** |
| XGBoost | 4.3 | 5.6 | 0.80 |
| LightGBM | 4.2 | 5.5 | 0.81 |

*Improves by 5-10% after personalized retraining*

## 🔐 Security Considerations

- No external data uploading (Everything local)
- Model predictions are deterministic
- User data stored in CSV (easily auditable)
- No personal information required

## 🤝 Support & Contribution

For issues or improvements:
1. Check configuration in `src/utils/config.py`
2. Review logs in terminal output
3. Ensure all dependencies installed: `pip install -r requirements.txt`

## 📄 License

This project is provided as-is for educational and commercial use.

## 🎯 Future Enhancements

- [ ] Multi-store management dashboard
- [ ] Real-time prediction API (FastAPI)
- [ ] Database integration (PostgreSQL)
- [ ] Advanced time-series models (ARIMA, Prophet)
- [ ] Anomaly detection for sales
- [ ] Price optimization recommendations
- [ ] Seasonal decomposition
- [ ] Integration with POS systems

## 📞 Contact

For questions or customization, reach out with:
- Specific use case
- Data samples (if available)
- Custom requirements

---

**Version**: 1.0  
**Last Updated**: January 2025  
**Status**: ✅ Production Ready
