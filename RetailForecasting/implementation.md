# 📋 Implementation Guide - Retail Forecasting System

## Complete Step-by-Step Implementation from Start to End

---

## 🎯 Project Overview

The **AI-Based Smart Grocery Account & Demand Forecasting System** is a complete machine learning-driven retail analytics platform that combines:

- **Smart Dataset Generation** - Creates realistic synthetic grocery data
- **ML Model Training** - Trains 5 different algorithms (Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM)
- **Demand Forecasting** - Predicts daily product demand with high accuracy
- **Inventory Optimization** - Calculates safety stock and reorder quantities
- **Sales Prediction** - Intelligent product recommendations based on opening month
- **Continuous Learning** - Automatically retrains with real user data

---

## 📦 Installation Steps

### Step 1: Install Python and Dependencies

#### Option A: Using Anaconda (Recommended)

```bash
# 1. Open terminal/PowerShell in project directory
cd path/to/RetailForecasting

# 2. Create conda environment
conda create -n sales_pred python=3.10

# 3. Activate environment
conda activate sales_pred

# 4. Install dependencies
conda install -c conda-forge pandas numpy scikit-learn xgboost lightgbm streamlit matplotlib seaborn python-dateutil
```

#### Option B: Using pip

```bash
# 1. Navigate to project
cd path/to/RetailForecasting

# 2. Create virtual environment (optional but recommended)
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

---

### Step 2: Verify Installation

```bash
# Test imports
python -c "import pandas; import sklearn; import xgboost; import lightgbm; print('✅ All dependencies installed!')"
```

---

## 🚀 Running the System

### Method 1: Complete Pipeline (First Time Setup)

```bash
# 1. Activate environment (if not already active)
# conda activate sales_pred   (for conda)
# venv\Scripts\activate       (for venv on Windows)

# 2. Run main application
python main.py

# 3. In the menu, select:
#    Option: 1 → 1.1 "Run complete pipeline"
#    This will:
#    - Generate synthetic dataset
#    - Preprocess data and engineer features
#    - Train all 5 models
#    - Select the best model
#    - Save model and metadata
#    - Initialize user interface
```

---

### Method 2: Interactive Terminal Application

```bash
# Run the interactive application
python main.py

# Menu Options:
# 1.1 - Run complete pipeline (first time)
# 2.1 - Record daily sales
# 2.2 - View sales history
# 3.1 - Get inventory recommendations
# 4.1 - Get demand forecast
# 5.2 - Trigger model retraining (after 2-4 weeks of data)
# 6.1 - View store profile
```

---

### Method 3: Quick Start System

```bash
# Run enhanced interactive application directly
python app_enhanced.py
```

---

## 🏗️ Architecture Flow

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              RETAIL FORECASTING SYSTEM                       │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
            ┌───────▼────────┐  ┌──────▼──────────┐
            │  MAIN.PY       │  │  APP_ENHANCED.PY│
            │ (Terminal CLI) │  │(Interactive App)│
            └───────┬────────┘  └──────┬──────────┘
                    │                   │
                    └───────────┬───────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼────────┐ ┌────▼────────┐ ┌──▼──────────┐
        │  USER MANAGER  │ │   DASHBOARD │ │INPUT HANDLER│
        │  (Store Mgmt)  │ │   (Terminal)│ │(Validation) │
        └───────┬────────┘ └────┬────────┘ └──┬──────────┘
                │               │               │
                └───────────────┼───────────────┘
                                │
        ┌───────────────┬───────┴────────┬──────────────┐
        │               │                │              │
    ┌───▼────┐  ┌──────▼──────┐  ┌──────▼────────┐ ┌──▼──────┐
    │PRODUCTS │  │DATA ENGINE  │  │SALES ANALYTICS│ │INVENTORY│
    │MANAGER  │  │ (Load/Save) │  │ (Metrics)     │ │OPTIMIZER │
    └────┬────┘  └──────┬──────┘  └───────┬───────┘ └──┬──────┘
         │               │                 │             │
         └───────────────┼─────────────────┼─────────────┘
                         │                 │
         ┌───────────────┴─────────────────┘
         │
    ┌────▼─────────────────────────────────────┐
    │  ML PIPELINE                              │
    │  ├─ Data Generation (src/data/)           │
    │  ├─ Preprocessing (src/preprocessing/)    │
    │  ├─ Model Training (src/models/)          │
    │  ├─ Prediction Engine (src/models/)       │
    │  └─ Sales Predictor (src/analytics/)      │
    └────┬─────────────────────────────────────┘
         │
    ┌────▼─────────────────┐
    │  DATA & MODELS       │
    │  ├─ data/raw/        │
    │  ├─ data/processed/  │
    │  ├─ data/user/       │
    │  └─ models/          │
    └──────────────────────┘
```

---

## 📊 Complete Data Flow

### Phase 1: System Initialization

```
1. SYNTHETIC DATA GENERATION
   ├─ File: src/data/generate_data.py
   ├─ Input: Config (3 stores, 14 products, 90 days)
   ├─ Process: Creates realistic grocery data
   └─ Output: data/raw/base_dataset.csv (3,780 rows, 20 features)

2. PREPROCESSING & FEATURE ENGINEERING
   ├─ File: src/preprocessing/preprocess.py
   ├─ Input: base_dataset.csv
   ├─ Process: 
   │   ├─ Handle missing values
   │   ├─ Encode categorical variables
   │   ├─ Create lag features (1-day, 7-day)
   │   ├─ Create rolling statistics
   │   ├─ Derive new features (ratios, flags)
   │   └─ Normalize numerical features
   └─ Output: data/processed/base_processed.csv (3,780 rows, 28 features)

3. MODEL TRAINING
   ├─ File: src/models/train_base.py
   ├─ Input: base_processed.csv
   ├─ Process:
   │   ├─ Split data (80% train, 20% test)
   │   ├─ Train 5 models:
   │   │   ├─ Linear Regression
   │   │   ├─ Decision Tree
   │   │   ├─ Random Forest
   │   │   ├─ XGBoost
   │   │   └─ LightGBM
   │   ├─ Evaluate each model
   │   └─ Select best performer
   └─ Output: models/base_model.pkl + model_metadata.json

4. SYSTEM INITIALIZATION
   ├─ File: app_enhanced.py
   ├─ Initialize user interface
   ├─ Create data directories
   └─ Ready for user interaction
```

### Phase 2: User Interaction & Data Collection

```
5. STORE REGISTRATION
   ├─ User registers new store
   ├─ Selects opening month (for sales prediction)
   ├─ System recommends products based on:
   │   ├─ Investment amount
   │   ├─ Store type (Small/Medium/Supermarket)
   │   ├─ Location type (Urban/Semi-Urban/Rural)
   │   └─ Opening month (seasonal factors)
   └─ Creates store directory and profile

6. DAILY SALES ENTRY
   ├─ User records daily sales:
   │   ├─ Product sold
   │   ├─ Units sold
   │   ├─ Unit price
   │   ├─ Discount percentage
   │   ├─ Promotional flag
   │   ├─ Holiday flag
   │   └─ Closed flag
   └─ Data saved to: data/user/{store_name}/sales.csv

7. INVENTORY MANAGEMENT
   ├─ Track current stock levels
   ├─ Add new products to inventory
   ├─ View inventory status
   └─ Data saved to: data/user/{store_name}/products.csv
```

### Phase 3: Predictions & Recommendations

```
8. DEMAND FORECASTING
   ├─ Input: Historical user data + base model
   ├─ Process: Make predictions for next month
   └─ Output: Expected units, revenue, trends

9. INVENTORY OPTIMIZATION
   ├─ Calculate safety stock = Z × σ × √LeadTime
   ├─ Calculate reorder point = Lead time demand + Safety stock
   ├─ Calculate Economic Order Quantity (EOQ)
   ├─ Risk level assessment
   └─ Stockout probability warnings

10. SALES PREDICTIONS (Month-based)
    ├─ Input: Opening month + store characteristics
    ├─ Analyze historical data for that month
    ├─ Calculate category-wise predictions
    ├─ Allocate investment proportionally
    └─ Recommend specific products with quantities
```

### Phase 4: Continuous Learning

```
11. AUTOMATIC RETRAINING (After 2-4 weeks)
    ├─ File: src/models/train_personalized.py
    ├─ Combine base data + user collected data
    ├─ Preprocess combined dataset
    ├─ Train model on personalized data
    ├─ Compare metrics with base model
    └─ Save as: models/personalized_model.pkl

12. ENHANCED PREDICTIONS
    ├─ Use personalized model for future predictions
    ├─ More accurate since trained on user's store data
    └─ Continuous improvement with more data
```

---

## 📁 Project Structure

```
RetailForecasting/
│
├── 📄 implementation.md              # Step-by-step implementation guide
├── 📄 Sales_Prediction_architecture.md  # Complete architecture & metrics
├── 📄 requirements.txt               # All dependencies
├── 📄 main.py                        # Terminal interactive CLI
├── 📄 app_enhanced.py                # Enhanced interactive application
│
├── 📂 src/
│   ├── data/
│   │   ├── generate_data.py          # Synthetic data generator
│   │   └── data_engine.py            # User data handler
│   │
│   ├── preprocessing/
│   │   └── preprocess.py             # Feature engineering
│   │
│   ├── models/
│   │   ├── train_base.py             # Base model training
│   │   ├── train_personalized.py     # Retraining pipeline
│   │   └── predict.py                # Prediction engine
│   │
│   ├── utils/
│   │   ├── config.py                 # Configuration
│   │   └── inventory.py              # Inventory optimization
│   │
│   ├── analytics/
│   │   ├── sales_analytics.py        # Sales analysis
│   │   └── sales_predictor.py        # AI sales prediction
│   │
│   ├── interface/
│   │   ├── dashboard.py              # Terminal UI
│   │   └── input_handler.py          # Input validation
│   │
│   ├── users/
│   │   ├── user_manager.py           # Store management
│   │   └── product_manager.py        # Product recommendations
│   │
│   ├── inventory/
│   │   └── inventory_manager.py      # Inventory optimization
│   │
│   └── pipeline/
│       └── run_pipeline.py           # Main orchestration
│
├── 📂 data/
│   ├── raw/
│   │   └── base_dataset.csv          # Generated synthetic data
│   │
│   ├── processed/
│   │   └── base_processed.csv        # Processed features
│   │
│   └── user/
│       └── {store_name}/
│           ├── profile.json          # Store metadata
│           ├── products.csv          # Inventory
│           ├── sales.csv             # Sales records
│           └── purchases.csv         # Cost tracking
│
├── 📂 models/
│   ├── base_model.pkl                # Trained model
│   ├── personalized_model.pkl        # Retrained model
│   └── model_metadata.json           # Model info & metrics
│
└── 📂 app/
    └── (deprecated Streamlit app removed)
```

---

## ✅ Quick Start Checklist

- [ ] Install Python 3.10+
- [ ] Install all dependencies (pip or conda)
- [ ] Run: `python main.py` → Select "1.1" to initialize system
- [ ] Test system by creating a store in the interactive menu
- [ ] Record some daily sales entries
- [ ] View demand forecasts and inventory recommendations
- [ ] After 2-4 weeks of data: Trigger model retraining for personalized predictions

---

## 🎯 Common Use Cases

### Use Case 1: New Grocery Store Owner
```
Goal: Set up forecasting system for new store

Steps:
1. pip install -r requirements.txt
2. python main.py
3. Select "1.1" - Run complete pipeline
4. Select "2.1" - Register new store (enter opening month)
5. Review recommended products
6. Daily: Record sales (Option 2.1)
7. After 2-4 weeks: Trigger retraining (Option 5.2)
8. Get monthly forecasts (Option 4.1)
```

### Use Case 2: Existing Store with Data
```
Goal: Improve predictions with historical data

Steps:
1. Load system with: python main.py
2. Load existing store (Option 2.1)
3. Enter historical sales data
4. Wait 2-4 weeks (or bulk import data)
5. Trigger retraining (Option 5.2)
6. Get personalized predictions
```

### Use Case 3: Inventory Planning
```
Goal: Optimize inventory levels

Steps:
1. python main.py
2. View current store (Option 6.1)
3. Get inventory recommendations (Option 3.1)
4. Review:
   - Safety stock levels
   - Reorder points
   - Economic Order Quantities
5. Plan procurement accordingly
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError" for imports
**Solution:** Ensure environment is activated and all dependencies installed
```bash
pip install -r requirements.txt
```

### Issue: "File not found" errors
**Solution:** Make sure you're in the correct directory
```bash
cd path/to/RetailForecasting
```

### Issue: Model training too slow
**Solution:** System takes 2-3 minutes first time (expected). Subsequent runs are faster.

### Issue: Predictions seem inaccurate
**Solution:** Collect more data (2-4 weeks) and trigger retraining for personalized model
```bash
# In main.py menu, select Option 5.2
```

---

## 📞 Support

For issues or questions about implementation:
1. Check this guide for common solutions
2. Review the project structure in the folder
3. Ensure all files in `src/` directory are present
4. Verify `data/` and `models/` directories exist and have required files

