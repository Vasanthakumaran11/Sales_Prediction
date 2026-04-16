# 🚀 Quick Start Guide

## Installation & Setup (5 minutes)

### 1. Install Dependencies
```bash
cd RetailForecasting
pip install -r requirements.txt
```

### 2. Generate Base Dataset & Train Model
Choose one method:

#### Option A: Terminal GUI (Recommended)
```bash
python main.py
# Then select: 1.1 → Run complete pipeline
```

#### Option B: Direct Script
```bash
python src/pipeline/run_pipeline.py
```

This will:
- Generate 90-day synthetic dataset ✅
- Preprocess and engineer features ✅
- Train 5 ML models ✅
- Select best performer ✅
- Initialize user data system ✅

**Time:** ~2-3 minutes

---

## Using the System

### Method 1: Terminal Interface
```bash
python main.py
```

Menu options:
- **1.1** - Setup & Run pipeline
- **2.1** - Record daily sales
- **2.2** - View sales summary
- **3.1** - Get inventory recommendations
- **4.1** - Get demand forecast
- **5.1** - Check model status
- **5.2** - Trigger retraining (after 2-4 weeks)

### Method 2: Web Interface
```bash
streamlit run app/app.py
```

Features:
- 📝 Daily sales entry form
- 📊 Sales analytics dashboard
- 📦 Inventory management
- 🔮 Demand forecasting
- 🤖 Model management

### Method 3: Python API
```python
import sys
sys.path.insert(0, 'src')

# Record a sale
from data.data_engine import UserDataEngine
user_engine = UserDataEngine()
user_engine.record_sale('Milk', 20, 25.50, discount=0.1)

# Get prediction
from models.predict import PredictionEngine
engine = PredictionEngine()
features = {'Store_Type_Encoded': 0, ...}
prediction = engine.predict_single(features)

# Get inventory recommendation
from utils.inventory import InventoryOptimizer
rec = InventoryOptimizer.get_inventory_recommendation(
    'Milk', 45, 12, 2  # product, inventory, mean_demand, std
)
```

---

## Daily Workflow

### Day 1-4 Weeks: Collection Phase
Use base model predictions while collecting user data

**Record Daily Sales:**
```
Main Menu → 2.1 → Record daily sales
Product: Milk
Units Sold: 20
Unit Price: ₹25
Discount: 10% (optional)
Promotional Sale: Yes/No
Holiday: Yes/No
```

### Week 2-4: Monitoring Phase
Check inventory and demand patterns

**Get Recommendations:**
```
Main Menu → 3.1 → View inventory recommendations
```

**View Demand Forecast:**
```
Main Menu → 4.1 → Get product forecast
```

### Week 4+: Intelligence Phase
System gets personalized and smarter

**Check Model Status:**
```
Main Menu → 5.1 → Check model status
```

**After 14 Days (2 weeks):**
```
Main Menu → 5.2 → Trigger retraining
```

---

## Example: Complete Day-in-Life

### Morning (9 AM)
```bash
python main.py
→ 2.1: Record yesterday's sales
  Product: Milk, 25 units, ₹25, no discount
  Product: Rice, 15 units, ₹60, 5% discount
  Product: Oil, 8 units, ₹150, no discount
```

### Mid-day (12 PM)
```bash
python main.py
→ 3.1: Check inventory levels
→ 4.1: Get today's demand forecast
```

 Results:
```
✅ Milk: 25 units forecast, Good stock level
⚠️  Rice: 42 units forecast, Reorder recommended
🔴 Oil: Critical - Order 15 units ASAP
```

### Evening (5 PM)
```bash
python main.py
→ 2.2: View daily sales summary
→ 5.1: Check model performance
```

---

## Data File Locations

All user data automatically saved to:

```
data/
├── user/
│   ├── sales.csv              # Daily sales (auto-save)
│   ├── products.csv           # Product catalog
│   └── purchases.csv          # Cost tracking
│
├── raw/
│   └── base_dataset.csv       # Synthetic data (generated once)
│
└── processed/
    └── base_processed.csv     # Features (generated once)
```

---

## Interpreting Results

### Inventory Recommendations

| Risk Level | Stock Status | Action |
|-----------|------------|--------|
| 🟢 LOW | Well-stocked | Monitor |
| 🟡 MEDIUM | Normal | Order as planned |
| 🔴 HIGH | Below threshold | Priority reorder |
| 🔴🔴 CRITICAL | Running out | URGENT order |

### Demand Forecast

```
Predicted Daily Demand: 25 units
Confidence: 85%
Expected Revenue: ₹625 (at ₹25/unit)
```

### Model Performance

```
Base Model R²: 0.82 (Initial)
↓
Personalized Model R²: 0.88 (After 2-4 weeks)
```

---

## Troubleshooting

### Issue: "Model not found"
```bash
# Regenerate model
python src/models/train_base.py
```

### Issue: "Module not found"
```bash
# Ensure path is correct
cd RetailForecasting
pip install -r requirements.txt
```

### Issue: Streamlit won't start
```bash
# Upgrade and reinitialize
pip install --upgrade streamlit
streamlit run app/app.py --logger.level=debug
```

### Issue: Very slow training
- Normal for first run (2-3 minutes)
- Subsequent runs are faster
- Consider running during off-hours for large datasets

---

## Key Files to Know

| File | Purpose | When to Use |
|------|---------|------------|
| `main.py` | Interactive terminal app | Daily operations |
| `app/app.py` | Web dashboard | Analytics & visualization |
| `src/data/data_engine.py` | Sales recording | Recording transactions |
| `src/models/predict.py` | Make predictions | Getting forecasts |
| `src/utils/config.py` | System configuration | Customization |

---

## Performance Tips

1. **First Time Setup**
   - Takes 2-3 minutes for full pipeline
   - Subsequent runs are < 1 minute

2. **Recording Sales**
   - Batch entry multiple products at once
   - Use consistent product names

3. **Getting Best Results**
   - Record sales for 2-4 weeks
   - Let system learn patterns
   - Retrain model after data accumulation

4. **Optimal Retraining**
   - Trigger after 14-30 days
   - More data = better accuracy
   - System suggests when ready

---

## Next Steps

1. ✅ Run `python main.py` → 1.1
2. ✅ Record sample sales (10-20 entries)
3. ✅ View recommendations (Menu 3.1)
4. ✅ Check forecasts (Menu 4.1)
5. ✅ Try web interface `streamlit run app/app.py`
6. ✅ Explore config in `src/utils/config.py`
7. ✅ Read full README.md

---

## Support

For issues:
- Check project `README.md`
- Review terminal error messages
- Verify all files in `data/` directory
- Ensure all dependencies installed: `pip list | grep -E "pandas|scikit|xgboost"`

---

**Happy Forecasting! 📊🚀**
