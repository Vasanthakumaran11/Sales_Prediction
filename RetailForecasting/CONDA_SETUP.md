# 🐍 Conda Setup Guide - Detailed Implementation Steps

This guide walks through setting up the Sales Prediction system using **Anaconda**, which avoids the build errors with pip.

---

## Why Conda?

✅ **Pre-built binaries** - No compilation needed (avoids `pkg_resources` error)
✅ **Faster installation** - Quicker than pip (minutes vs. hours)
✅ **Dependency management** - Better handling of complex packages like pandas, xgboost
✅ **Environment isolation** - Keep projects separate and clean

---

## Implementation Steps

### Step 1: Verify Anaconda Installation

```powershell
# Open PowerShell and run:
conda --version

# Expected output: conda 4.x.x
```

If conda is not found:
- Download from: https://www.anaconda.com/download
- Install with default options
- Restart PowerShell after installation

---

### Step 2: Navigate to Project Directory

```powershell
cd C:\Users\Raja\Desktop\Sales_Prediction\RetailForecasting
pwd  # Verify you're in correct directory
```

Expected output:
```
Path
----
C:\Users\Raja\Desktop\Sales_Prediction\RetailForecasting
```

---

### Step 3: Create Conda Environment

Create a fresh Python 3.10 environment for this project:

```powershell
conda create -n sales_pred python=3.10
```

What this does:
- Creates new isolated environment named `sales_pred`
- Uses Python 3.10 (compatible with all packages)
- Takes 15-30 seconds

Expected output:
```
# To activate this environment, use
#
#     $ conda activate sales_pred
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

---

### Step 4: Activate the Environment

```powershell
conda activate sales_pred
```

Verify activation - prompt should change to:
```
(sales_pred) PS C:\Users\Raja\Desktop\Sales_Prediction\RetailForecasting>
```

**Key**: Notice `(sales_pred)` prefix - this means environment is active.

---

### Step 5: Install Core Data Science Packages

```powershell
# Install primary packages from conda-forge channel (better compatibility)
conda install -c conda-forge pandas numpy scikit-learn xgboost lightgbm streamlit matplotlib seaborn python-dateutil
```

**What's installing:**
- **pandas 1.5.3+** - Data manipulation
- **numpy 1.24.3+** - Numerical computing
- **scikit-learn 1.3.0+** - Machine learning algorithms
- **xgboost 2.0.0+** - Gradient boosting
- **lightgbm 4.0.0+** - Light gradient boosting
- **streamlit 1.28.0+** - Web UI framework
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization
- **python-dateutil** - Date utilities

**Time:** 2-5 minutes (pre-built binaries, very fast)

When prompted "Proceed ([y]/n)?", type **y** and press Enter.

Expected output at end:
```
done
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
```

---

### Step 6: Verify Installation

Check that all packages are installed correctly:

```powershell
# List all installed packages in current environment
conda list

# Or check specific packages
pip list | Select-String "pandas|numpy|scikit|xgboost|lightgbm|streamlit"
```

Expected output (sample):
```
pandas                         1.5.3
numpy                          1.24.3
scikit-learn                   1.3.0
xgboost                        2.0.0
lightgbm                       4.0.0
streamlit                      1.28.0
matplotlib                     3.7.2
seaborn                        0.12.2
python-dateutil                2.8.2
```

---

### Step 7: Run the Complete Pipeline

Now that environment is ready, generate data and train models:

```powershell
# Make sure conda environment is active (should show (sales_pred) in prompt)
python src/pipeline/run_pipeline.py
```

**What happens:**
1. **Generates synthetic dataset** - Creates 3,780 transactions (90 days × 3 stores × 14 products)
   - File: `data/raw/base_dataset.csv`
   - Time: 10-15 seconds

2. **Preprocesses data** - Engineers 28 features from raw data
   - File: `data/processed/base_processed.csv`
   - Time: 20-30 seconds

3. **Trains ML models** - Trains 5 different algorithms:
   - Linear Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - LightGBM
   - Time: 30-45 seconds

4. **Selects best model** - Picks model with highest R² score
   - File: `models/base_model.pkl`
   - Time: 5 seconds

5. **Initializes user data system** - Creates data/user/ directory with sample products
   - Files: `data/user/products.csv`, `data/user/sales.csv`, `data/user/purchases.csv`
   - Time: 2-3 seconds

6. **Tests predictions** - Makes sample predictions with trained model
   - Output: Prints sample predictions to console
   - Time: 2-3 seconds

**Total time:** ~2-3 minutes (first run)

**Expected final output:**
```
Pipeline execution complete!
✓ Data generated
✓ Features engineered
✓ Models trained
✓ Best model selected: Linear Regression (R²=0.82)
✓ User data initialized
✓ Predictions tested
```

---

### Step 8: Start Using the System

#### Option A: Terminal Menu Interface

```powershell
python main.py
```

Interactive menu appears:
```
========================================
  Retail Forecasting System
========================================
1. Pipeline Setup
2. Sales Management
3. Inventory Management
4. Demand Forecasting
5. Model Management
6. Exit

Enter your choice: 
```

**Example workflow:**
- Select `1.1` → Run complete pipeline
- Select `2.1` → Record a sale
- Select `3.1` → Get inventory recommendation
- Select `4.1` → Get demand forecast

---

#### Option B: Web Dashboard

```powershell
streamlit run app/app.py
```

Browser opens automatically with interactive dashboard:
- 📊 Sales Dashboard
- 📝 Sales Entry Form
- 📈 Analytics & Visualizations
- 📦 Inventory Management
- 🔮 Demand Forecasting

---

#### Option C: Python API (Programmatic)

```python
import sys
sys.path.insert(0, 'src')

from data.data_engine import UserDataEngine
from models.predict import PredictionEngine

# Record a sale
engine = UserDataEngine()
engine.record_sale('Milk', 20, 25.50, discount=0.05)

# Get prediction
predictor = PredictionEngine()
prediction = predictor.predict_single({...features...})
print(f"Predicted demand: {prediction}")
```

---

## Environment Management

### Deactivate Environment (When Done)

```powershell
conda deactivate
```

Prompt returns to:
```
(base) PS C:\Users\Raja\Desktop\Sales_Prediction\RetailForecasting>
```
Notice: `(base)` instead of `(sales_pred)`

---

### Reactivate Environment (For Future Use)

Every time you want to use the system:

```powershell
cd C:\Users\Raja\Desktop\Sales_Prediction\RetailForecasting
conda activate sales_pred
python main.py
```

---

### Remove Environment (If Needed)

```powershell
conda remove -n sales_pred --all
```

---

## Troubleshooting Conda Installation

### Issue: "conda: command not found"
**Solution:** Anaconda not in PATH
- Restart PowerShell after installing Anaconda
- Or manually add to PATH: `C:\Users\[YourUser]\anaconda3\Scripts`

---

### Issue: "Solving environment: failed"
**Solution:** Conflicting package versions
```powershell
# Clear conda cache and retry
conda clean --all
conda create -n sales_pred python=3.10
conda activate sales_pred
conda install -c conda-forge pandas numpy scikit-learn xgboost lightgbm streamlit
```

---

### Issue: "FileNotFoundError: no_module_named 'pandas'"
**Solution:** Environment not activated
```powershell
# Check prompt - should show (sales_pred)
# If not, run:
conda activate sales_pred
```

---

### Issue: Installation hangs/takes too long
**Solution:** Switch conda channel or use mamba (faster)
```powershell
# Option 1: Use default channel instead of conda-forge
conda create -n sales_pred_v2 python=3.10
conda activate sales_pred_v2
conda install pandas numpy scikit-learn xgboost lightgbm streamlit

# Option 2: Install mamba (faster solver)
conda install mamba -c conda-forge
mamba create -n sales_pred python=3.10
mamba install -c conda-forge pandas numpy scikit-learn xgboost lightgbm streamlit
```

---

## Verification Checklist

After completing all steps, verify everything works:

```powershell
# ✅ Check conda environment active
conda info | Select-String "active environment"
# Should show: active environment : sales_pred

# ✅ Check Python version
python --version
# Should show: Python 3.10.x

# ✅ Check imports work
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, streamlit; print('All imports successful!')"

# ✅ Check data files exist
ls data/raw/base_dataset.csv
ls models/base_model.pkl

# ✅ Try a prediction
python src/models/predict.py
```

---

## Next Steps After Setup

1. ✅ **Explore the system**
   ```powershell
   python main.py
   ```

2. ✅ **Try web dashboard**
   ```powershell
   streamlit run app/app.py
   ```

3. ✅ **Records sales for 2 weeks**
   - Menu option 2.1
   - Add 20-50 transactions
   - Let system learn patterns

4. ✅ **Check inventory recommendations**
   - Menu option 3.1
   - Analyze risk levels

5. ✅ **Get demand forecasts**
   - Menu option 4.1
   - Plan purchasing

6. ✅ **Trigger retraining (after 2+ weeks)**
   - Menu option 5.2
   - System creates personalized model

---

## Advanced: Create Backup Environment

Keep a backup copy in case main environment breaks:

```powershell
# Export current environment
conda env export > sales_pred_backup.yml

# Later, recreate from backup
conda env create -f sales_pred_backup.yml -n sales_pred_restored
conda activate sales_pred_restored
```

---

## Support

If issues persist:
1. Check [Conda Documentation](https://docs.conda.io/)
2. Review project README.md
3. Check terminal error messages carefully
4. Try Step 6 verification checklist

---

**Happy Forecasting! 🚀📊**
