# 📤 GitHub Deployment Guide

## What to Push to GitHub ✅

```
RetailForecasting/
├── src/                              # All Python modules (PUSH)
│   ├── data/generate_data.py        ✅
│   ├── data/data_engine.py          ✅
│   ├── preprocessing/preprocess.py  ✅
│   ├── models/train_base.py         ✅
│   ├── models/train_personalized.py ✅
│   ├── models/predict.py            ✅
│   ├── utils/config.py              ✅
│   ├── utils/inventory.py           ✅
│   └── pipeline/run_pipeline.py     ✅
│
├── app/app.py                        ✅ Streamlit UI
├── main.py                           ✅ Terminal CLI
├── examples.py                       ✅ Usage examples
│
├── requirements.txt                  ✅ Dependencies
├── README.md                         ✅ Documentation
├── QUICKSTART.md                     ✅ Setup guide
├── ARCHITECTURE.md                   ✅ Technical docs
├── FILE_INVENTORY.md                 ✅ File reference
│
├── .gitignore                        ✅ NEW - Git config
└── Directory structure (.gitkeep)    ✅ NEW - Preserve folders
```

---

## What NOT to Push ❌

```
RetailForecasting/
├── data/                             ❌ (Ignored)
│   ├── raw/*.csv                    ❌ Large synthetic data
│   ├── user/*.csv                   ❌ Private user data
│   └── processed/*.csv              ❌ Computed features
│
├── models/                           ❌ (Ignored)
│   ├── *.pkl                        ❌ Serialized models
│   ├── *.joblib                     ❌ Large binaries
│   └── model_metadata.json          ❌ Runtime metadata
│
├── **/__pycache__/                  ❌ Python cache
├── *.pyc                            ❌ Compiled Python
├── .pytest_cache/                   ❌ Test cache
│
├── .venv/, venv/, env/              ❌ Virtual environments
├── .vscode/, .idea/                 ❌ IDE files
├── .env                             ❌ Secrets/credentials
│
└── OS files                         ❌
    ├── .DS_Store                    ❌ macOS
    ├── Thumbs.db                    ❌ Windows
    └── *.swp, *~                    ❌ Editor temps
```

---

## File Sizes (For Reference)

| File/Directory | Size | Push? |
|---|---|---|
| `src/` (all code) | ~250 KB | ✅ YES |
| `requirements.txt` | ~0.2 KB | ✅ YES |
| `README.md` | ~14.5 KB | ✅ YES |
| `data/raw/base_dataset.csv` | ~1.2 MB | ❌ NO |
| `models/base_model.pkl` | ~10 KB | ❌ NO |
| `.gitignore` | ~2 KB | ✅ YES |
| **Total to push** | **~280 KB** | ✅ |
| **Total excluded** | **~1.2 MB** | ❌ |

---

## Git Commands

### Initialize Repository

```bash
cd RetailForecasting

# Initialize git
git init

# Add remote (replace USERNAME/REPO)
git remote add origin https://github.com/USERNAME/RetailForecasting.git

# Add files (respects .gitignore)
git add .

# Verify what will be pushed
git status

# Commit
git commit -m "Initial commit: AI-Based Retail Forecasting System"

# Push
git branch -M main
git push -u origin main
```

### After Initial Push (Regular Updates)

```bash
# Update source code only
git add src/ *.py *.md requirements.txt .gitignore

# Commit
git commit -m "Update: Feature X or Bug fix Y"

# Push
git push origin main
```

### Never Push

```bash
# These are automatically ignored by .gitignore
data/              # Contains user/generated data
models/            # Contains trained models
__pycache__/       # Python cache
.venv/             # Virtual environment
```

---

## .gitignore Verification

Check what would be ignored:

```bash
# List files that would be ignored
git check-ignore -v *

# See what would be committed
git status

# Dry-run to see what would be added
git add -n .
```

---

## GitHub Repository Setup

### Recommended Structure

```
https://github.com/USERNAME/RetailForecasting/
├── main branch
│   ├── All source code ✅
│   ├── Documentation ✅
│   ├── .gitignore ✅
│   └── requirements.txt ✅
│
└── Releases
    └── v1.0: Initial release
```

### README for Users Cloning

Add to start of README.md:

```markdown
## ⚡ Quick Start

1. **Clone repository**
   ```bash
   git clone https://github.com/USERNAME/RetailForecasting.git
   cd RetailForecasting
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run system**
   ```bash
   python main.py
   # Or: streamlit run app/app.py
   ```

**Note:** The `data/` and `models/` directories are empty initially.
They will be populated when you run the system for the first time.
```

---

## .gitignore Rules Explained

```
# Python Cache
__pycache__/          # Byte-compiled Python
*.py[cod]             # .pyc, .pyo, .pyd files
*.egg-info/           # Pip metadata

# Data (Important!)
data/raw/*.csv        # Large synthetic data (~1.2 MB)
data/user/*.csv       # Private user data
data/processed/*.csv  # Computed features

# Models (Important!)
models/*.pkl          # Trained models (binary)
models/*.joblib       # Joblib format

# Virtual Environment
venv/                 # Project-specific venv
.venv/                # Alternative venv name

# IDE
.vscode/              # VS Code settings
.idea/                # PyCharm settings

# OS
.DS_Store             # macOS
Thumbs.db             # Windows icons
```

---

## What Happens When Users Clone

### Initial State After Clone
```
RetailForecasting/
├── src/              ✅ Present
├── app/              ✅ Present
├── main.py           ✅ Present
├── requirements.txt  ✅ Present
├── data/             ⚠️ Empty (only .gitkeep)
└── models/           ⚠️ Empty (only .gitkeep)
```

### After Running `python main.py → 1.1`
```
RetailForecasting/
├── data/
│   ├── raw/base_dataset.csv         ✅ Generated
│   ├── processed/base_processed.csv ✅ Generated
│   └── user/                        ✅ Ready for input
├── models/
│   ├── base_model.pkl               ✅ Generated
│   └── model_metadata.json          ✅ Generated
└── [All code files]                 ✅ Unchanged
```

---

## Best Practices

### Do ✅
- Push source code changes regularly
- Push documentation updates
- Update requirements.txt if adding packages
- Include .gitignore in first commit
- Use meaningful commit messages

### Don't ❌
- Push large CSV files (use environment or data loading from external sources)
- Push model files (.pkl, .joblib)
- Push __pycache__ or .pyc files
- Push virtual environment folders
- Push IDE configuration files
- Push secrets or credentials

### If Users Need Initial Data
- Provide a data download script
- Or document data generation step
- Or use an external data source URL

---

## File Structure for GitHub

### Recommended Directory Tree
```
RetailForecasting/
├── README.md          ✅ Main documentation
├── QUICKSTART.md      ✅ Setup guide
├── ARCHITECTURE.md    ✅ Technical documentation
├── FILE_INVENTORY.md  ✅ File reference
├── requirements.txt   ✅ Dependencies
├── .gitignore         ✅ Git configuration
│
├── src/               ✅ Source code
│   ├── __init__.py
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── utils/
│   └── pipeline/
│
├── app/               ✅ Streamlit UI
│   └── app.py
│
├── main.py            ✅ CLI
├── examples.py        ✅ Examples
│
├── data/              ⚠️ Empty (ignored by git)
│   ├── .gitkeep
│   ├── raw/
│   ├── user/
│   └── processed/
│
├── models/            ⚠️ Empty (ignored by git)
│   └── .gitkeep
│
└── notebooks/         ⚠️ Empty (ignored by git)
    └── .gitkeep
```

---

## Sample First Push

```bash
# From project root
git init
git add .                    # Respects .gitignore
git status                   # Verify what's included
git commit -m "Initial commit: AI Retail Forecasting System v1.0"
git remote add origin https://github.com/username/RetailForecasting.git
git branch -M main
git push -u origin main
```

---

## CI/CD Consideration (Future)

If you add GitHub Actions for testing:
- Create `.github/workflows/` directory
- Add test automation
- Still respect .gitignore for data/models

Example:
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/  # (if you add tests)
```

---

## ✅ Summary

**Created:**
- ✅ `.gitignore` - Excludes data, models, cache
- ✅ `.gitkeep` files - Preserve directory structure
- ✅ This guide - For GitHub deployment

**Ready to push to GitHub:**
- ✅ 13 Python modules
- ✅ 3 UI/CLI files
- ✅ 1 requirements.txt
- ✅ 5 documentation files
- ✅ .gitignore and .gitkeep

**Total:** ~280 KB of clean, production-ready code

---

**Next Step:** Run these commands:

```bash
cd C:\Users\Raja\Desktop\Sales_Prediction\RetailForecasting
git init
git add .
git status                    # Verify
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/RetailForecasting.git
git push -u origin main
```

Good to push! 🚀
