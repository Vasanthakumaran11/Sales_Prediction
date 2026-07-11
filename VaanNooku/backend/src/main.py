import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.database import engine, Base
from src.routes.api import router as api_router
from src.routes.prediction_routes import router as pred_router
from src.models import models

# Add backend root to sys.path so `ml` package is importable
_backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

# Auto-compile table structures if they do not exist
print("Compiling PostgreSQL tables schema using SQLAlchemy ORM engine...")
Base.metadata.create_all(bind=engine)
print("PostgreSQL tables successfully built and verified.")

app = FastAPI(
    title="RetailAI Predictive Console Backend",
    description="Relational FastAPI + PostgreSQL demand predictions backend server",
    version="1.0.0"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(api_router, prefix="/api")
app.include_router(pred_router)  # already has /api/predictions prefix

@app.on_event("startup")
def startup_event():
    """Load ML models once at startup — not per-request."""
    try:
        from ml.model_loader import load_all
        load_all()
        print("[startup] ML models loaded and ready.")
    except Exception as e:
        print(f"[startup] WARNING: ML models could not be loaded: {e}")
        print("[startup] Prediction endpoints will return rule-based fallback.")

@app.get("/health")
def check_health():
    return {"status": "healthy", "port": 5000}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=5000, reload=True)
