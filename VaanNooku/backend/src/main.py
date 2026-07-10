import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.database import engine, Base
from src.routes.api import router as api_router
from src.models import models

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

# Register API Router
app.include_router(api_router, prefix="/api")

@app.get("/health")
def check_health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=5000, reload=True)
