from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.controllers.predictions import get_next_month_prediction

router = APIRouter(prefix="/api/predictions", tags=["ML Predictions"])

@router.get("/next-month/{store_id}/{item_id}")
def next_month_forecast(store_id: str, item_id: str, db: Session = Depends(get_db)):
    """
    Run a 30-day ensemble ML demand forecast for a specific store + product item.
    Returns daily forecast, monthly totals, and inventory business metrics.
    """
    return get_next_month_prediction(db, store_id, item_id)