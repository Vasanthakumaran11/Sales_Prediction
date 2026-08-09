from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.schemas import schemas
from src.controllers import auth, sales, inventory, predictions, settings
from src.models import models
from src.deps import get_current_store, require_matching_store

router = APIRouter()

# Authentication & Registration routes
@router.post("/auth/login")
def login(request: schemas.LoginRequest, db: Session = Depends(get_db)):
    return auth.login_store(db, request)

@router.post("/stores/register")
def register(request: schemas.StoreRegisterRequest, db: Session = Depends(get_db)):
    return auth.register_store(db, request)

# Suggested wholesale suppliers directory suggestions
@router.get("/suppliers/suggestions")
def get_supplier_suggestions(db: Session = Depends(get_db)):
    suppliers = db.query(models.Supplier).all()
    return suppliers

# Plan spreadsheet CSV upload parser
@router.post("/catalog/parse-plan")
def parse_plan(file: UploadFile = File(...)):
    return sales.parse_sales_csv(file)

# Daily Sales Ledger postings
@router.post("/stores/{store_id}/daily-log")
def submit_daily_log(store_id: str, request: schemas.DailyLogSubmitRequest, db: Session = Depends(get_db),
                      _current_store: models.Store = Depends(require_matching_store)):
    return sales.submit_log(db, store_id, request)

@router.get("/stores/{store_id}/daily-logs")
def get_daily_logs(store_id: str, db: Session = Depends(get_db)):
    return sales.get_logs(db, store_id)

# Inventory Stock Replenishments
@router.get("/stores/{store_id}/inventory")
def get_store_inventory(store_id: str, db: Session = Depends(get_db)):
    return inventory.get_inventory(db, store_id)

@router.get("/inventory/bulk-consolidation")
def get_consolidated_shortages(db: Session = Depends(get_db)):
    return inventory.get_bulk_consolidation(db)

@router.post("/inventory/bulk-consolidation/orders")
def order_bulk_inventory(request: schemas.POOrderRequest):
    return inventory.place_bulk_order(request)

# Catalog Products Routing
@router.get("/stores/{store_id}/products")
def get_catalog_products(
    store_id: str, 
    category: str = None, 
    search: str = None, 
    db: Session = Depends(get_db)
):
    return inventory.get_products(db, store_id, category, search)

@router.post("/stores/{store_id}/products")
def add_new_product(store_id: str, request: schemas.SkuAddRequest, db: Session = Depends(get_db),
                     _current_store: models.Store = Depends(require_matching_store)):
    return inventory.add_product(db, store_id, request)

# ML Forecast Metrics Routing
@router.get("/stores/{store_id}/predictions/stock-summary")
def get_category_stock_summary(store_id: str, db: Session = Depends(get_db)):
    return predictions.get_stock_summary(db, store_id)

@router.post("/stores/{store_id}/predictions/forecast")
def get_demand_forecast(store_id: str, db: Session = Depends(get_db)):
    return predictions.get_forecast(db, store_id)

# Preferences & Account Settings Routing
@router.put("/users/profile")
def update_profile_settings(request: schemas.ProfileUpdateRequest,
                             _current_store: models.Store = Depends(get_current_store)):
    return settings.update_profile(request)

@router.put("/users/password")
def update_password_settings(request: schemas.PasswordUpdateRequest, db: Session = Depends(get_db),
                              current_store: models.Store = Depends(get_current_store)):
    if current_store.id != request.storeId:
        raise HTTPException(status_code=403, detail="Not authorized to change this store's password.")
    return settings.update_password(db, request)

@router.post("/stores/{store_id}/backup")
def backup_system_configuration(store_id: str, _current_store: models.Store = Depends(require_matching_store)):
    return settings.trigger_backup(store_id)

@router.get("/stores/{store_id}/backup/download")
def download_backup_file():
    return settings.download_backup()

@router.post("/stores/{store_id}/reset")
def reset_database_logs(store_id: str, db: Session = Depends(get_db),
                         _current_store: models.Store = Depends(require_matching_store)):
    return settings.reset_store_data(db, store_id)

@router.delete("/stores/{store_id}")
def delete_store_account(store_id: str, db: Session = Depends(get_db),
                          _current_store: models.Store = Depends(require_matching_store)):
    return settings.delete_store(db, store_id)
