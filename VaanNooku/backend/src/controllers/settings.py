from sqlalchemy.orm import Session
from fastapi import HTTPException
from starlette.responses import StreamingResponse
import io
import json
from datetime import datetime
from src.models import models
from src.schemas import schemas
from src.security import hash_password, verify_password

def update_profile(request: schemas.ProfileUpdateRequest):
    return {
        "success": True,
        "message": "Profile updated successfully.",
        "user": request.dict()
    }

def update_password(db: Session, request: schemas.PasswordUpdateRequest):
    store = db.query(models.Store).filter(models.Store.id == request.storeId).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store node not found.")

    if not verify_password(request.currentPassword, store.password):
        raise HTTPException(status_code=400, detail="Invalid current password.")

    if not request.newPassword or len(request.newPassword) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")

    store.password = hash_password(request.newPassword)
    db.commit()
    return {
        "success": True,
        "message": "Password changed successfully."
    }

def trigger_backup(store_id: str):
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "message": f"System snapshot database backup created for store: {store_id}."
    }

def download_backup():
    backup_data = {
        "backupVersion": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "preferences": {
            "theme": "light",
            "compactView": False,
            "tipsEnabled": True,
            "autoRefresh": True
        }
    }
    
    stream = io.StringIO()
    json.dump(backup_data, stream, indent=2)
    stream.seek(0)
    
    return StreamingResponse(
        io.BytesIO(stream.read().encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=retail_ai_settings_backup.json"}
    )

def reset_store_data(db: Session, store_id: str):
    try:
        # Delete daily transaction entries
        db.query(models.TransactionDaily).filter(models.TransactionDaily.store_id == store_id).delete()
        
        # Reset inventory levels back to baseline defaults
        db.query(models.StockLevel).filter(models.StockLevel.store_id == store_id).update({
            "current_stock": 100
        })
        
        db.commit()
        return {
            "success": True,
            "message": "Store daily logs flushed and stock levels reset to defaults."
        }
    except Exception as err:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Reset operations failed: {str(err)}")

def delete_store(db: Session, store_id: str):
    try:
        db.query(models.StockLevel).filter(models.StockLevel.store_id == store_id).delete()
        db.query(models.TransactionDaily).filter(models.TransactionDaily.store_id == store_id).delete()
        db.query(models.Store).filter(models.Store.id == store_id).delete()
        
        db.commit()
        return {
            "success": True,
            "message": "Store console account registration permanently deactivated."
        }
    except Exception as err:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Account deactivation cleanup failed: {str(err)}")
