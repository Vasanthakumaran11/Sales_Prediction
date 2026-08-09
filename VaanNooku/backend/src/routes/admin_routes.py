from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.schemas import schemas
from src.controllers import admin, admin_auth
from src.deps import get_current_admin
from src.models import models

router = APIRouter(prefix="/api/admin", tags=["Admin AI Console"])


@router.post("/auth/login")
def login(request: schemas.AdminLoginRequest, db: Session = Depends(get_db)):
    return admin_auth.login_admin(db, request.email, request.password)


# ---- Model status / monitoring ----
@router.get("/models/status")
def model_status(_current_admin: models.AdminUser = Depends(get_current_admin)):
    return admin.get_model_status()


@router.post("/models/reload")
def reload_models(_current_admin: models.AdminUser = Depends(get_current_admin)):
    return admin.reload_models()


# ---- Datasets ----
@router.get("/datasets")
def list_datasets(_current_admin: models.AdminUser = Depends(get_current_admin)):
    return admin.list_datasets()


@router.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...), _current_admin: models.AdminUser = Depends(get_current_admin)):
    return admin.upload_dataset(file)


# ---- Retraining ----
@router.post("/retraining/trigger")
def trigger_retraining(request: schemas.RetrainingTriggerRequest, _current_admin: models.AdminUser = Depends(get_current_admin)):
    return admin.trigger_retraining(request.datasetFilename)


@router.get("/retraining/status/{job_id}")
def retraining_status(job_id: str, _current_admin: models.AdminUser = Depends(get_current_admin)):
    return admin.get_job_status(job_id)


# ---- Complaints ----
@router.get("/complaints")
def list_complaints(db: Session = Depends(get_db), _current_admin: models.AdminUser = Depends(get_current_admin)):
    return admin.list_complaints(db)


@router.post("/complaints")
def create_complaint(request: schemas.ComplaintCreateRequest, db: Session = Depends(get_db)):
    # Called from the merchant-facing app, not the admin console — no admin auth required,
    # a store just needs to be a real registered store id.
    return admin.create_complaint(db, request.storeId, request.subject, request.description)


@router.put("/complaints/{ticket_id}")
def update_complaint(ticket_id: str, request: schemas.ComplaintStatusUpdateRequest, db: Session = Depends(get_db),
                      _current_admin: models.AdminUser = Depends(get_current_admin)):
    return admin.update_complaint_status(db, ticket_id, request.status)
