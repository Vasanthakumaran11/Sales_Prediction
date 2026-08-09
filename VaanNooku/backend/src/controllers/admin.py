import csv
import json
import time
import uuid
import threading
from pathlib import Path
from datetime import datetime, timezone

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from src.models import models

ML_WORKSPACE_DIR = Path(__file__).resolve().parents[3] / "ml_workspace"
METRICS_DIR = ML_WORKSPACE_DIR / "metrics"
DATASETS_DIR = ML_WORKSPACE_DIR / "datasets"
UPLOADS_DIR = DATASETS_DIR / "uploads"

REQUIRED_DATASET_COLUMNS = {
    "Store_ID", "Item_ID", "Date", "Category", "Store_Type", "Location_Type",
    "Supplier_ID", "Units_Sold", "Units_Stocked", "Units_Remaining", "Unit_Price"
}

# In-memory job registry — fine for a v1 single-process admin tool; a real
# task queue (Celery/RQ) is the correct upgrade once retraining needs to run
# without contending with live prediction traffic on the same process.
_jobs: dict[str, dict] = {}


# ============================================================
# MODEL STATUS
# ============================================================
def get_model_status():
    ensemble_path = METRICS_DIR / "ensemble_metrics.json"
    comparison_path = METRICS_DIR / "all_model_comparison.csv"

    ensemble = json.loads(ensemble_path.read_text()) if ensemble_path.exists() else None

    comparison = []
    if comparison_path.exists():
        with open(comparison_path, newline="") as f:
            comparison = list(csv.DictReader(f))

    return {
        "ensemble": ensemble,
        "modelComparison": comparison,
        "lastUpdated": datetime.fromtimestamp(
            comparison_path.stat().st_mtime, tz=timezone.utc
        ).isoformat() if comparison_path.exists() else None,
    }


def reload_models():
    from ml.model_loader import load_all
    load_all(force=True)
    return {"success": True, "message": "ML models reloaded from disk."}


# ============================================================
# DATASETS
# ============================================================
def list_datasets():
    if not DATASETS_DIR.exists():
        return []
    files = []
    for p in DATASETS_DIR.rglob("*.csv"):
        stat = p.stat()
        files.append({
            "filename": str(p.relative_to(DATASETS_DIR)),
            "sizeBytes": stat.st_size,
            "modifiedAt": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    return sorted(files, key=lambda f: f["modifiedAt"], reverse=True)


def upload_dataset(file: UploadFile):
    import pandas as pd
    import io

    contents = file.file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {err}")

    missing = REQUIRED_DATASET_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset is missing required columns for training: {sorted(missing)}"
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / (file.filename or f"dataset-{int(time.time())}.csv")
    dest.write_bytes(contents)

    return {"success": True, "filename": str(dest.relative_to(DATASETS_DIR)), "rows": len(df)}


# ============================================================
# RETRAINING
# ============================================================
def _run_job(job_id: str, dataset_filename: str | None):
    from ml.train import run_training

    _jobs[job_id]["status"] = "running"
    try:
        dataset_path = str(DATASETS_DIR / dataset_filename) if dataset_filename else None
        result = run_training(dataset_path)
        _jobs[job_id]["status"] = "success"
        _jobs[job_id]["result"] = result
    except Exception as err:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(err)


def trigger_retraining(dataset_filename: str | None):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "jobId": job_id,
        "status": "queued",
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "result": None,
        "error": None,
    }
    thread = threading.Thread(target=_run_job, args=(job_id, dataset_filename), daemon=True)
    thread.start()
    return _jobs[job_id]


def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Retraining job not found.")
    return job


# ============================================================
# COMPLAINTS
# ============================================================
def list_complaints(db: Session):
    return db.query(models.ComplaintTicket).order_by(models.ComplaintTicket.created_at.desc()).all()


def create_complaint(db: Session, store_id: str, subject: str, description: str | None):
    store = db.query(models.Store).filter(models.Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found.")

    ticket = models.ComplaintTicket(
        id=f"CMP-{uuid.uuid4().hex[:10]}",
        store_id=store_id,
        subject=subject,
        description=description,
        status="Open",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


def update_complaint_status(db: Session, ticket_id: str, status: str):
    ticket = db.query(models.ComplaintTicket).filter(models.ComplaintTicket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Complaint ticket not found.")
    ticket.status = status
    db.commit()
    db.refresh(ticket)
    return ticket
