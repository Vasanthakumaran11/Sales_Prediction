"""
seed_admin.py — Run once to create the first ops/admin console account.
Change the email/password below (or pass them as env vars) before running
against a real environment.

Run from backend directory:
    python seed_admin.py
"""
import sys, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from src.config.database import SessionLocal
from src.models import models
from src.security import hash_password

ADMIN_EMAIL = os.environ.get("SEED_ADMIN_EMAIL", "admin@vaannooku.com")
ADMIN_PASSWORD = os.environ.get("SEED_ADMIN_PASSWORD", "changeme123")

db = SessionLocal()
try:
    existing = db.query(models.AdminUser).filter(models.AdminUser.email.ilike(ADMIN_EMAIL)).first()
    if existing:
        print(f"Admin account already exists: {ADMIN_EMAIL}")
    else:
        admin = models.AdminUser(
            id=f"admin-{ADMIN_EMAIL.split('@')[0]}",
            email=ADMIN_EMAIL,
            password=hash_password(ADMIN_PASSWORD),
            role="admin",
        )
        db.add(admin)
        db.commit()
        print(f"Created admin account: {ADMIN_EMAIL}")
        print(f"Password: {ADMIN_PASSWORD}  (change this after first login)")
finally:
    db.close()
