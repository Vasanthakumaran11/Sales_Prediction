"""
migrate_hash_passwords.py — Run ONCE after deploying bcrypt-based auth.

Existing rows (e.g. seed_store.py's STORE_001, or anything registered before
this change) may have plaintext or null passwords. Since login now verifies
against a bcrypt hash, those stores would be locked out otherwise. This
hashes any password that isn't already a bcrypt hash, in place.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from src.config.database import SessionLocal
from src.models import models
from src.security import hash_password, is_hashed

db = SessionLocal()
try:
    stores = db.query(models.Store).all()
    migrated = 0
    skipped_no_password = 0

    for store in stores:
        if not store.password:
            print(f"[skip] {store.id}: no password set — cannot migrate, must reset manually.")
            skipped_no_password += 1
            continue
        if is_hashed(store.password):
            continue
        store.password = hash_password(store.password)
        migrated += 1
        print(f"[migrated] {store.id}")

    db.commit()
    print(f"\nDone. Migrated {migrated} store password(s). {skipped_no_password} store(s) had no password set.")
finally:
    db.close()
