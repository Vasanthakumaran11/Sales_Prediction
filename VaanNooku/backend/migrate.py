"""
migrate.py — Run once to add admin columns to Supabase stores table.
"""
import sys, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from src.config.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE stores ADD COLUMN IF NOT EXISTS admin_name  VARCHAR"))
    conn.execute(text("ALTER TABLE stores ADD COLUMN IF NOT EXISTS admin_email VARCHAR"))
    conn.execute(text("ALTER TABLE stores ADD COLUMN IF NOT EXISTS admin_phone VARCHAR"))
    conn.execute(text("ALTER TABLE stores ADD COLUMN IF NOT EXISTS admin_role  VARCHAR DEFAULT 'Store Owner'"))
    conn.commit()
    print("Migration complete: admin columns added to stores table.")
