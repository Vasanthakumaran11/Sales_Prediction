"""
seed_store.py
=============
Run once to:
1. Delete demo stores (balaji-store, shiva-stores, surya-markets) + all cascaded records
2. Create STORE_001 (Vasanthakumaran Store) with all 20 products from the CSV dataset
3. Seed 31 days of realistic May 2026 transactions (May 1-31 2026)

Run from backend directory:
    python seed_store.py
"""

import sys, os, random, uuid
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env so DATABASE_URL is available
load_dotenv(Path(__file__).parent / ".env")

# Add src to path for SQLAlchemy models
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.models.models import Base, Store, Supplier, Product, StockLevel, TransactionDaily, TransactionItem
from src.security import hash_password

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found in .env")

engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
db = Session()

# =============================================================
# CATALOG — exactly as in retailai_finalized_dataset.csv
# =============================================================
PRODUCTS = [
    ("ITM01", "Milk",        "Perishable",     29.86, 24.49, "SUP_FRESH",    1),
    ("ITM02", "Curd",        "Perishable",     38.04, 30.43, "SUP_FRESH",    1),
    ("ITM03", "Paneer",      "Perishable",     89.61, 69.90, "SUP_FRESH",    2),
    ("ITM04", "Bread",       "Perishable",     43.53, 35.26, "SUP_FRESH",    1),
    ("ITM05", "Eggs",        "Perishable",     76.27, 63.30, "SUP_FRESH",    2),
    ("ITM06", "Rice",        "Non-Perishable", 72.84, 61.91, "SUP_STAPLE",   5),
    ("ITM07", "Wheat Flour", "Non-Perishable", 48.56, 41.28, "SUP_STAPLE",   4),
    ("ITM08", "Oil",         "Non-Perishable",173.81,149.48, "SUP_STAPLE",   6),
    ("ITM09", "Sugar",       "Non-Perishable", 50.29, 43.75, "SUP_STAPLE",   4),
    ("ITM10", "Salt",        "Non-Perishable", 23.52, 18.82, "SUP_STAPLE",   4),
    ("ITM11", "Dal",         "Non-Perishable",152.56,128.15, "SUP_STAPLE",   5),
    ("ITM12", "Spices Mix",  "Non-Perishable", 90.48, 65.15, "SUP_PACKAGED", 7),
    ("ITM13", "Biscuits",    "Non-Perishable", 10.84,  8.13, "SUP_PACKAGED", 3),
    ("ITM14", "Chips",       "Non-Perishable", 20.77, 15.16, "SUP_PACKAGED", 3),
    ("ITM15", "Namkeen",     "Non-Perishable", 25.98, 19.74, "SUP_PACKAGED", 3),
    ("ITM16", "Chocolate",   "Non-Perishable", 43.20, 30.24, "SUP_PACKAGED", 3),
    ("ITM17", "Tea",         "Non-Perishable",124.38, 97.02, "SUP_STAPLE",   5),
    ("ITM18", "Coffee",      "Non-Perishable",159.86,119.90, "SUP_STAPLE",   5),
    ("ITM19", "Juice",       "Non-Perishable", 65.25, 52.20, "SUP_PACKAGED", 3),
    ("ITM20", "Soft Drinks", "Non-Perishable", 41.42, 33.96, "SUP_PACKAGED", 2),
]

SUPPLIERS = [
    ("SUP_FRESH",    "Fresh Produce Suppliers",    "Perishable"),
    ("SUP_STAPLE",   "Staple Foods Wholesalers",   "Non-Perishable"),
    ("SUP_PACKAGED", "Packaged Goods Distributors","Packaged"),
]

STORE_ID     = "STORE_001"
STORE_NAME   = "Vasanthakumaran Store"
INVESTMENT   = 85000.0
STORE_TYPE   = "Small"
LOCATION     = "Urban"
OPENING_MONTH= "January"
MONTHS_ACTIVE= 17   # as per dataset (Jan 2025 context)

# Average daily demand from dataset (ITM01..20) — used to generate realistic qtys
# Derived from CSV mean Units_Sold per item across all dates
DEMAND_MEAN = {
    "ITM01": 26, "ITM02": 22, "ITM03": 12, "ITM04": 18, "ITM05": 20,
    "ITM06": 30, "ITM07": 25, "ITM08": 14, "ITM09": 20, "ITM10": 15,
    "ITM11": 16, "ITM12":  8, "ITM13": 45, "ITM14": 35, "ITM15": 28,
    "ITM16": 20, "ITM17": 18, "ITM18": 12, "ITM19": 22, "ITM20": 30,
}

# =============================================================
# STEP 1 — DELETE demo stores (cascades automatically)
# =============================================================
DEMO_IDS = ["balaji-store", "shiva-stores", "surya-markets",
            "balaji", "shiva", "surya"]

print("==> Step 1: Deleting demo stores...")
for did in DEMO_IDS:
    store = db.query(Store).filter(Store.id == did).first()
    if store:
        db.delete(store)
        print(f"    Deleted store: {did}")
db.commit()
print("    Done.\n")

# =============================================================
# STEP 2 — Delete STORE_001 if exists (fresh start)
# =============================================================
print("==> Step 2: Resetting STORE_001 if exists...")
existing = db.query(Store).filter(Store.id == STORE_ID).first()
if existing:
    db.delete(existing)
    db.commit()
    print("    Existing STORE_001 deleted.\n")
else:
    print("    STORE_001 not found — creating fresh.\n")

# =============================================================
# STEP 3 — Create suppliers
# =============================================================
print("==> Step 3: Seeding suppliers...")
for sup_id, sup_name, sup_cat in SUPPLIERS:
    s = db.query(Supplier).filter(Supplier.id == sup_id).first()
    if not s:
        s = Supplier(id=sup_id, name=sup_name, category=sup_cat,
                     lead_time_days=2, min_order_qty=10)
        db.add(s)
        print(f"    Created supplier: {sup_id}")
    else:
        print(f"    Supplier already exists: {sup_id}")
db.commit()
print()

# =============================================================
# STEP 4 — Create the new store
# =============================================================
print("==> Step 4: Creating STORE_001...")
new_store = Store(
    id           = STORE_ID,
    name         = STORE_NAME,
    password     = hash_password("vaannooku123"),
    type         = STORE_TYPE,
    location     = LOCATION,
    investment   = INVESTMENT,
    opening_month= OPENING_MONTH,
    months_active= MONTHS_ACTIVE,
    admin_name   = "Vasanthakumaran",
    admin_email  = "vasantha@vaannookustore.com",
    admin_phone  = "+91 98765 43210",
    admin_role   = "Store Owner",
)
db.add(new_store)
db.commit()
print(f"    Created: {STORE_ID} — {STORE_NAME}\n")

# =============================================================
# STEP 5 — Create products + stock levels
# =============================================================
print("==> Step 5: Seeding 20 products + stock levels...")
for item_id, item_name, category, sell_price, cost_price, sup_id, _ in PRODUCTS:
    # Product
    p = db.query(Product).filter(Product.id == item_id).first()
    if not p:
        sku = f"SKU-{item_id}"
        p = Product(
            id           = item_id,
            name         = item_name,
            sku          = sku,
            category     = category,
            cost_price   = cost_price,
            selling_price= sell_price,
            supplier_id  = sup_id,
        )
        db.add(p)

    # Stock Level — initial stock = 15 days demand
    daily_demand = DEMAND_MEAN.get(item_id, 20)
    init_stock   = daily_demand * 15
    sl = db.query(StockLevel).filter(
        StockLevel.store_id == STORE_ID,
        StockLevel.product_id == item_id
    ).first()
    if not sl:
        sl = StockLevel(
            store_id     = STORE_ID,
            product_id   = item_id,
            current_stock= init_stock,
            safety_stock = int(daily_demand * 3),
            reorder_point= int(daily_demand * 7),
        )
        db.add(sl)
    print(f"    {item_id} {item_name:12s} — stock: {init_stock}")
db.commit()
print()

# =============================================================
# STEP 6 — Seed 31 days of May 2026 transactions
# =============================================================
print("==> Step 6: Seeding May 2026 transactions (May 1–31)...")

random.seed(42)   # reproducible

# May 2026 festivals / salary days in India
FESTIVAL_DAYS = {5, 9}    # Eid ~May 5, Mother's Day ~May 10 (approx)
SALARY_DAYS   = {1, 5, 20}

for day_num in range(1, 32):
    tx_date   = date(2026, 5, day_num)
    dow       = tx_date.weekday()         # 0=Mon … 6=Sun
    is_weekend= 1 if dow >= 5 else 0
    is_salary = 1 if day_num in SALARY_DAYS else 0
    is_festival=1 if day_num in FESTIVAL_DAYS else 0

    # Demand multiplier: weekend +15%, salary day +20%, festival +30%
    mult = 1.0 + 0.15*is_weekend + 0.20*is_salary + 0.30*is_festival

    tx_id = f"TX-STORE001-2026-05-{day_num:02d}"

    # Build items for this day
    items_for_tx = []
    gross_sales  = 0.0
    discount_amt = 0.0

    for item_id, _, _, sell_price, _, _, _ in PRODUCTS:
        base    = DEMAND_MEAN[item_id]
        noise   = random.gauss(0, base * 0.15)      # ±15% std
        qty     = max(1, round((base + noise) * mult))
        disc_pct= random.choice([0, 0, 0, 5, 10])  # 60% no discount
        disc_val= round(sell_price * qty * disc_pct / 100, 2)

        items_for_tx.append({
            "item_id"   : item_id,
            "qty"       : qty,
            "sell_price": sell_price,
            "discount"  : disc_val,
        })
        gross_sales += sell_price * qty
        discount_amt+= disc_val

    net_sales = gross_sales - discount_amt

    # Payment split: ~50% cash, 35% UPI, 15% card
    pay_cash = round(net_sales * (0.45 + random.uniform(-0.05, 0.05)), 2)
    pay_upi  = round(net_sales * (0.35 + random.uniform(-0.03, 0.03)), 2)
    pay_card = round(net_sales - pay_cash - pay_upi, 2)

    # Insert TransactionDaily
    existing_tx = db.query(TransactionDaily).filter(TransactionDaily.id == tx_id).first()
    if existing_tx:
        db.delete(existing_tx)
        db.commit()

    tx = TransactionDaily(
        id               = tx_id,
        store_id         = STORE_ID,
        date             = tx_date.isoformat(),
        transaction_count= len(PRODUCTS),
        gross_sales      = round(gross_sales, 2),
        discount_amount  = round(discount_amt, 2),
        net_sales        = round(net_sales, 2),
        payment_cash     = pay_cash,
        payment_upi      = pay_upi,
        payment_card     = pay_card,
        audit_status     = "Synced & Closed",
    )
    db.add(tx)
    db.flush()   # flush to get tx.id FK for items

    # Insert TransactionItems
    for it in items_for_tx:
        ti = TransactionItem(
            id            = f"{tx_id}-{it['item_id']}",
            transaction_id= tx_id,
            product_id    = it["item_id"],
            qty           = it["qty"],
            discount      = it["discount"],
        )
        db.add(ti)

    db.commit()
    print(f"    May {day_num:02d} - gross Rs. {gross_sales:,.0f}  net Rs. {net_sales:,.0f}  "
          f"{'[weekend]' if is_weekend else ''}{'[salary]' if is_salary else ''}{'[festival]' if is_festival else ''}")

print()
print("=" * 60)
print("==> Seeding complete!")
print(f"   Store  : {STORE_ID} - {STORE_NAME}")
print(f"   Products: {len(PRODUCTS)}")
print(f"   Transactions: 31 days (May 1 - 31 2026)")
print()
print("Login credentials:")
print(f"   Username: {STORE_ID}")
print(f"   Password: vaannooku123")
print("=" * 60)

db.close()
