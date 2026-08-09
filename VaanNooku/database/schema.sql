-- ============================================================
-- VaanNooku database schema (PostgreSQL / Supabase)
--
-- This file is generated FROM the SQLAlchemy models at
-- backend/src/models/models.py, which remain the source of truth — the
-- backend creates/updates these tables automatically via
-- Base.metadata.create_all() on startup (see backend/src/main.py). This
-- file exists so the schema is readable without running Python, and so
-- database/views.sql has documented tables to reference.
-- ============================================================

CREATE TABLE IF NOT EXISTS stores (
    id             VARCHAR PRIMARY KEY,
    name           VARCHAR NOT NULL,
    password       VARCHAR,              -- bcrypt hash (see backend/src/security.py)
    type           VARCHAR DEFAULT 'Supermarket',  -- Small | Medium | Supermarket
    location       VARCHAR DEFAULT 'Urban',        -- Urban | Semi-Urban | Rural
    investment     FLOAT NOT NULL,
    opening_month  VARCHAR NOT NULL,
    months_active  INTEGER DEFAULT 12,
    admin_name     VARCHAR,
    admin_email    VARCHAR,
    admin_phone    VARCHAR,
    admin_role     VARCHAR DEFAULT 'Store Owner'
);

CREATE TABLE IF NOT EXISTS suppliers (
    id             VARCHAR PRIMARY KEY,
    name           VARCHAR NOT NULL,
    category       VARCHAR NOT NULL,
    lead_time_days INTEGER DEFAULT 3,
    min_order_qty  INTEGER DEFAULT 10,
    phone          VARCHAR,
    email          VARCHAR
);

CREATE TABLE IF NOT EXISTS products (
    id             VARCHAR PRIMARY KEY,
    name           VARCHAR NOT NULL,
    sku            VARCHAR UNIQUE NOT NULL,
    category       VARCHAR NOT NULL,
    brand          VARCHAR,
    cost_price     FLOAT NOT NULL,
    selling_price  FLOAT NOT NULL,
    supplier_id    VARCHAR REFERENCES suppliers(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS stock_levels (
    store_id       VARCHAR REFERENCES stores(id) ON DELETE CASCADE,
    product_id     VARCHAR REFERENCES products(id) ON DELETE CASCADE,
    current_stock  INTEGER DEFAULT 0,
    safety_stock   INTEGER DEFAULT 15,
    reorder_point  INTEGER DEFAULT 30,
    PRIMARY KEY (store_id, product_id)
);

CREATE TABLE IF NOT EXISTS transactions_daily (
    id                 VARCHAR PRIMARY KEY,
    store_id           VARCHAR NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
    date               VARCHAR NOT NULL,
    transaction_count  INTEGER DEFAULT 0,
    gross_sales        FLOAT DEFAULT 0.0,
    discount_amount    FLOAT DEFAULT 0.0,
    net_sales          FLOAT DEFAULT 0.0,
    payment_cash       FLOAT DEFAULT 0.0,
    payment_upi        FLOAT DEFAULT 0.0,
    payment_card       FLOAT DEFAULT 0.0,
    audit_status       VARCHAR DEFAULT 'Synced & Closed'  -- Synced & Closed | Draft
);

CREATE TABLE IF NOT EXISTS transaction_items (
    id              VARCHAR PRIMARY KEY,
    transaction_id  VARCHAR NOT NULL REFERENCES transactions_daily(id) ON DELETE CASCADE,
    product_id      VARCHAR NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    qty             INTEGER NOT NULL,
    discount        FLOAT DEFAULT 0.0
);

-- Ops/staff console accounts — intentionally separate from `stores` so a
-- merchant login can never be mistaken for admin access.
CREATE TABLE IF NOT EXISTS admin_users (
    id        VARCHAR PRIMARY KEY,
    email     VARCHAR UNIQUE NOT NULL,
    password  VARCHAR NOT NULL,          -- bcrypt hash
    role      VARCHAR DEFAULT 'admin'    -- admin | ml_engineer | support
);

CREATE TABLE IF NOT EXISTS complaint_tickets (
    id           VARCHAR PRIMARY KEY,
    store_id     VARCHAR NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
    subject      VARCHAR NOT NULL,
    description  VARCHAR,
    status       VARCHAR DEFAULT 'Open',  -- Open | In Progress | Resolved
    created_at   VARCHAR NOT NULL
);

-- Read-optimized views for reporting/BI — see views.sql and
-- docs/powerbi-integration.md.
\i views.sql
