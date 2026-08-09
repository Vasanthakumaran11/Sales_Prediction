-- ============================================================
-- Read-optimized views for BI tools (PowerBI, etc.)
--
-- database/schema.sql has no content — the real schema lives only in the
-- SQLAlchemy models at backend/src/models/models.py. These views are the
-- stable contract BI tools should build against, so future column renames
-- in models.py don't ripple into every downstream report.
--
-- Run once against the Supabase Postgres database (via `psql` or the
-- Supabase SQL editor). See docs/powerbi-integration.md for the full
-- connection guide.
-- ============================================================

-- Current stock position per store/product, with a computed health status.
-- Backed entirely by existing tables — no schema changes required.
CREATE OR REPLACE VIEW vw_inventory_status AS
SELECT
    sl.store_id,
    s.name              AS store_name,
    s.type              AS store_type,
    s.location          AS store_location,
    p.id                AS product_id,
    p.name              AS product_name,
    p.sku,
    p.category,
    p.cost_price,
    p.selling_price,
    sl.current_stock,
    sl.safety_stock,
    sl.reorder_point,
    CASE
        WHEN sl.current_stock = 0 THEN 'Out of Stock'
        WHEN sl.current_stock < sl.reorder_point THEN 'Low Stock'
        ELSE 'Healthy'
    END AS status
FROM stock_levels sl
JOIN stores s   ON s.id = sl.store_id
JOIN products p ON p.id = sl.product_id;

-- Flattened historical sales, one row per (transaction, line item). Carries
-- real sales only — there is currently no table persisting ML prediction
-- outputs (predict_next_month runs on-demand, nothing writes its result to
-- the DB), so predictions can't be joined in here yet. See the "Optional
-- follow-up" note in docs/powerbi-integration.md if live prediction history
-- is needed in BI reports later.
CREATE OR REPLACE VIEW vw_sales_with_predictions AS
SELECT
    td.id               AS transaction_id,
    td.store_id,
    s.name              AS store_name,
    s.type              AS store_type,
    s.location          AS store_location,
    td.date,
    p.id                AS product_id,
    p.name              AS product_name,
    p.category,
    ti.qty              AS units_sold,
    ti.discount,
    p.selling_price,
    (ti.qty * p.selling_price) - ti.discount AS line_net_sales,
    td.payment_cash,
    td.payment_upi,
    td.payment_card,
    td.audit_status
FROM transactions_daily td
JOIN transaction_items ti ON ti.transaction_id = td.id
JOIN products p           ON p.id = ti.product_id
JOIN stores s              ON s.id = td.store_id;
