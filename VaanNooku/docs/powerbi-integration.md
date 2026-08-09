# PowerBI Integration Guide

VaanNooku's database is Postgres (hosted on Supabase). PowerBI connects to it
directly using its native PostgreSQL connector — no new backend code or API
is required for this. This document covers the two SQL views to create
first, the connection steps, and the Import-vs-DirectQuery tradeoff.

## 1. Create the reporting views

The real schema lives in `backend/src/models/models.py` (auto-created via
SQLAlchemy on backend startup); `database/schema.sql` mirrors it for
reference. Rather than pointing PowerBI at raw tables, run
[`database/views.sql`](../database/views.sql) once against the database
first:

```bash
psql "$DATABASE_URL" -f database/views.sql
```

(Or paste its contents into the Supabase SQL editor.) This creates:

- **`vw_inventory_status`** — current stock, safety stock, reorder point, and
  a computed `status` (`Out of Stock` / `Low Stock` / `Healthy`) per
  store/product. Backed entirely by existing tables.
- **`vw_sales_with_predictions`** — flattened historical sales, one row per
  transaction line item, joined with store and product details.

Model your PowerBI relationships against these **views**, not the raw
tables — that way, if a column in `models.py` gets renamed later, only the
view definition needs updating, not every downstream report.

> **Known gap — predictions aren't persisted yet.** `predict_next_month`
> (the ML ensemble forecast) runs on-demand and its output is never written
> to a table, so `vw_sales_with_predictions` can only carry *historical*
> sales today, not live forecasts. If you need forecasts inside PowerBI
> reports, the clean fix is a small `prediction_cache` table populated
> whenever `/api/predictions/next-month/*` runs, then joined into the view.
> Treat this as an optional follow-up — it wasn't required for the views
> above to be useful.

## 2. Connect PowerBI Desktop to the database

1. **Get Data → Database → PostgreSQL database.**
2. **Server**: use the host from `backend/.env`'s `DATABASE_URL`, e.g.
   `db.<project-ref>.supabase.co` (direct connection) — see the note on
   pooled vs. direct connections below.
3. **Database**: `postgres`.
4. **Credentials**: the Postgres database user/password from the same
   `DATABASE_URL` (**not** `SUPABASE_ANON_KEY` — that key is for Supabase's
   JS/REST client SDK and is irrelevant to PowerBI's native Postgres
   connector, which authenticates as a real Postgres role).
5. Once connected, select `vw_inventory_status` and `vw_sales_with_predictions`
   from the table/view picker (they'll appear alongside the raw tables).

## 3. Import vs. DirectQuery

VaanNooku's `backend/.env` currently points at Supabase's **direct**
connection (port `5432`), not the pooled connection (port `6543`,
PgBouncer transaction-mode) mentioned in `backend/.env_example`. That
matters for PowerBI:

- **Import mode (recommended default)**: PowerBI copies the data into its
  own storage and refreshes on a schedule (hourly/daily). This works
  reliably against either connection type and is the safest default for
  this dataset size.
- **DirectQuery**: PowerBI queries live on every report interaction. If your
  environment is on the pooled connection string, be aware PgBouncer's
  transaction-mode pooling has known compatibility gaps with some
  DirectQuery query patterns (prepared statements in particular). If you
  need real-time BI, use the **direct** (non-pooled) connection string for
  DirectQuery, and check Supabase's connection-limit for your plan tier
  first — PowerBI will hold its own persistent connections against that cap.

For VaanNooku's use case (daily sales entry, not high-frequency trading),
**Import mode with a scheduled refresh** is the recommended setup.

## 4. Suggested first reports

- **Inventory health** — bar chart of `status` counts from
  `vw_inventory_status`, filterable by `store_name` / `category`.
- **Sales trend** — `line_net_sales` over `date` from
  `vw_sales_with_predictions`, sliced by `store_name` and `category`.
- **Store comparison** — total `line_net_sales` by `store_name`, useful once
  more than one real (non-demo) store is registered.
