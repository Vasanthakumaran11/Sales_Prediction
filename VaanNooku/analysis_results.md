# Analysis Report: Frontend Static Data & Integration Gaps

This document identifies all occurrences of static data fallbacks, hardcoded mock values, and integration gaps between the Next.js React frontend and the Python FastAPI/PostgreSQL backend database.

---

## 1. Authentication & Onboarding Gaps

### 1.1 Signup Registration Missing Password Field
*   **File**: [Gateway.js](file:///c:/Users/Raja/Desktop/Sales_Prediction/VaanNooku/Frontend/src/components/gateway/Gateway.js#L350-L435)
*   **Logical Flaw**: The store registration onboarding wizard (Step 1) gathers `storeName`, `storeType`, `locationType`, `investment`, and `openingMonth` but **does not capture or set a password**.
*   **Implication**: New stores are registered with blank or uninitialized passwords, making it impossible for them to log in again.

### 1.2 Login Form Bypasses Backend API
*   **File**: [Gateway.js](file:///c:/Users/Raja/Desktop/Sales_Prediction/VaanNooku/Frontend/src/components/gateway/Gateway.js#L105-L117)
*   **Logical Flaw**: Submitting the login form executes `handleLoginSubmit`, which searches for a match inside the static mock profile array `STORE_PROFILES` and logs the user in immediately. It does not call any API endpoint (`POST /api/auth/login`).
*   **Implication**: Users cannot log in using credentials registered in the database; the application only recognizes the pre-seeded mock stores.

---

## 2. Dynamic Data & Dashboard Disconnects

### 2.1 Daily Sales Data Entry Screen
*   **File**: [DataEntryView.js](file:///c:/Users/Raja/Desktop/Sales_Prediction/VaanNooku/Frontend/src/components/dashboard/DataEntryView.js)
*   **Logical Flaw**:
    *   **KPI Statistics**: Summary tiles (Total Sales, Total Quantity, Average Bill) are calculated using hardcoded values and static factors instead of aggregating transaction database records.
    *   **Fallback Grid Rows**: Product items default to a static list of prefilled rows (e.g., Milk, Namkeen) instead of pulling from the dynamic product catalog database (`GET /api/stores/:storeId/products`).

### 2.2 Historical Sales Ledger
*   **File**: [HistoryView.js](file:///c:/Users/Raja/Desktop/Sales_Prediction/VaanNooku/Frontend/src/components/history/HistoryView.js)
*   **Logical Flaw**:
    *   **Prefilled Table Rows**: Displays a hardcoded ledger of transactions dated May 11, 2026, through May 17, 2026, using fixed numbers.
    *   **Database Disconnection**: The page does not perform fetch requests to retrieve actual historical transactions logged in the database (`GET /api/stores/:storeId/daily-logs`).

### 2.3 Store Product Catalog Listing
*   **File**: [ProductsView.js](file:///c:/Users/Raja/Desktop/Sales_Prediction/VaanNooku/Frontend/src/components/products/ProductsView.js)
*   **Logical Flaw**: The items table loads mock catalog products with static mock status tags ("Healthy", "Low Stock") instead of requesting live items list states from the database.

---

## 3. Analytics & Prediction Charts

### 3.1 Sales Analytics Page
*   **File**: [SalesAnalytics.js](file:///c:/Users/Raja/Desktop/Sales_Prediction/VaanNooku/Frontend/src/components/analytics/SalesAnalytics.js)
*   **Logical Flaw**:
    *   **Static Charts**: Renders charts (Revenue trends, Category profit allocations) using prefilled arrays or math functions simulating cycles rather than reading historical user sales logs.
    *   **Empty State Missing**: Shows detailed curves immediately on new stores, even when no transactions have been logged.

### 3.2 AI Predictions Page
*   **File**: [AIPredictions.js](file:///c:/Users/Raja/Desktop/Sales_Prediction/VaanNooku/Frontend/src/components/predictions/AIPredictions.js)
*   **Logical Flaw**:
    *   **Hardcoded Stock Category Boxes**: Category boxes (Staples, Beverages, Perishables) display static count indicators rather than calling the predictions API (`GET /api/stores/:storeId/predictions/stock-summary`).
    *   **Regression Plots**: Forecast graphs plot fixed trendlines instead of fetching the machine learning prediction outputs (`POST /api/stores/:storeId/predictions/forecast`).

---

## 4. Suppliers Directory

### 4.1 Wholesale Suppliers Page
*   **File**: [SuppliersView.js](file:///c:/Users/Raja/Desktop/Sales_Prediction/VaanNooku/Frontend/src/components/suppliers/SuppliersView.js)
*   **Logical Flaw**: Loads static suppliers (e.g. Balaji Agro, Shiva Dairy) from a local mock list instead of querying the backend (`GET /api/suppliers/suggestions`).
