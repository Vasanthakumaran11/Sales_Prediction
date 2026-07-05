// Mock data representing the 6-layer model configuration
export const PRODUCT_CATALOG = {
  "Perishables": {
    "Milk": { price: 28, leadTime: 1, turnover: "High", margin: 0.12 },
    "Curd": { price: 45, leadTime: 2, turnover: "High", margin: 0.15 },
    "Paneer": { price: 280, leadTime: 2, turnover: "Medium", margin: 0.20 },
    "Bread": { price: 35, leadTime: 1, turnover: "High", margin: 0.10 },
    "Eggs": { price: 45, leadTime: 1, turnover: "High", margin: 0.08 },
  },
  "Non-Perishables": {
    "Rice": { price: 65, leadTime: 3, turnover: "High", margin: 0.18 },
    "Wheat Flour": { price: 45, leadTime: 3, turnover: "High", margin: 0.15 },
    "Oil": { price: 160, leadTime: 2, turnover: "High", margin: 0.10 },
    "Sugar": { price: 50, leadTime: 3, turnover: "Medium", margin: 0.12 },
    "Salt": { price: 20, leadTime: 5, turnover: "Low", margin: 0.25 },
    "Dal": { price: 85, leadTime: 3, turnover: "High", margin: 0.14 },
    "Spices Mix": { price: 150, leadTime: 4, turnover: "Medium", margin: 0.30 },
  },
  "Snacks & Biscuits": {
    "Biscuits": { price: 40, leadTime: 2, turnover: "High", margin: 0.22 },
    "Chips": { price: 50, leadTime: 2, turnover: "High", margin: 0.25 },
    "Namkeen": { price: 80, leadTime: 3, turnover: "Medium", margin: 0.28 },
    "Chocolate": { price: 60, leadTime: 2, turnover: "High", margin: 0.20 },
    "Cookies": { price: 50, leadTime: 2, turnover: "High", margin: 0.24 },
  },
  "Beverages": {
    "Tea": { price: 250, leadTime: 3, turnover: "High", margin: 0.22 },
    "Coffee": { price: 180, leadTime: 3, turnover: "Medium", margin: 0.25 },
    "Juice": { price: 80, leadTime: 2, turnover: "Medium", margin: 0.20 },
    "Soft Drinks": { price: 45, leadTime: 2, turnover: "High", margin: 0.18 },
  },
  "Frozen Foods": {
    "Ice Cream": { price: 120, leadTime: 1, turnover: "Medium", margin: 0.28 },
    "Frozen Vegetables": { price: 100, leadTime: 1, turnover: "Low", margin: 0.22 },
  },
  "Personal Care": {
    "Soap": { price: 35, leadTime: 3, turnover: "High", margin: 0.30 },
    "Shampoo": { price: 120, leadTime: 3, turnover: "Medium", margin: 0.32 },
    "Toothpaste": { price: 80, leadTime: 3, turnover: "High", margin: 0.25 },
    "Deodorant": { price: 150, leadTime: 3, turnover: "Low", margin: 0.35 },
  }
};

export const FESTIVALS = [
  { name: "Pongal", month: "January", modifier: 1.30, description: "Harvest festival - Heavy demand for staples & non-perishables" },
  { name: "Ramzan", month: "March", modifier: 1.25, description: "Holy month - High demand for beverages, snacks & frozen foods" },
  { name: "Diwali", month: "November", modifier: 1.40, description: "Festival of Lights - Massive surge in snacks, beverages & premium oils" },
  { name: "Christmas", month: "December", modifier: 1.20, description: "Year-end holidays - Higher demand for snacks, ice cream & beverages" }
];

export const STORE_PROFILES = [
  {
    id: "balaji_store",
    name: "Balaji Store",
    type: "Supermarket",
    location: "Urban",
    investment: 850000,
    openingMonth: "October",
    activeDays: 45,
    metrics: {
      r2: "0.932",
      wasteMargin: "2.4%",
      stockouts: 2,
      leakingMargin: "4.8%",
      deficitCount: 3,
      revenue: 294200,
      inventoryValue: 185000
    }
  },
  {
    id: "shiva_stores",
    name: "Shiva Stores",
    type: "Medium",
    location: "Semi-Urban",
    investment: 300000,
    openingMonth: "January",
    activeDays: 20,
    metrics: {
      r2: "0.918",
      wasteMargin: "4.1%",
      stockouts: 8,
      leakingMargin: "7.2%",
      deficitCount: 6,
      revenue: 120500,
      inventoryValue: 92000
    }
  },
  {
    id: "surya_markets",
    name: "Surya Markets",
    type: "Small",
    location: "Rural",
    investment: 90000,
    openingMonth: "June",
    activeDays: 12,
    metrics: {
      r2: "0.895",
      wasteMargin: "8.5%",
      stockouts: 14,
      leakingMargin: "11.3%",
      deficitCount: 9,
      revenue: 41200,
      inventoryValue: 28400
    }
  }
];

export const ALGORITHMS = [
  { name: "Random Forest", r2: 0.9306, mae: 11.4568, rmse: 16.3197, type: "Tree-based Ensemble", isBest: true },
  { name: "XGBoost", r2: 0.9281, mae: 11.6659, rmse: 16.6181, type: "Gradient Boosting", isBest: false },
  { name: "LightGBM", r2: 0.9260, mae: 11.7396, rmse: 16.8597, type: "Gradient Boosting", isBest: false },
  { name: "Linear Regression", r2: 0.9178, mae: 12.7521, rmse: 17.7659, type: "Regression", isBest: false },
  { name: "Decision Tree", r2: 0.8689, mae: 15.2916, rmse: 22.4395, type: "Tree-based", isBest: false },
  { name: "KNN Model", r2: 0.8154, mae: 18.5088, rmse: 26.6222, type: "Instance-based", isBest: false },
  { name: "SVR Model", r2: 0.7893, mae: 17.0347, rmse: 28.4443, type: "Kernel-SVM", isBest: false }
];

export const INVENTORY_ITEMS = [
  { name: "Milk", category: "Perishables", stock: 85, cap: 200, minStock: 25, rop: 40, eoq: 120, risk: "Low", price: 28, margin: 0.12 },
  { name: "Curd", category: "Perishables", stock: 12, cap: 80, minStock: 15, rop: 25, eoq: 50, risk: "High", price: 45, margin: 0.15 },
  { name: "Paneer", category: "Perishables", stock: 18, cap: 40, minStock: 8, rop: 12, eoq: 20, risk: "Medium", price: 280, margin: 0.20 },
  { name: "Bread", category: "Perishables", stock: 60, cap: 100, minStock: 15, rop: 30, eoq: 60, risk: "Low", price: 35, margin: 0.10 },
  { name: "Eggs", category: "Perishables", stock: 8, cap: 150, minStock: 20, rop: 35, eoq: 100, risk: "High", price: 45, margin: 0.08 },
  { name: "Rice", category: "Non-Perishables", stock: 340, cap: 500, minStock: 80, rop: 120, eoq: 250, risk: "Low", price: 65, margin: 0.18 },
  { name: "Wheat Flour", category: "Non-Perishables", stock: 210, cap: 400, minStock: 60, rop: 100, eoq: 200, risk: "Low", price: 45, margin: 0.15 },
  { name: "Oil", category: "Non-Perishables", stock: 45, cap: 150, minStock: 30, rop: 50, eoq: 100, risk: "Medium", price: 160, margin: 0.10 },
  { name: "Sugar", category: "Non-Perishables", stock: 95, cap: 250, minStock: 40, rop: 65, eoq: 150, risk: "Low", price: 50, margin: 0.12 },
  { name: "Salt", category: "Non-Perishables", stock: 110, cap: 150, minStock: 20, rop: 35, eoq: 80, risk: "Low", price: 20, margin: 0.25 },
  { name: "Dal", category: "Non-Perishables", stock: 75, cap: 200, minStock: 35, rop: 55, eoq: 120, risk: "Medium", price: 85, margin: 0.14 },
  { name: "Spices Mix", category: "Non-Perishables", stock: 32, cap: 100, minStock: 20, rop: 35, eoq: 60, risk: "Medium", price: 150, margin: 0.30 },
  { name: "Biscuits", category: "Snacks & Biscuits", stock: 140, cap: 300, minStock: 40, rop: 75, eoq: 150, risk: "Low", price: 40, margin: 0.22 },
  { name: "Chips", category: "Snacks & Biscuits", stock: 180, cap: 250, minStock: 30, rop: 60, eoq: 120, risk: "Low", price: 50, margin: 0.25 },
  { name: "Namkeen", category: "Snacks & Biscuits", stock: 22, cap: 150, minStock: 25, rop: 45, eoq: 90, risk: "High", price: 80, margin: 0.28 },
  { name: "Chocolate", category: "Snacks & Biscuits", stock: 95, cap: 200, minStock: 30, rop: 50, eoq: 100, risk: "Low", price: 60, margin: 0.20 },
  { name: "Tea", category: "Beverages", stock: 85, cap: 120, minStock: 20, rop: 35, eoq: 80, risk: "Low", price: 250, margin: 0.22 },
  { name: "Coffee", category: "Beverages", stock: 15, cap: 80, minStock: 15, rop: 25, eoq: 50, risk: "High", price: 180, margin: 0.25 },
  { name: "Juice", category: "Beverages", stock: 45, cap: 100, minStock: 15, rop: 30, eoq: 60, risk: "Low", price: 80, margin: 0.20 },
  { name: "Soft Drinks", category: "Beverages", stock: 90, cap: 150, minStock: 25, rop: 45, eoq: 100, risk: "Low", price: 45, margin: 0.18 },
  { name: "Soap", category: "Personal Care", stock: 120, cap: 200, minStock: 25, rop: 40, eoq: 100, risk: "Low", price: 35, margin: 0.30 },
  { name: "Shampoo", category: "Personal Care", stock: 28, cap: 80, minStock: 15, rop: 25, eoq: 40, risk: "Medium", price: 120, margin: 0.32 },
  { name: "Toothpaste", category: "Personal Care", stock: 55, cap: 120, minStock: 20, rop: 35, eoq: 70, risk: "Low", price: 80, margin: 0.25 },
  { name: "Deodorant", category: "Personal Care", stock: 12, cap: 60, minStock: 10, rop: 18, eoq: 30, risk: "High", price: 150, margin: 0.35 }
];
