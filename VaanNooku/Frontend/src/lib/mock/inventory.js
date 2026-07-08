export const INVENTORY_ITEMS = [
  { name: "Milk", category: "Perishables", stock: 85, cap: 200, minStock: 25, rop: 40, eoq: 120, risk: "Low", price: 28, margin: 0.12 },
  { name: "Curd", category: "Perishables", stock: 12, cap: 80, minStock: 15, rop: 25, eoq: 50, risk: "High", price: 45, margin: 0.15 },
  { name: "Paneer", category: "Perishables", stock: 18, cap: 40, minStock: 8, rop: 12, eoq: 20, risk: "Medium", price: 280, margin: 0.2 },
  { name: "Bread", category: "Perishables", stock: 60, cap: 100, minStock: 15, rop: 30, eoq: 60, risk: "Low", price: 35, margin: 0.1 },
  { name: "Eggs", category: "Perishables", stock: 8, cap: 150, minStock: 20, rop: 35, eoq: 100, risk: "High", price: 45, margin: 0.08 },
  { name: "Rice", category: "Staples & Grains", stock: 340, cap: 500, minStock: 80, rop: 120, eoq: 250, risk: "Low", price: 65, margin: 0.18 },
  { name: "Wheat Flour", category: "Staples & Grains", stock: 210, cap: 400, minStock: 60, rop: 100, eoq: 200, risk: "Low", price: 45, margin: 0.15 },
  { name: "Oil", category: "Staples & Grains", stock: 45, cap: 150, minStock: 30, rop: 50, eoq: 100, risk: "Medium", price: 160, margin: 0.1 },
  { name: "Sugar", category: "Staples & Grains", stock: 95, cap: 250, minStock: 40, rop: 65, eoq: 150, risk: "Low", price: 50, margin: 0.12 },
  { name: "Dal", category: "Staples & Grains", stock: 75, cap: 200, minStock: 35, rop: 55, eoq: 120, risk: "Medium", price: 85, margin: 0.14 },
  { name: "Biscuits", category: "Snacks & Biscuits", stock: 140, cap: 300, minStock: 40, rop: 75, eoq: 150, risk: "Low", price: 40, margin: 0.22 },
  { name: "Chips", category: "Snacks & Biscuits", stock: 180, cap: 250, minStock: 30, rop: 60, eoq: 120, risk: "Low", price: 50, margin: 0.25 },
  { name: "Namkeen", category: "Snacks & Biscuits", stock: 22, cap: 150, minStock: 25, rop: 45, eoq: 90, risk: "High", price: 80, margin: 0.28 },
  { name: "Tea", category: "Beverages", stock: 85, cap: 120, minStock: 20, rop: 35, eoq: 80, risk: "Low", price: 250, margin: 0.22 },
  { name: "Coffee", category: "Beverages", stock: 15, cap: 80, minStock: 15, rop: 25, eoq: 50, risk: "High", price: 180, margin: 0.25 },
  { name: "Soft Drinks", category: "Beverages", stock: 90, cap: 150, minStock: 25, rop: 45, eoq: 100, risk: "Low", price: 45, margin: 0.18 },
  { name: "Soap", category: "Personal Care", stock: 120, cap: 200, minStock: 25, rop: 40, eoq: 100, risk: "Low", price: 35, margin: 0.3 },
  { name: "Shampoo", category: "Personal Care", stock: 28, cap: 80, minStock: 15, rop: 25, eoq: 40, risk: "Medium", price: 120, margin: 0.32 },
];

export const BULK_CONSOLIDATION_DISCOUNT = 0.15;

export const BULK_CONSOLIDATION_ITEMS = [
  { name: "Curd", "balaji-store": 13, "shiva-stores": 25, "surya-markets": 12, unit: "units", price: 45 },
  { name: "Eggs", "balaji-store": 27, "shiva-stores": 50, "surya-markets": 35, unit: "crates", price: 45 },
  { name: "Oil", "balaji-store": 15, "shiva-stores": 30, "surya-markets": 25, unit: "bottles", price: 160 },
  { name: "Namkeen", "balaji-store": 23, "shiva-stores": 45, "surya-markets": 30, unit: "packs", price: 80 },
];
