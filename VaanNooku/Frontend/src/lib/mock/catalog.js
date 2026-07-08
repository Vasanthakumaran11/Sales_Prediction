export const PRODUCT_CATALOG = {
  Perishables: {
    Milk: { price: 28, leadTime: 1, turnover: "High", margin: 0.12 },
    Curd: { price: 45, leadTime: 2, turnover: "High", margin: 0.15 },
    Paneer: { price: 280, leadTime: 2, turnover: "Medium", margin: 0.2 },
    Bread: { price: 35, leadTime: 1, turnover: "High", margin: 0.1 },
    Eggs: { price: 45, leadTime: 1, turnover: "High", margin: 0.08 },
  },
  "Staples & Grains": {
    Rice: { price: 65, leadTime: 3, turnover: "High", margin: 0.18 },
    "Wheat Flour": { price: 45, leadTime: 3, turnover: "High", margin: 0.15 },
    Oil: { price: 160, leadTime: 2, turnover: "High", margin: 0.1 },
    Sugar: { price: 50, leadTime: 3, turnover: "Medium", margin: 0.12 },
    Salt: { price: 20, leadTime: 5, turnover: "Low", margin: 0.25 },
    Dal: { price: 85, leadTime: 3, turnover: "High", margin: 0.14 },
    "Spices Mix": { price: 150, leadTime: 4, turnover: "Medium", margin: 0.3 },
  },
  "Snacks & Biscuits": {
    Biscuits: { price: 40, leadTime: 2, turnover: "High", margin: 0.22 },
    Chips: { price: 50, leadTime: 2, turnover: "High", margin: 0.25 },
    Namkeen: { price: 80, leadTime: 3, turnover: "Medium", margin: 0.28 },
    Chocolate: { price: 60, leadTime: 2, turnover: "High", margin: 0.2 },
  },
  Beverages: {
    Tea: { price: 250, leadTime: 3, turnover: "High", margin: 0.22 },
    Coffee: { price: 180, leadTime: 3, turnover: "Medium", margin: 0.25 },
    Juice: { price: 80, leadTime: 2, turnover: "Medium", margin: 0.2 },
    "Soft Drinks": { price: 45, leadTime: 2, turnover: "High", margin: 0.18 },
  },
  "Personal Care": {
    Soap: { price: 35, leadTime: 3, turnover: "High", margin: 0.3 },
    Shampoo: { price: 120, leadTime: 3, turnover: "Medium", margin: 0.32 },
    Toothpaste: { price: 80, leadTime: 3, turnover: "High", margin: 0.25 },
  },
};

export function flattenCatalog() {
  return Object.entries(PRODUCT_CATALOG).flatMap(([category, items]) =>
    Object.entries(items).map(([name, detail]) => ({ name, category, ...detail }))
  );
}

export const FESTIVALS = [
  { name: "Pongal", month: "January", modifier: 1.3, description: "Harvest festival — heavy demand for staples & non-perishables." },
  { name: "Ramzan", month: "March", modifier: 1.25, description: "Holy month — high demand for beverages, snacks & frozen foods." },
  { name: "Diwali", month: "November", modifier: 1.4, description: "Festival of Lights — surge in snacks, beverages & premium oils." },
  { name: "Christmas", month: "December", modifier: 1.2, description: "Year-end holidays — higher demand for snacks, ice cream & beverages." },
];

export function getFestivalForMonth(month) {
  return FESTIVALS.find((f) => f.month.toLowerCase() === String(month).toLowerCase()) ?? null;
}
