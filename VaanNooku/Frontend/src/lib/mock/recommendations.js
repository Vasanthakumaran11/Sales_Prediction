// Baseline SKU recommendations, always relevant regardless of season.
export const BASELINE_RECOMMENDATIONS = [
  { name: "Tata Tea Premium 250g", category: "Beverages", reason: "High base turnover staple item." },
  { name: "Fortune Sunflower Oil 1L", category: "Staples & Grains", reason: "Everyday kitchen necessity." },
  { name: "Amul Salted Butter 100g", category: "Perishables", reason: "Consistent cold-storage demand." },
  { name: "Aashirvaad Chakki Atta 5kg", category: "Staples & Grains", reason: "Daily kitchen staple item." },
];

// Festival-specific SKU boosts, keyed by festival name.
export const FESTIVAL_RECOMMENDATIONS = {
  Diwali: [
    { name: "Haldiram's Bhujia 400g", category: "Snacks & Biscuits", reason: "High-demand festival launch item." },
    { name: "Cadbury Celebrations Gift Pack", category: "Snacks & Biscuits", reason: "High-margin seasonal gifting SKU." },
  ],
  Christmas: [
    { name: "Cadbury Celebrations Gift Pack", category: "Snacks & Biscuits", reason: "High-margin seasonal gifting SKU." },
  ],
  Pongal: [
    { name: "Kolam Rice Premium 5kg", category: "Staples & Grains", reason: "Traditional harvest-festival staple." },
    { name: "Madhur Pure Sugar 1kg", category: "Staples & Grains", reason: "Volume driver for festival sweet prep." },
  ],
};
