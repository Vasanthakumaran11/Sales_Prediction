"use client";

import React, { useState } from "react";
import { Sparkles, Package, HelpCircle, ArrowRight, TrendingUp, DollarSign, BarChart3, Layers, CheckCircle } from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";

export default function AIPredictions() {
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [showPredictions, setShowPredictions] = useState(false);

  // Mock catalog categories and items list
  const categories = [
    {
      id: "staples",
      name: "Staples & Grains",
      itemCount: 4,
      totalStock: 380,
      color: "bg-sky-50 border-sky-200 text-sky-700",
      products: [
        { name: "India Gate Basmati Rice 1kg", sku: "IG-RICE-1KG", stock: 120 },
        { name: "Aashirvaad Atta 5kg", sku: "AASH-ATTA-5KG", stock: 60 },
        { name: "Sugar Premium 1kg", sku: "SUG-1KG", stock: 150 },
        { name: "Moong Dal 1kg", sku: "DAL-MOONG-1", stock: 50 },
      ],
    },
    {
      id: "dairy",
      name: "Dairy & Bakery",
      itemCount: 3,
      totalStock: 395,
      color: "bg-blue-50 border-blue-200 text-blue-700",
      products: [
        { name: "Amul Taaza Milk 1L", sku: "AMUL-MILK-1L", stock: 245 },
        { name: "Amul Salted Butter 100g", sku: "AMUL-BUTTER-100", stock: 90 },
        { name: "Britannia Bread Family Pack", sku: "BRIT-BREAD-F", stock: 60 },
      ],
    },
    {
      id: "beverages",
      name: "Beverages",
      itemCount: 3,
      totalStock: 150,
      color: "bg-teal-50 border-teal-200 text-teal-700",
      products: [
        { name: "Tata Tea Premium 250g", sku: "TATA-TEA-250", stock: 0 },
        { name: "Nescafe Gold 100g", sku: "NES-GOLD-100", stock: 45 },
        { name: "Coca Cola 1.25L", sku: "COKE-1.25L", stock: 105 },
      ],
    },
    {
      id: "household",
      name: "Household Essentials",
      itemCount: 2,
      totalStock: 110,
      color: "bg-indigo-50 border-indigo-200 text-indigo-700",
      products: [
        { name: "Surf Excel Matic 1kg", sku: "SURF-1KG", stock: 35 },
        { name: "Vim Liquid Soap 500ml", sku: "VIM-500ML", stock: 75 },
      ],
    },
  ];

  const handlePredict = () => {
    setPredicting(true);
    setTimeout(() => {
      setPredicting(false);
      setShowPredictions(true);
    }, 1500);
  };

  return (
    <div className="space-y-6 font-sans px-6">
      <PageHeader
        title="AI Demand & Sales Predictions"
        subtitle="Ensemble machine learning projection maps for next month's retail turnover."
        icon={Sparkles}
      />

      {/* Top Phase: Category Stock Counter Boxes */}
      <div className="space-y-3">
        <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider font-serif">
          Current Inventory Categories
        </h3>
        <p className="text-xs text-slate-500 font-sans">Click on any category to view available products and counts.</p>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          {categories.map((cat) => (
            <div
              key={cat.id}
              onClick={() => setSelectedCategory(cat)}
              className={`p-5 rounded-2xl border transition-all hover:scale-[1.02] cursor-pointer shadow-sm flex flex-col justify-between h-32 ${cat.color}`}
            >
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider block opacity-70">Category</span>
                <span className="text-sm font-bold block font-serif mt-1">{cat.name}</span>
              </div>

              <div className="flex justify-between items-center text-xs border-t border-current/15 pt-3.5 font-sans">
                <span>Products: <strong>{cat.itemCount} SKUs</strong></span>
                <span>Stock: <strong>{cat.totalStock} units</strong></span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Modal Popup */}
      {selectedCategory && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white border border-sky-100 rounded-2xl max-w-lg w-full shadow-2xl p-6 relative animate-fade-in font-sans">
            <div className="flex justify-between items-start pb-4 border-b border-slate-100">
              <div>
                <h3 className="text-base font-bold text-slate-900 font-serif">{selectedCategory.name}</h3>
                <p className="text-[10px] text-slate-400 font-sans mt-0.5">List of items and current stock ledger.</p>
              </div>
              <button
                onClick={() => setSelectedCategory(null)}
                className="text-slate-400 hover:text-slate-600 font-bold text-xs"
              >
                ✕ Close
              </button>
            </div>

            <div className="py-4 space-y-2.5 max-h-60 overflow-y-auto">
              {selectedCategory.products.map((p, idx) => (
                <div key={idx} className="flex justify-between items-center text-xs py-1 border-b border-slate-50 last:border-b-0">
                  <div>
                    <span className="font-semibold text-slate-800 block">{p.name}</span>
                    <span className="text-[9px] text-slate-450 uppercase">{p.sku}</span>
                  </div>
                  <span
                    className={`font-bold px-2 py-0.5 rounded text-[10px] ${
                      p.stock === 0
                        ? "bg-rose-50 text-rose-600"
                        : p.stock < 50
                        ? "bg-amber-50 text-amber-600"
                        : "bg-sky-50 text-sky-600"
                    }`}
                  >
                    {p.stock} Units
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Bottom Phase: Next Month Predictions */}
      <Card className="p-6 space-y-6 bg-sky-50/20 border-sky-100 flex flex-col justify-between">
        <div className="flex justify-between items-center pb-3 border-b border-slate-100">
          <div>
            <h3 className="text-sm font-bold text-slate-905 uppercase tracking-wider font-serif">
              Project Sales for Next Month
            </h3>
            <p className="text-xs text-slate-500 font-sans mt-1">Execute the Hybrid Regression forecast engine using existing logs.</p>
          </div>
          <button
            onClick={handlePredict}
            disabled={predicting}
            className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-lg transition-all shadow disabled:opacity-50"
          >
            {predicting ? "Running ML Models..." : "Predict Next Month Sales"} <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {predicting && (
          <div className="py-8 flex flex-col items-center justify-center text-center gap-3">
            <div className="w-8 h-8 rounded-full border-2 border-sky-200 border-t-blue-600 animate-spin" />
            <span className="text-xs text-slate-500 font-bold">Aggregating daily transaction records & seasonal factors...</span>
          </div>
        )}

        {showPredictions && !predicting && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in font-sans">
            {/* Forecast Metrics */}
            <div className="space-y-4 md:col-span-1 flex flex-col justify-between h-full">
              <div className="bg-white border border-sky-100 p-4 rounded-xl shadow-sm flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center">
                  <DollarSign className="w-4 h-4" />
                </div>
                <div>
                  <span className="block text-[9px] text-slate-400 font-bold uppercase tracking-wider">Projected Revenue</span>
                  <span className="text-base font-black text-slate-900 leading-none">₹9,45,230.50</span>
                </div>
              </div>

              <div className="bg-white border border-sky-100 p-4 rounded-xl shadow-sm flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center">
                  <Package className="w-4 h-4" />
                </div>
                <div>
                  <span className="block text-[9px] text-slate-400 font-bold uppercase tracking-wider">Projected Demand</span>
                  <span className="text-base font-black text-slate-900 leading-none">5,842 Units</span>
                </div>
              </div>

              <div className="bg-white border border-sky-100 p-4 rounded-xl shadow-sm flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center">
                  <TrendingUp className="w-4 h-4" />
                </div>
                <div>
                  <span className="block text-[9px] text-slate-400 font-bold uppercase tracking-wider">Model Confidence</span>
                  <span className="text-base font-black text-slate-900 leading-none">94.8% R²</span>
                </div>
              </div>
            </div>

            {/* Visual SVG chart */}
            <div className="md:col-span-2 bg-white border border-sky-100 rounded-xl p-4 shadow-sm flex flex-col justify-between h-full min-h-[220px]">
              <div>
                <span className="text-[10px] text-slate-400 font-bold uppercase block tracking-wider">4-Week Forward Trend</span>
                <span className="text-[8px] text-slate-500">Weekly sales projections matching historical seasonality</span>
              </div>

              <div className="h-32 w-full mt-3 relative flex items-end">
                <svg viewBox="0 0 100 40" className="w-full h-full">
                  <line x1="0" y1="35" x2="100" y2="35" className="stroke-slate-100" strokeWidth="0.5" />
                  <line x1="0" y1="20" x2="100" y2="20" className="stroke-slate-100" strokeWidth="0.5" strokeDasharray="1 1" />
                  <line x1="0" y1="5" x2="100" y2="5" className="stroke-slate-100" strokeWidth="0.5" strokeDasharray="1 1" />
                  
                  {/* Projected curve */}
                  <path d="M 5 32 Q 25 24 50 15 T 95 8" fill="none" stroke="#2563eb" strokeWidth="2.5" strokeDasharray="1.5 1" />
                  <circle cx="5" cy="32" r="1.5" fill="#2563eb" />
                  <circle cx="35" cy="22" r="1.5" fill="#2563eb" />
                  <circle cx="65" cy="14" r="1.5" fill="#2563eb" />
                  <circle cx="95" cy="8" r="1.5" fill="#2563eb" />

                  <text x="5" y="38" className="fill-slate-400" fontSize="3">Week 1</text>
                  <text x="35" y="38" className="fill-slate-400" fontSize="3">Week 2</text>
                  <text x="65" y="38" className="fill-slate-400" fontSize="3">Week 3</text>
                  <text x="90" y="38" className="fill-slate-400" fontSize="3">Week 4</text>
                </svg>
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
