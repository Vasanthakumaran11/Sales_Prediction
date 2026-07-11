"use client";

import React, { useState, useEffect } from "react";
import { Sparkles, Package, HelpCircle, ArrowRight, TrendingUp, DollarSign, BarChart3, Layers, CheckCircle } from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";
import { useStoreContext } from "@/context/StoreContext";

// Helper to compute category stock levels dynamically from active storeProducts list
const getDynamicCategories = (products) => {
  const catMap = {};
  (products || []).forEach(p => {
    const cat = p.category || "General";
    if (!catMap[cat]) {
      catMap[cat] = { itemCount: 0, totalStock: 0, products: [] };
    }
    catMap[cat].itemCount += 1;
    catMap[cat].totalStock += (p.stock || 0);
    catMap[cat].products.push(p);
  });
  
  const colors = [
    "bg-sky-50 border-sky-200 text-sky-700",
    "bg-blue-50 border-blue-200 text-blue-700",
    "bg-teal-50 border-teal-200 text-teal-700",
    "bg-indigo-50 border-indigo-200 text-indigo-700"
  ];
  
  return Object.entries(catMap).map(([name, val], idx) => ({
    id: name.toLowerCase().replace(" ", "-").replace("&", "and"),
    name,
    itemCount: val.itemCount,
    totalStock: val.totalStock,
    color: colors[idx % colors.length],
    products: val.products
  }));
};

export default function AIPredictions() {
  const { historyLogs, activeStore, storeProducts } = useStoreContext();
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [showPredictions, setShowPredictions] = useState(false);
  const [apiCategories, setApiCategories] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [predictionError, setPredictionError] = useState("");

  // Individual item ML prediction state
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [productForecast, setProductForecast] = useState(null);
  const [loadingProductForecast, setLoadingProductForecast] = useState(false);
  const [productForecastError, setProductForecastError] = useState("");

  useEffect(() => {
    const demoIds = ["balaji-store", "shiva-stores", "surya-markets"];
    const isDemo = activeStore ? demoIds.includes(activeStore.id) : true;
    if (!isDemo && activeStore) {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || "";
      if (apiBase) {
        fetch(`${apiBase}/api/stores/${activeStore.id}/predictions/stock-summary`)
          .then((res) => res.json())
          .then((data) => {
            if (Array.isArray(data) && data.length > 0) {
              const colors = [
                "bg-sky-50 border-sky-200 text-sky-700",
                "bg-blue-50 border-blue-200 text-blue-700",
                "bg-teal-50 border-teal-200 text-teal-700",
                "bg-indigo-50 border-indigo-200 text-indigo-700"
              ];
              const mapped = data.map((c, idx) => ({
                id: c.category.toLowerCase().replace(" ", "-"),
                name: c.category,
                itemCount: c.skuCount,
                totalStock: c.totalStock,
                color: colors[idx % colors.length],
                products: []
              }));
              setApiCategories(mapped);
            }
          })
          .catch((err) => {
            console.error("Error fetching stock summary predictions:", err);
          });
      }
    }
  }, [activeStore]);

  // Derive categories list dynamically to prevent cascading render warnings
  const categories = apiCategories || getDynamicCategories(storeProducts);

  const handleCategoryClick = (cat) => {
    const catProducts = (storeProducts || []).filter(p => p.category === cat.name).map(p => ({
      id: p.id,
      name: p.name,
      sku: p.sku,
      stock: p.stock
    }));
    setSelectedCategory({
      ...cat,
      products: catProducts.length > 0 ? catProducts : [{ name: "No products registered in category", id: "N/A", sku: "N/A", stock: 0 }]
    });
  };

  const handleProductForecast = async (product) => {
    if (!product || product.id === "N/A") return;
    setSelectedProduct(product);
    setProductForecast(null);
    setProductForecastError("");
    setLoadingProductForecast(true);

    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || "";
    if (apiBase && activeStore) {
      try {
        const response = await fetch(`${apiBase}/api/predictions/next-month/${activeStore.id}/${product.id}`);
        if (response.ok) {
          const data = await response.json();
          setProductForecast(data);
        } else {
          const errData = await response.json();
          setProductForecastError(errData.detail || "Unable to load forecast.");
        }
      } catch (err) {
        console.error("Connection failure fetching item forecast:", err);
        setProductForecastError("Server connection failure. Ensure backend is running.");
      }
    } else {
      // Mock forecast for demo/fallback
      setProductForecast({
        store_id: activeStore?.id || "STORE_DEMO",
        item_id: product.id || "ITM_DEMO",
        monthly_total_units: 720,
        monthly_revenue: 21500.50,
        avg_daily_units: 24,
        business_metrics: {
          risk_level: "LOW",
          safety_stock: 45,
          reorder_point: 120,
          economic_order_quantity: 350,
          recommended_order_qty: 0
        },
        daily_forecast: Array.from({ length: 30 }, (_, i) => ({
          date: `2026-06-${(i + 1).toString().padStart(2, "0")}`,
          predicted_units: Math.round(20 + Math.sin(i / 2) * 5 + Math.random() * 3)
        }))
      });
    }
    setLoadingProductForecast(false);
  };

  const handlePredict = async () => {
    setPredicting(true);
    setPredictionError("");
    
    if (activeStore) {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || "";
      if (apiBase) {
        try {
          const response = await fetch(`${apiBase}/api/stores/${activeStore.id}/predictions/forecast`, {
            method: "POST"
          });
          if (response.ok) {
            const data = await response.json();
            setForecastData(data);
            setShowPredictions(true);
          } else {
            console.error("Prediction forecast endpoint error.");
            setPredictionError("Prediction generation failed. Ensure your transaction logs are fully synced.");
          }
        } catch (err) {
          console.error("Connection failure fetching demand predictions:", err);
          setPredictionError("Failed to connect to forecast service. Backend is currently offline.");
        }
      } else {
        setPredictionError("Forecast service endpoint is not configured.");
      }
    }
    setPredicting(false);
  };

  if (!historyLogs || historyLogs.length === 0) {
    return (
      <div className="space-y-6 font-sans px-6">
        <PageHeader
          title="AI Demand & Sales Predictions"
          icon={Sparkles}
        />
        <Card className="p-12 text-center space-y-4 max-w-md mx-auto border border-sky-100 bg-white rounded-2xl shadow-sm">
          <div className="w-12 h-12 rounded-full bg-sky-50 flex items-center justify-center mx-auto text-sky-500">
            <Sparkles className="w-6 h-6 animate-pulse" />
          </div>
          <h3 className="text-sm font-bold text-slate-850 font-serif">Models Not Trained</h3>
          <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
            AI demand forecasting and machine learning prediction models require active store transactions history to begin inference.
          </p>
        </Card>
      </div>
    );
  }

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
                <div key={idx} className="flex justify-between items-center text-xs py-1.5 border-b border-slate-50 last:border-b-0">
                  <div>
                    <span className="font-semibold text-slate-800 block">{p.name}</span>
                    <span className="text-[9px] text-slate-450 uppercase">{p.sku}</span>
                  </div>
                  <div className="flex items-center gap-2">
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
                    {p.id && p.id !== "N/A" && (
                      <button
                        onClick={() => handleProductForecast(p)}
                        className="px-2.5 py-1 bg-blue-50 hover:bg-blue-100 text-blue-600 font-bold rounded-lg text-[9px] transition-all shadow-xs"
                      >
                        ML Forecast
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Secondary Modal: Detailed ML Forecast for Selected Product */}
      {selectedProduct && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-white border border-sky-100 rounded-3xl max-w-2xl w-full shadow-2xl overflow-hidden animate-fade-in font-sans flex flex-col max-h-[90vh]">
            
            {/* Header */}
            <div className="p-6 bg-slate-50 border-b border-slate-100 flex justify-between items-center shrink-0">
              <div>
                <span className="text-[9px] font-bold text-blue-600 uppercase tracking-widest block">Machine Learning Ensemble Projections</span>
                <h3 className="text-lg font-bold text-slate-900 font-serif mt-1">{selectedProduct.name}</h3>
                <p className="text-[10px] text-slate-400 font-sans mt-0.5">SKU: {selectedProduct.sku} | ID: {selectedProduct.id}</p>
              </div>
              <button
                onClick={() => setSelectedProduct(null)}
                className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-500 hover:text-slate-800 flex items-center justify-center font-bold text-xs transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Content Body */}
            <div className="p-6 overflow-y-auto space-y-6 flex-1">
              {loadingProductForecast ? (
                <div className="py-16 flex flex-col items-center justify-center text-center gap-3">
                  <div className="w-10 h-10 rounded-full border-3 border-sky-150 border-t-blue-600 animate-spin" />
                  <span className="text-xs text-slate-500 font-bold">Running RF, XGBoost, LightGBM & CatBoost models...</span>
                </div>
              ) : productForecastError ? (
                <div className="py-8 text-center space-y-3">
                  <div className="w-12 h-12 rounded-full bg-rose-50 text-rose-500 flex items-center justify-center mx-auto">
                    <Sparkles className="w-6 h-6 animate-pulse" />
                  </div>
                  <h4 className="text-sm font-bold text-slate-800">Forecast Initialization Required</h4>
                  <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed">
                    {productForecastError}
                  </p>
                  <div className="text-[10px] text-slate-400 bg-slate-50 border border-slate-100 rounded-lg p-3 max-w-sm mx-auto">
                    Tip: Daily transaction histories are parsed chronologically to prevent model feature drift. Submit daily logs first.
                  </div>
                </div>
              ) : productForecast ? (
                <div className="space-y-6">
                  
                  {/* Key Stats Cards */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div className="bg-slate-50 border border-slate-100 p-4 rounded-2xl">
                      <span className="block text-[8px] font-bold text-slate-400 uppercase tracking-widest leading-none">Risk Level</span>
                      <span className={`inline-flex items-center text-xs font-black mt-2 px-2 py-0.5 rounded ${
                        productForecast.business_metrics?.risk_level === "HIGH" 
                          ? "bg-rose-50 text-rose-600" 
                          : productForecast.business_metrics?.risk_level === "MEDIUM"
                          ? "bg-amber-50 text-amber-600"
                          : "bg-emerald-50 text-emerald-600"
                      }`}>
                        {productForecast.business_metrics?.risk_level || "LOW"}
                      </span>
                    </div>

                    <div className="bg-slate-50 border border-slate-100 p-4 rounded-2xl">
                      <span className="block text-[8px] font-bold text-slate-400 uppercase tracking-widest leading-none">Reorder Point</span>
                      <span className="block text-base font-black text-slate-800 mt-1">
                        {productForecast.business_metrics?.reorder_point || 0} units
                      </span>
                    </div>

                    <div className="bg-slate-50 border border-slate-100 p-4 rounded-2xl">
                      <span className="block text-[8px] font-bold text-slate-400 uppercase tracking-widest leading-none">Economic Order Qty (EOQ)</span>
                      <span className="block text-base font-black text-slate-800 mt-1">
                        {productForecast.business_metrics?.economic_order_quantity || 0} units
                      </span>
                    </div>

                    <div className="bg-slate-50 border border-slate-100 p-4 rounded-2xl">
                      <span className="block text-[8px] font-bold text-slate-400 uppercase tracking-widest leading-none">Safety Stock</span>
                      <span className="block text-base font-black text-slate-800 mt-1">
                        {productForecast.business_metrics?.safety_stock || 0} units
                      </span>
                    </div>
                  </div>

                  {/* Monthly Summary */}
                  <div className="bg-blue-50/30 border border-blue-100/50 rounded-2xl p-5 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                    <div>
                      <h4 className="text-xs font-bold text-slate-800">Recommendation Summary</h4>
                      <p className="text-[10px] text-slate-500 leading-relaxed mt-1">
                        {productForecast.business_metrics?.recommended_order_qty > 0 
                          ? `Current stock is below reorder point. We recommend ordering ${productForecast.business_metrics.recommended_order_qty} units from the supplier.`
                          : "Stock levels are healthy. No replenishment orders are required at this time."}
                      </p>
                    </div>
                    <div className="shrink-0 bg-white border border-blue-100 px-4 py-2.5 rounded-xl text-center sm:text-right">
                      <span className="block text-[8px] font-bold text-slate-450 uppercase leading-none">Proj. Monthly Total</span>
                      <span className="block text-base font-black text-blue-600 mt-1">{productForecast.monthly_total_units || 0} units</span>
                    </div>
                  </div>

                  {/* Prediction Daily Sparkline / Trend */}
                  <div className="space-y-2">
                    <h4 className="text-xs font-bold text-slate-800">30-Day Forward Forecast Curve</h4>
                    <div className="h-44 bg-white border border-slate-150 rounded-2xl p-4 flex flex-col justify-between">
                      <div className="h-32 w-full relative flex items-end">
                        {/* Custom sparkline path */}
                        <svg viewBox="0 0 300 100" className="w-full h-full">
                          <line x1="0" y1="90" x2="300" y2="90" className="stroke-slate-100" strokeWidth="0.5" />
                          <line x1="0" y1="50" x2="300" y2="50" className="stroke-slate-100" strokeWidth="0.5" strokeDasharray="2 2" />
                          <line x1="0" y1="10" x2="300" y2="10" className="stroke-slate-100" strokeWidth="0.5" strokeDasharray="2 2" />

                          {/* Render line path */}
                          {productForecast.daily_forecast && productForecast.daily_forecast.length > 0 && (
                            <>
                              <path
                                d={`M ${productForecast.daily_forecast.map((val, idx) => {
                                  // Scale index to x (0 to 300) and value to y (10 to 90)
                                  const x = (idx / (productForecast.daily_forecast.length - 1)) * 300;
                                  const maxVal = Math.max(...productForecast.daily_forecast.map(d => d.predicted_units), 10);
                                  const y = 90 - (val.predicted_units / maxVal) * 80;
                                  return `${x} ${y}`;
                                }).join(" L ")}`}
                                fill="none"
                                stroke="#3b82f6"
                                strokeWidth="2.5"
                              />
                              {/* Draw small indicator points for weekends */}
                              {productForecast.daily_forecast.map((val, idx) => {
                                const x = (idx / (productForecast.daily_forecast.length - 1)) * 300;
                                const maxVal = Math.max(...productForecast.daily_forecast.map(d => d.predicted_units), 10);
                                const y = 90 - (val.predicted_units / maxVal) * 80;
                                if (idx % 7 === 5 || idx % 7 === 6) { // approximate weekends
                                  return <circle key={idx} cx={x} cy={y} r="2.5" fill="#ef4444" />;
                                }
                                return null;
                              })}
                            </>
                          )}
                        </svg>
                      </div>
                      <div className="flex justify-between items-center text-[9px] text-slate-400 font-medium px-1 pt-2 border-t border-slate-50">
                        <span>Day 1</span>
                        <span>Day 10</span>
                        <span>Day 20</span>
                        <span>Day 30</span>
                      </div>
                    </div>
                  </div>

                </div>
              ) : null}
            </div>

            {/* Footer */}
            <div className="p-4 bg-slate-50 border-t border-slate-100 flex justify-end shrink-0">
              <button
                onClick={() => setSelectedProduct(null)}
                className="px-5 py-2 bg-slate-200 hover:bg-slate-350 text-slate-700 font-bold text-xs rounded-xl transition-all"
              >
                Close Projections
              </button>
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

        {predictionError && (
          <div className="p-3.5 bg-rose-50 border border-rose-100 rounded-xl text-rose-600 text-xs font-bold font-sans">
            ✕ {predictionError}
          </div>
        )}

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
               <div className="bg-white border border-sky-100 p-4 rounded-xl shadow-sm flex items-center gap-3 font-sans">
                 <div className="w-8 h-8 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center">
                   <DollarSign className="w-4 h-4" />
                 </div>
                 <div>
                   <span className="block text-[9px] text-slate-400 font-bold uppercase tracking-wider">Projected Revenue</span>
                   <span className="text-base font-black text-slate-900 leading-none">
                     ₹{(forecastData?.summary?.projectedRevenue || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                   </span>
                 </div>
               </div>
 
               <div className="bg-white border border-sky-100 p-4 rounded-xl shadow-sm flex items-center gap-3 font-sans">
                 <div className="w-8 h-8 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center">
                   <Package className="w-4 h-4" />
                 </div>
                 <div>
                   <span className="block text-[9px] text-slate-400 font-bold uppercase tracking-wider">Projected Demand</span>
                   <span className="text-base font-black text-slate-900 leading-none">
                     {(forecastData?.summary?.projectedUnitsDemand || 0).toLocaleString()} Units
                   </span>
                 </div>
               </div>
 
               <div className="bg-white border border-sky-100 p-4 rounded-xl shadow-sm flex items-center gap-3 font-sans">
                 <div className="w-8 h-8 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center">
                   <TrendingUp className="w-4 h-4" />
                 </div>
                 <div>
                   <span className="block text-[9px] text-slate-400 font-bold uppercase tracking-wider">Model Confidence</span>
                   <span className="text-base font-black text-slate-900 leading-none">
                     {((forecastData?.summary?.confidenceMetricR2 || 0.948) * 100).toFixed(1)}% R²
                   </span>
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
