import React, { useState } from "react";
import { Upload, ArrowRight, CheckCircle2, ChevronLeft, Sparkles, Building2 } from "lucide-react";

export function ShelfRecommendations({
  selectedProducts,
  setSelectedProducts,
  formData,
  setGatewayState,
  handleUploadPurchasePlan
}) {
  const [hasPurchasePlan, setHasPurchasePlan] = useState(null); // null | 'yes' | 'no'

  return (
    <div className="w-full max-w-4xl bg-white border border-sky-100 rounded-2xl shadow-xl overflow-hidden relative z-10 flex flex-col min-h-[520px]">
      {/* macOS-style Top Bar */}
      <div className="relative p-4 bg-slate-100 border-b border-slate-200 flex items-center justify-center h-12 shrink-0">
        <div className="absolute left-4 flex items-center gap-1.5 group">
          <span
            onClick={() => {
              if (hasPurchasePlan !== null) {
                setHasPurchasePlan(null);
              } else {
                setGatewayState("register");
              }
            }}
            className="w-3 h-3 rounded-full bg-rose-500 hover:bg-rose-600 cursor-pointer flex items-center justify-center text-[7.5px] text-rose-950 font-bold shrink-0 transition-colors after:content-[''] hover:after:content-['×']"
            title="Go Back"
          />
          <span className="w-3 h-3 rounded-full bg-amber-500 hover:bg-amber-600 cursor-pointer flex items-center justify-center text-[7.5px] text-amber-950 font-bold shrink-0 transition-colors after:content-[''] hover:after:content-['−']" />
          <span className="w-3 h-3 rounded-full bg-emerald-500 hover:bg-emerald-600 cursor-pointer flex items-center justify-center text-[7.5px] text-emerald-950 font-bold shrink-0 transition-colors after:content-[''] hover:after:content-['+']" />
        </div>

        <div className="flex items-center gap-6 text-[11px] font-bold tracking-wide">
          <div className="flex items-center gap-2 text-slate-400">
            <span className="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-[10px] font-bold">
              1
            </span>
            <span>Onboarding</span>
          </div>
          <div className="w-16 h-0.5 bg-slate-200 rounded-full" />

          <div className="flex items-center gap-2">
            <span className="w-5 h-5 rounded-full bg-sky-600 flex items-center justify-center text-white font-bold text-[10px]">
              2
            </span>
            <span className="text-slate-800">Shelf Recommendations</span>
          </div>
          <div className="w-16 h-0.5 bg-slate-200 rounded-full" />

          <div className="flex items-center gap-2 text-slate-400">
            <span className="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-[10px] font-bold">
              3
            </span>
            <span>Purchase Order</span>
          </div>
        </div>
      </div>

      <div className="p-8 flex-1 bg-white font-sans flex flex-col justify-between">
        {/* Step A: Ask question */}
        {hasPurchasePlan === null && (
          <div className="flex-1 flex flex-col items-center justify-center text-center max-w-xl mx-auto space-y-7 py-6">
            <div className="w-14 h-14 rounded-2xl bg-sky-100 flex items-center justify-center text-sky-600">
              <Sparkles className="w-7 h-7" />
            </div>
            <div className="space-y-2">
              <h2 className="text-2xl font-bold text-slate-900 font-serif">Do you have a pre-existing purchase plan?</h2>
              <p className="text-xs text-slate-500 max-w-md leading-relaxed">
                Select if you have a custom spreadsheet catalog ready to import, or if you want our AI model to recommend high-margin shelf products matching your initial investment.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full pt-2">
              <button
                type="button"
                onClick={() => {
                  handleUploadPurchasePlan();
                  setHasPurchasePlan("yes");
                }}
                className="flex flex-col items-center justify-center p-6 border-2 border-sky-100 hover:border-sky-300 rounded-2xl transition-all gap-3 text-center bg-sky-50/20 hover:bg-sky-50/50 group"
              >
                <div className="w-10 h-10 rounded-xl bg-sky-100 text-sky-600 flex items-center justify-center shrink-0">
                  <Upload className="w-5 h-5" />
                </div>
                <div>
                  <span className="font-bold text-slate-900 text-sm block">Yes, upload my plan</span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">Import custom SKU spreadsheet</span>
                </div>
              </button>

              <button
                type="button"
                onClick={() => setHasPurchasePlan("no")}
                className="flex flex-col items-center justify-center p-6 border-2 border-slate-200 hover:border-sky-200 rounded-2xl transition-all gap-3 text-center bg-white hover:bg-slate-50 group"
              >
                <div className="w-10 h-10 rounded-xl bg-slate-100 text-slate-600 flex items-center justify-center shrink-0">
                  <Building2 className="w-5 h-5" />
                </div>
                <div>
                  <span className="font-bold text-slate-900 text-sm block">No, suggest products</span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">Generate model suggested shelf lists</span>
                </div>
              </button>
            </div>
          </div>
        )}

        {/* Step B: Render product selection or uploaded custom layout */}
        {hasPurchasePlan !== null && (
          <div className="space-y-6 w-full flex-1">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-100 pb-3">
              <div className="space-y-1">
                <h2 className="text-2xl font-bold text-slate-900 font-serif">
                  {hasPurchasePlan === "yes" ? "Custom Purchase Plan Loaded" : "Recommended Product Strategy"}
                </h2>
                <p className="text-xs text-slate-500 font-sans">
                  {hasPurchasePlan === "yes"
                    ? "Verify the SKU counts and units imported from your custom purchase plan."
                    : `Suggested starting items to buy matching your investment of ₹${(parseInt(formData.investment) || 0).toLocaleString()}`}
                </p>
              </div>
              {hasPurchasePlan === "yes" && (
                <button
                  type="button"
                  onClick={handleUploadPurchasePlan}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold rounded-lg shadow-sm font-sans"
                >
                  <Upload className="w-3.5 h-3.5 text-slate-500" /> Re-upload Plan
                </button>
              )}
            </div>

            <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1 border border-slate-150 rounded-xl p-2.5 bg-slate-50/50">
              {selectedProducts.map((p) => {
                const productCost = p.buyingPrice * p.qty;
                return (
                  <div
                    key={p.id}
                    className={`flex items-center justify-between p-3.5 rounded-xl border transition-all ${
                      p.checked
                        ? "bg-white border-sky-300 shadow-sm"
                        : "bg-slate-100/50 border-slate-200 opacity-60"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        checked={p.checked}
                        onChange={() =>
                          setSelectedProducts((prev) =>
                            prev.map((item) => (item.id === p.id ? { ...item, checked: !item.checked } : item))
                          )
                        }
                        className="w-4.5 h-4.5 rounded text-sky-600 accent-sky-600 cursor-pointer"
                      />
                      <div>
                        <span className="font-bold text-slate-800 text-xs block">{p.name}</span>
                        <span className="text-[9px] text-slate-500">{p.category}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 text-xs font-semibold">
                      <div className="text-right">
                        <span className="text-[10px] text-slate-400 block font-bold">COST / UNIT</span>
                        <span className="text-slate-800">₹{p.buyingPrice}</span>
                      </div>

                      <div className="w-16">
                        <span className="text-[10px] text-slate-400 block font-bold text-center">QTY</span>
                        <input
                          type="number"
                          min="1"
                          value={p.qty}
                          onChange={(e) =>
                            setSelectedProducts((prev) =>
                              prev.map((item) => (item.id === p.id ? { ...item, qty: parseInt(e.target.value) || 0 } : item))
                            )
                          }
                          disabled={!p.checked}
                          className="w-full bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 text-center text-xs focus:outline-none focus:border-sky-500 font-bold"
                        />
                      </div>

                      <div className="text-right w-20">
                        <span className="text-[10px] text-slate-400 block font-bold">TOTAL</span>
                        <span className="text-slate-900 font-bold">₹{productCost.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Footer controls */}
        <div className="flex items-center justify-between border-t border-slate-200 pt-5 mt-4 text-xs font-sans">
          <button
            type="button"
            onClick={() => {
              if (hasPurchasePlan !== null) {
                setHasPurchasePlan(null);
              } else {
                setGatewayState("register");
              }
            }}
            className="px-4 py-2 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50 hover:text-slate-900 font-bold transition-all"
          >
            Back
          </button>
          {hasPurchasePlan !== null && (
            <button
              id="btn-initialize-store-engine"
              type="button"
              onClick={() => setGatewayState("purchase-order")}
              className="bg-linear-to-r from-sky-600 to-blue-600 hover:from-sky-500 hover:to-blue-500 text-white font-bold text-xs px-6 py-2.5 rounded-lg shadow transition-all flex items-center gap-1.5"
            >
              Initialize Store Engine <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
