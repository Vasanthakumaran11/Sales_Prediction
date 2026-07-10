import React from "react";
import { ChevronLeft, ArrowRight, Building2 } from "lucide-react";
import { STORE_PROFILES } from "@/lib/mock/stores";

export function ExecutiveGatewayView({ setGatewayState, enterExecutiveMode }) {
  return (
    <div className="w-full max-w-4xl bg-white border border-sky-100 rounded-2xl shadow-xl p-8 relative z-10 space-y-6">
      <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
        <button
          onClick={() => setGatewayState("landing")}
          className="p-1 rounded bg-slate-50 border border-slate-200 hover:bg-slate-100 text-slate-500"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
        <div>
          <h3 className="text-base font-bold text-slate-900 font-serif">Multi-Store Executive Control Tower</h3>
          <p className="text-[10px] text-slate-500 font-sans">
            Aggregated operations and performance benchmarking across all registered nodes.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 font-sans">
        <div className="bg-sky-50/50 border border-sky-100 p-4 rounded-xl">
          <span className="text-[10px] text-slate-500 font-bold uppercase block">Total Chain Revenue</span>
          <span className="text-lg font-bold text-slate-900">₹455,900</span>
          <span className="text-[10px] text-sky-600 block mt-1">+14.2% projected MoM</span>
        </div>
        <div className="bg-sky-50/50 border border-sky-100 p-4 rounded-xl">
          <span className="text-[10px] text-slate-500 font-bold uppercase block">Active Asset Value</span>
          <span className="text-lg font-bold text-teal-600">₹305,400</span>
          <span className="text-[10px] text-slate-500 block mt-1">Spread across 3 active store formats</span>
        </div>
        <div className="bg-sky-50/50 border border-sky-100 p-4 rounded-xl">
          <span className="text-[10px] text-slate-500 font-bold uppercase block">Global Stockout Alerts</span>
          <span className="text-lg font-bold text-rose-600">17 Items</span>
          <span className="text-[10px] text-rose-500 block mt-1">Requires immediate bulk purchasing</span>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto border border-sky-100 rounded-xl bg-white">
        <table className="w-full border-collapse text-left text-xs font-sans">
          <thead>
            <tr className="bg-slate-50 border-b border-sky-100 text-slate-500 font-bold text-[9px] uppercase tracking-wider">
              <th className="p-3">Store Node</th>
              <th className="p-3">Format Type / Location</th>
              <th className="p-3 text-right">R² Accuracy Score</th>
              <th className="p-3 text-right">Waste / Expiry Margin</th>
              <th className="p-3 text-right">Active Inventory Value</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {STORE_PROFILES.map((store) => (
              <tr key={store.id} className="hover:bg-sky-50/30 text-slate-700">
                <td className="p-3 font-semibold text-slate-900">{store.name}</td>
                <td className="p-3">
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 text-[9px] text-slate-500 border border-slate-200">
                    {store.type} - {store.location}
                  </span>
                </td>
                <td className="p-3 text-right font-medium text-sky-600">
                  {parseFloat(store.metrics.forecastR2) * 100}%
                </td>
                <td className="p-3 text-right font-medium text-amber-600">
                  {parseFloat(store.metrics.wasteMargin) * 100}%
                </td>
                <td className="p-3 text-right text-slate-800">₹{store.metrics.inventoryValue.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex justify-end">
        <button
          id="btn-launch-chain"
          onClick={enterExecutiveMode}
          className="flex items-center gap-2 bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs px-6 py-2.5 rounded-lg transition-all shadow"
        >
          Access Executive Control Tower <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
