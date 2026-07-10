import React from "react";
import { CheckCircle2, ChevronLeft, ArrowRight } from "lucide-react";

export function PurchaseOrderConfirm({
  selectedProducts,
  supplierForm,
  setSupplierForm,
  setGatewayState,
  handleActivateStore
}) {
  return (
    <div className="w-full max-w-5xl bg-white border border-sky-100 rounded-2xl shadow-xl overflow-hidden relative z-10 flex flex-col min-h-[520px]">
      {/* macOS-style Top Bar */}
      <div className="relative p-4 bg-slate-100 border-b border-slate-200 flex items-center justify-center h-12 shrink-0">
        <div className="absolute left-4 flex items-center gap-1.5 group">
          <span
            onClick={() => setGatewayState("recommendations")}
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

          <div className="flex items-center gap-2 text-slate-400">
            <span className="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-[10px] font-bold">
              2
            </span>
            <span>Shelf Recommendations</span>
          </div>
          <div className="w-16 h-0.5 bg-slate-200 rounded-full" />

          <div className="flex items-center gap-2">
            <span className="w-5 h-5 rounded-full bg-sky-600 flex items-center justify-center text-white font-bold text-[10px]">
              3
            </span>
            <span className="text-slate-800">Purchase Order</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 p-8 flex-1 items-start bg-white font-sans">
        {/* Left 3 Columns: Purchase Order summary and Custom Supplier field */}
        <div className="lg:col-span-3 space-y-6">
          <div className="space-y-1">
            <h2 className="text-2xl font-bold text-slate-900 tracking-wide font-serif">Purchase Order Reconfirmation</h2>
            <p className="text-xs text-slate-500">
              Review the items being purchased and input custom wholesale supplier details if available.
            </p>
          </div>

          {/* Purchase summary */}
          <div className="border border-slate-200 rounded-xl overflow-hidden bg-slate-50/50">
            <table className="w-full border-collapse text-left text-xs font-sans">
              <thead>
                <tr className="bg-slate-100 border-b border-slate-200 text-slate-500 font-bold text-[10px] uppercase">
                  <th className="p-3">Product SKU</th>
                  <th className="p-3 text-right">Units</th>
                  <th className="p-3 text-right">Unit Price</th>
                  <th className="p-3 text-right">Total Cost</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-150 text-slate-700">
                {selectedProducts
                  .filter((p) => p.checked)
                  .map((p) => (
                    <tr key={p.id} className="hover:bg-white transition-colors">
                      <td className="p-3 font-semibold text-slate-900">{p.name}</td>
                      <td className="p-3 text-right font-medium">{p.qty}</td>
                      <td className="p-3 text-right">₹{p.buyingPrice}</td>
                      <td className="p-3 text-right font-bold text-slate-800">₹{(p.buyingPrice * p.qty).toLocaleString()}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* Supplier details form */}
          <div className="space-y-4 pt-2">
            <h3 className="text-sm font-bold text-slate-900 font-serif">Supplier Details (If Available)</h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Supplier Name</label>
                <input
                  type="text"
                  value={supplierForm.name}
                  onChange={(e) => setSupplierForm((prev) => ({ ...prev, name: e.target.value }))}
                  placeholder="e.g. Balaji Agro Distributors"
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-xs text-slate-800 focus:outline-none focus:border-sky-500 font-sans"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Contact Phone</label>
                <input
                  type="text"
                  value={supplierForm.phone}
                  onChange={(e) => setSupplierForm((prev) => ({ ...prev, phone: e.target.value }))}
                  placeholder="e.g. +91 98765 43210"
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-xs text-slate-800 focus:outline-none focus:border-sky-500 font-sans"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right 2 Columns: Supplier suggestions based on previous details */}
        <div className="lg:col-span-2 space-y-6">
          <div className="w-full bg-sky-50/50 border border-sky-100 rounded-2xl p-6 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-900 tracking-wide font-serif">Suggested Supplier Partners</h3>
            <p className="text-[10px] text-slate-500 leading-normal">
              If custom supplier details are not provided, we suggest the following regional wholesale contacts based on target categories. Click to select a supplier:
            </p>

            <div className="space-y-3 pt-1">
              <button
                type="button"
                onClick={() => setSupplierForm({ name: "Balaji Agro Distributors", phone: "+91 98765 43210", email: "contact@balajiagro.com" })}
                className={`w-full text-left p-3 rounded-xl border transition-all ${
                  supplierForm.name === "Balaji Agro Distributors"
                    ? "bg-white border-sky-600 shadow-md ring-2 ring-sky-100"
                    : "bg-white border-sky-100 hover:border-sky-300"
                }`}
              >
                <span className="text-[9px] text-slate-400 font-bold block uppercase">STAPLES & GRAINS supplier</span>
                <span className="font-bold text-slate-850 block">Balaji Agro Distributors</span>
                <span className="text-[9px] text-slate-500 font-medium">98.5% reliability • Chennai HQ • Click to Select</span>
              </button>

              <button
                type="button"
                onClick={() => setSupplierForm({ name: "Surya Packaged Goods Ltd", phone: "+91 87654 32109", email: "sales@suryapackaged.com" })}
                className={`w-full text-left p-3 rounded-xl border transition-all ${
                  supplierForm.name === "Surya Packaged Goods Ltd"
                    ? "bg-white border-sky-600 shadow-md ring-2 ring-sky-100"
                    : "bg-white border-sky-100 hover:border-sky-300"
                }`}
              >
                <span className="text-[9px] text-slate-400 font-bold block uppercase">BEVERAGES supplier</span>
                <span className="font-bold text-slate-850 block">Surya Packaged Goods Ltd</span>
                <span className="text-[9px] text-slate-500 font-medium">96.8% fulfillment rate • Bangalore • Click to Select</span>
              </button>

              <button
                type="button"
                onClick={() => setSupplierForm({ name: "Shiva Dairy & Farms", phone: "+91 76543 21098", email: "orders@shivadairy.com" })}
                className={`w-full text-left p-3 rounded-xl border transition-all ${
                  supplierForm.name === "Shiva Dairy & Farms"
                    ? "bg-white border-sky-600 shadow-md ring-2 ring-sky-100"
                    : "bg-white border-sky-100 hover:border-sky-300"
                }`}
              >
                <span className="text-[9px] text-slate-400 font-bold block uppercase">DAIRY & BAKERY supplier</span>
                <span className="font-bold text-slate-850 block">Shiva Dairy & Farms</span>
                <span className="text-[9px] text-slate-500 font-medium">Fresh cold storage • Coimbatore • Click to Select</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Controls */}
      <div className="p-5 bg-slate-50 border-t border-slate-200 flex items-center justify-between h-16 shrink-0 font-sans">
        <button
          type="button"
          onClick={() => setGatewayState("recommendations")}
          className="px-4 py-2 rounded-lg border border-slate-200 text-slate-650 hover:bg-slate-50 hover:text-slate-900 font-bold transition-all"
        >
          Back to Product Selection
        </button>
        <button
          id="btn-confirm-po"
          type="button"
          onClick={handleActivateStore}
          className="bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs px-6 py-2.5 rounded-lg shadow-sm flex items-center gap-1.5"
        >
          Confirm Purchase & Log In <CheckCircle2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
