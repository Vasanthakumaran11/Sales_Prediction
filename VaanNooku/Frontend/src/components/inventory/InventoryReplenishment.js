"use client";

import React, { useState, useCallback } from "react";
import { Package, PlusCircle, AlertTriangle, FileSpreadsheet, RotateCw, Check, CheckCircle2 } from "lucide-react";
import { useStoreContext } from "@/context/StoreContext";
import { useAsync } from "@/hooks/useAsync";
import { getInventory, placeBulkOrder } from "@/lib/api/inventory";
import { PageHeader, Card } from "@/components/ui/Card";
import { StatTile } from "@/components/ui/StatTile";
import { Skeleton } from "@/components/ui/Skeleton";

export default function InventoryReplenishment() {
  const { activeStore } = useStoreContext();

  const [activeTab, setActiveTab] = useState("all"); // 'all' | 'shortages'
  const [cart, setCart] = useState([]);
  const [isSubmittingOrder, setIsSubmittingOrder] = useState(false);
  const [orderSuccess, setOrderSuccess] = useState(false);

  const loader = useCallback(async () => {
    const rawItems = await getInventory(activeStore?.id);
    return rawItems.map((item) => {
      const isShortage = item.stock <= item.rop;
      return {
        ...item,
        stockLevel: item.stock,
        safetyStock: item.minStock,
        reorderPoint: item.rop,
        status: isShortage ? "shortage" : "healthy",
        costPrice: Math.round(item.price * (1 - item.margin)),
      };
    });
  }, [activeStore]);

  const { data: gridItems, isLoading, reload } = useAsync(loader, [activeStore]);

  if (isLoading || !gridItems) {
    return (
      <div className="space-y-6">
        <div className="h-10 w-1/3 bg-slate-200 rounded animate-pulse" />
        <Skeleton className="h-62.5" />
      </div>
    );
  }

  // Filter grid items
  const filteredGrid = gridItems.filter((item) => {
    if (activeTab === "shortages") return item.status === "shortage";
    return true;
  });

  const totalShortages = gridItems.filter((i) => i.status === "shortage").length;

  // Add all shortages to order sheet
  const handleAddShortagesToCart = () => {
    const shortages = gridItems.filter((i) => i.status === "shortage");
    const cartEntries = shortages.map((item) => ({
      name: item.name,
      qty: item.eoq,
      cost: Math.round(item.eoq * item.costPrice * 0.85), // 15% bulk contract discount
    }));
    setCart(cartEntries);
    setOrderSuccess(false);
  };

  const handleBulkReplenishSubmit = async () => {
    if (cart.length === 0) return;
    setIsSubmittingOrder(true);
    try {
      await placeBulkOrder(cart);
      setOrderSuccess(true);
      setCart([]);
      reload();
    } catch (err) {
      console.error(err);
    } finally {
      setIsSubmittingOrder(false);
    }
  };


  const cartTotal = cart.reduce((sum, item) => sum + item.cost, 0);

  return (
    <div className="space-y-6 font-sans px-6">
      <PageHeader
        title="Scientific Inventory Replenishment"
        subtitle="Calculate item-specific Safety Stock, ROP limits, and trigger consolidated bulk orders."
        icon={Package}
      />

      {/* ROP Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatTile
          label="Total Tracked SKUs"
          value={`${gridItems.length} items`}
          icon={Package}
          hint="SKU catalog catalog count"
        />
        <StatTile
          label="Deficit Shortages"
          value={`${totalShortages} SKUs`}
          icon={AlertTriangle}
          delta={totalShortages > 0 ? "Requires Reorder" : "Healthy"}
          deltaDirection={totalShortages > 0 ? "down" : "up"}
          hint="Items at or below safety ROP"
        />
        <StatTile
          label="Procurement Discount"
          value="15% Contract"
          icon={FileSpreadsheet}
          hint="Multi-store wholesale volume rate"
        />
      </div>

      {/* Grid Table & Shopping Cart Split */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {/* Left Side: Table of SKUs */}
        <Card className="lg:col-span-2 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-100 pb-3">
            <div className="flex gap-2">
              <button
                onClick={() => setActiveTab("all")}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  activeTab === "all"
                    ? "bg-sky-50 border border-sky-100 text-sky-700"
                    : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
                }`}
              >
                All SKUs ({gridItems.length})
              </button>
              <button
                onClick={() => setActiveTab("shortages")}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  activeTab === "shortages"
                    ? "bg-rose-50 border border-rose-100 text-rose-700"
                    : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
                }`}
              >
                Shortages Only ({totalShortages})
              </button>
            </div>

            {totalShortages > 0 && (
              <button
                onClick={handleAddShortagesToCart}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs rounded-lg transition-all shadow"
              >
                <PlusCircle className="w-3.5 h-3.5" /> Add Shortages to Order
              </button>
            )}
          </div>

          <div className="overflow-x-auto border border-sky-100 rounded-xl bg-white">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-sky-100 text-slate-500 font-bold text-[9px] uppercase tracking-wider">
                  <th className="p-3">Product SKU</th>
                  <th className="p-3 text-right">In Stock</th>
                  <th className="p-3 text-right">Safety Stock</th>
                  <th className="p-3 text-right">ROP Limit</th>
                  <th className="p-3 text-right">EOQ Value</th>
                  <th className="p-3 text-center">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredGrid.map((item) => (
                  <tr
                    key={item.name}
                    className="hover:bg-sky-50/20 text-slate-700 transition-colors"
                  >
                    <td className="p-3 font-semibold text-slate-900 font-serif">{item.name}</td>
                    <td className="p-3 text-right">{item.stockLevel}</td>
                    <td className="p-3 text-right text-slate-500">{item.safetyStock}</td>
                    <td className="p-3 text-right text-slate-500 font-semibold">{item.reorderPoint}</td>
                    <td className="p-3 text-right font-medium text-slate-800">{item.eoq}</td>
                    <td className="p-3 text-center">
                      {item.status === "shortage" ? (
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-rose-50 border border-rose-100 text-rose-700 text-[9px] font-bold uppercase">
                          Shortage
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-sky-50 border border-sky-100 text-sky-700 text-[9px] font-bold uppercase">
                          Healthy
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Right Side: Consolidated Order Sheet Cart */}
        <div className="bg-white border border-sky-100 rounded-2xl p-5 flex flex-col h-120 justify-between shadow-sm">
          <div className="space-y-1.5 border-b border-slate-100 pb-3">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider font-serif">
              Consolidated Bulk Purchase Order
            </h3>
            <p className="text-[10px] text-slate-500">
              Aggregated procurement order. Applies a 15% wholesale discount rate.
            </p>
          </div>

          <div className="flex-1 overflow-y-auto py-3 space-y-2 pr-1 font-sans text-xs">
            {cart.length > 0 ? (
              cart.map((item, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-lg bg-sky-50/30 border border-sky-100 flex justify-between items-center text-slate-700"
                >
                  <div>
                    <span className="font-bold text-slate-900 block">{item.name}</span>
                    <span className="text-[10px] text-slate-500">{item.qty} units (EOQ)</span>
                  </div>
                  <span className="font-bold text-slate-800">₹{item.cost.toLocaleString()}</span>
                </div>
              ))
            ) : orderSuccess ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 gap-3">
                <div className="w-12 h-12 rounded-full bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600 animate-bounce">
                  <CheckCircle2 className="w-6 h-6" />
                </div>
                <div className="space-y-1">
                  <span className="block text-sm font-bold text-slate-900 font-serif">Order Ingestion Successful</span>
                  <p className="text-[10.5px] text-slate-500 leading-normal">
                    Invoice dispatch triggered. Inventory database synced.
                  </p>
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-400">
                <Package className="w-8 h-8 text-slate-300 mb-2" />
                <span className="block text-[11px] font-bold uppercase tracking-wider">Cart is Empty</span>
                <p className="text-[10px] text-slate-400 mt-1 leading-normal">
                  Add shortage SKUs to formulate the bulk procurement invoice.
                </p>
              </div>
            )}
          </div>

          {/* Cart checkout footer panel */}
          <div className="border-t border-slate-100 pt-3 space-y-3 font-sans text-xs">
            <div className="p-3 bg-sky-50/40 border border-sky-100 rounded-lg flex justify-between items-center">
              <span className="text-slate-500 font-bold uppercase text-[9px] tracking-wider">Total Est Cost</span>
              <span className="font-black text-slate-950 text-base">₹{cartTotal.toLocaleString()}</span>
            </div>

            <button
              id="btn-dispatch-order"
              onClick={handleBulkReplenishSubmit}
              disabled={cart.length === 0 || isSubmittingOrder}
              className="w-full py-3 bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs rounded-lg transition-all flex items-center justify-center gap-1.5 shadow disabled:opacity-40"
            >
              {isSubmittingOrder ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" /> Dispatching Bulk PO...
                </>
              ) : (
                <>
                  <Check className="w-4 h-4 text-sky-200" /> Dispatch Purchase Order
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
