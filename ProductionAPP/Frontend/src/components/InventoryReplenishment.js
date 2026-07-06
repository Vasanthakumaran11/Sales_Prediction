import React, { useState } from 'react';
import { 
  Package, 
  AlertTriangle, 
  ShoppingBag, 
  CheckCircle, 
  Layers, 
  Info, 
  Maximize2, 
  Boxes 
} from 'lucide-react';
import { INVENTORY_ITEMS } from './mockData';

export default function InventoryReplenishment({ storeInfo, isMultiStoreMode }) {
  const [items, setItems] = useState(INVENTORY_ITEMS);
  const [showBulkConsolidator, setShowBulkConsolidator] = useState(false);
  const [consolidatedOrderPlaced, setConsolidatedOrderPlaced] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Calculate critical items (stock <= ROP)
  const criticalItems = items.filter(item => item.stock <= item.rop);

  // Filter items based on search
  const filteredItems = items.filter(item => 
    item.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
    item.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getStockColorClass = (stock, minStock, cap) => {
    const ratio = stock / cap;
    if (stock <= minStock) {
      return {
        bar: 'bg-rose-500',
        text: 'text-rose-600 dark:text-rose-450',
        bg: 'bg-rose-50 dark:bg-rose-500/10 border-rose-100 dark:border-rose-500/20'
      };
    }
    if (ratio < 0.25) {
      return {
        bar: 'bg-amber-500',
        text: 'text-amber-600 dark:text-amber-400',
        bg: 'bg-amber-50 dark:bg-amber-500/10 border-amber-100 dark:border-amber-500/20'
      };
    }
    return {
      bar: 'bg-emerald-500',
      text: 'text-emerald-600 dark:text-emerald-450',
      bg: 'bg-emerald-50 dark:bg-emerald-500/10 border-emerald-100 dark:border-emerald-500/20'
    };
  };

  const getRiskBadgeColor = (risk) => {
    switch (risk) {
      case 'High': return 'bg-rose-50 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-100 dark:border-rose-500/20';
      case 'Medium': return 'bg-amber-50 dark:bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-100 dark:border-amber-500/20';
      default: return 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-100 dark:border-emerald-500/20';
    }
  };

  // Bulk Consolidation details
  const getBulkConsolidatedItems = () => {
    return [
      { name: "Curd", balaji: 13, shiva: 25, surya: 12, eoq: 50, price: 45, unit: "units" },
      { name: "Eggs", balaji: 27, shiva: 50, surya: 35, eoq: 100, price: 45, unit: "crates" },
      { name: "Oil", balaji: 15, shiva: 30, surya: 25, eoq: 100, price: 160, unit: "bottles" },
      { name: "Namkeen", balaji: 23, shiva: 45, surya: 30, eoq: 90, price: 80, unit: "packs" }
    ];
  };

  const bulkItems = getBulkConsolidatedItems();
  const totalStandardCost = bulkItems.reduce((acc, item) => {
    const totalQty = item.balaji + item.shiva + item.surya;
    return acc + (totalQty * item.price);
  }, 0);

  const bulkDiscount = 0.15;
  const totalDiscountedCost = Math.round(totalStandardCost * (1 - bulkDiscount));

  return (
    <div className="space-y-6 font-sans">
      {/* Top Header Row */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-zinc-900 dark:text-white tracking-tight flex items-center gap-2">
            <Package className="w-5 h-5 text-emerald-500" /> Scientific Replenishment & Inventory
          </h2>
          <p className="text-xs text-zinc-550 dark:text-zinc-400 mt-0.5">
            Optimize replenishment cycles using Safety Stock buffers, Reorder Points (ROP), and Economic Order Quantities (EOQ).
          </p>
        </div>
        
        {/* Actions bar */}
        <div className="flex items-center gap-3">
          {(isMultiStoreMode || storeInfo?.id === 'new_store') && (
            <button
              id="btn-toggle-bulk"
              onClick={() => {
                setShowBulkConsolidator(!showBulkConsolidator);
                setConsolidatedOrderPlaced(false);
              }}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                showBulkConsolidator 
                  ? 'bg-blue-600 border-blue-500 text-white shadow-md shadow-blue-900/20' 
                  : 'bg-zinc-50 dark:bg-zinc-900 border-zinc-250 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-850'
              }`}
            >
              <Boxes className="w-3.5 h-3.5" />
              <span>Consolidated Bulk Tool</span>
            </button>
          )}
          <div className="relative">
            <input
              id="input-inventory-search"
              type="text"
              placeholder="Search catalog..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-850 rounded-lg px-3 py-1.5 text-xs text-zinc-900 dark:text-white placeholder-zinc-450 focus:outline-none focus:border-emerald-500 w-44"
            />
          </div>
        </div>
      </div>

      {/* Bulk Consolidator Panel */}
      {showBulkConsolidator && (
        <div className="p-6 rounded-2xl bg-white dark:bg-zinc-900/60 border border-blue-200 dark:border-blue-500/30 shadow-xl relative overflow-hidden space-y-5">
          <div className="absolute top-0 right-0 w-48 h-48 bg-blue-500/5 rounded-full blur-3xl pointer-events-none" />
          
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/25 text-[10px] text-blue-600 dark:text-blue-400 font-bold uppercase tracking-wider mb-2">
                <Layers className="w-3 h-3" /> Multi-Store Purchasing Engine
              </div>
              <h3 className="text-lg font-bold text-zinc-900 dark:text-white">Consolidated ROP Order Planner</h3>
              <p className="text-xs text-zinc-550 dark:text-zinc-400 mt-0.5">
                Automatically aggregates individual store ROP demands to trigger unified vendor purchase orders with volume discounts.
              </p>
            </div>
            {!consolidatedOrderPlaced && (
              <div className="text-left sm:text-right">
                <span className="block text-[10px] text-zinc-400 dark:text-zinc-500 uppercase font-semibold">Consolidated Savings</span>
                <span className="text-lg font-bold text-emerald-600 dark:text-emerald-400">15% Volume Discount Active</span>
              </div>
            )}
          </div>

          {!consolidatedOrderPlaced ? (
            <div className="space-y-4">
              {/* Aggregation Table */}
              <div className="overflow-x-auto border border-zinc-200 dark:border-zinc-800/80 rounded-xl bg-zinc-50/50 dark:bg-zinc-950/80">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="bg-zinc-100 dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800 text-zinc-500 dark:text-zinc-400 font-semibold text-[9px] uppercase tracking-wider">
                      <th className="p-3">Item Profile</th>
                      <th className="p-3 text-right">Balaji Store</th>
                      <th className="p-3 text-right">Shiva Stores</th>
                      <th className="p-3 text-right">Surya Markets</th>
                      <th className="p-3 text-right font-bold text-zinc-800 dark:text-zinc-200">Unified Qty</th>
                      <th className="p-3 text-right">Unit Price</th>
                      <th className="p-3 text-right">Consolidated Cost</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-150 dark:divide-zinc-900">
                    {bulkItems.map(item => {
                      const totalQty = item.balaji + item.shiva + item.surya;
                      const standardItemCost = totalQty * item.price;
                      return (
                        <tr key={item.name} className="hover:bg-zinc-100/50 dark:hover:bg-zinc-900/20 text-zinc-700 dark:text-zinc-300">
                          <td className="p-3 font-semibold text-zinc-900 dark:text-white">{item.name}</td>
                          <td className="p-3 text-right text-zinc-500 dark:text-zinc-400">{item.balaji} {item.unit}</td>
                          <td className="p-3 text-right text-zinc-500 dark:text-zinc-400">{item.shiva} {item.unit}</td>
                          <td className="p-3 text-right text-zinc-500 dark:text-zinc-400">{item.surya} {item.unit}</td>
                          <td className="p-3 text-right font-bold text-emerald-600 dark:text-emerald-400">{totalQty} {item.unit}</td>
                          <td className="p-3 text-right">₹{item.price}</td>
                          <td className="p-3 text-right font-semibold">₹{standardItemCost.toLocaleString()}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Summary */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-4 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800">
                <div className="space-y-1">
                  <div className="text-xs text-zinc-500">
                    Standard Cost: <span className="line-through text-zinc-400">₹{totalStandardCost.toLocaleString()}</span>
                  </div>
                  <div className="text-sm text-zinc-800 dark:text-zinc-300">
                    Consolidated Contract Price: <span className="font-bold text-zinc-900 dark:text-white">₹{totalDiscountedCost.toLocaleString()}</span>{' '}
                    <span className="text-emerald-600 dark:text-emerald-400 text-xs font-semibold">(Saved ₹{(totalStandardCost - totalDiscountedCost).toLocaleString()})</span>
                  </div>
                </div>
                <button
                  id="btn-place-bulk-order"
                  onClick={() => setConsolidatedOrderPlaced(true)}
                  className="bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs px-5 py-2.5 rounded-lg flex items-center gap-1.5 shadow-lg"
                >
                  <ShoppingBag className="w-4 h-4" /> Place Consolidated Purchase Order
                </button>
              </div>
            </div>
          ) : (
            <div className="p-8 text-center bg-zinc-50 dark:bg-zinc-950/80 rounded-xl border border-emerald-500/20 py-10 space-y-4">
              <CheckCircle className="w-12 h-12 text-emerald-500 mx-auto" />
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-zinc-900 dark:text-white">Consolidated Purchase Order Dispatched</h4>
                <p className="text-xs text-zinc-500 dark:text-zinc-400 max-w-sm mx-auto">
                  A purchase order for the aggregated {bulkItems.reduce((acc, item) => acc + item.balaji + item.shiva + item.surya, 0)} items has been sent. Inventory ledger updates will reflect upon vendor receipt.
                </p>
              </div>
              <button
                onClick={() => setConsolidatedOrderPlaced(false)}
                className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-500 font-semibold"
              >
                Place Another Order
              </button>
            </div>
          )}
        </div>
      )}

      {/* Critical Stock Alerts Panel */}
      {criticalItems.length > 0 && !showBulkConsolidator && (
        <div className="bg-rose-50 dark:bg-rose-950/15 border border-rose-100 dark:border-rose-900/30 rounded-2xl p-5 space-y-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-600 dark:text-rose-500" />
            <h3 className="text-xs font-bold text-rose-700 dark:text-rose-400 uppercase tracking-wider">ROP Critical Deficit Alerts</h3>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {criticalItems.slice(0, 4).map(item => (
              <div key={item.name} className="p-3 rounded-lg bg-white dark:bg-zinc-950 border border-rose-150 dark:border-rose-900/30 flex justify-between items-center text-xs shadow-sm">
                <div>
                  <span className="font-bold text-zinc-900 dark:text-white block">{item.name}</span>
                  <span className="text-[10px] text-zinc-450 dark:text-zinc-500">Category: {item.category}</span>
                </div>
                <div className="text-right">
                  <span className="font-bold text-rose-600 dark:text-rose-400 block">{item.stock} Units left</span>
                  <span className="text-[9px] text-zinc-450 dark:text-zinc-500 font-semibold">ROP: {item.rop}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Core Data Matrix */}
      <div className="bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 space-y-4 shadow-sm">
        <div>
          <h3 className="text-sm font-bold text-zinc-900 dark:text-white uppercase tracking-wider">Inventory Metrics Grid (Layer 6)</h3>
          <p className="text-[10px] text-zinc-500">Calculated Safety Stock buffers, Reorder Point thresholds, and EOQ metrics.</p>
        </div>

        <div className="overflow-x-auto border border-zinc-200 dark:border-zinc-800 rounded-xl bg-white dark:bg-zinc-950">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="bg-zinc-50 dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800 text-zinc-500 dark:text-zinc-400 font-semibold text-[9px] uppercase tracking-wider">
                <th className="p-3">Item Profile</th>
                <th className="p-3">Available Stock Level</th>
                <th className="p-3 text-right">Unit Price</th>
                <th className="p-3 text-right">Margin %</th>
                <th className="p-3 text-right">Safety Stock</th>
                <th className="p-3 text-right">Reorder Point (ROP)</th>
                <th className="p-3 text-right">EOQ (Ideal Batch)</th>
                <th className="p-3 text-right">Stockout Risk</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-150 dark:divide-zinc-900">
              {filteredItems.map(item => {
                const stockColor = getStockColorClass(item.stock, item.minStock, item.cap);
                const percentFull = Math.min(100, Math.round((item.stock / item.cap) * 100));
                
                return (
                  <tr key={item.name} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-900/40 text-zinc-700 dark:text-zinc-300 transition-colors">
                    <td className="p-3">
                      <div>
                        <span className="font-bold text-zinc-900 dark:text-white block">{item.name}</span>
                        <span className="text-[9px] text-zinc-450 dark:text-zinc-500">{item.category}</span>
                      </div>
                    </td>
                    
                    <td className="p-3 w-48">
                      <div className="space-y-1">
                        <div className="flex justify-between items-center text-[10px]">
                          <span className={`font-semibold ${stockColor.text}`}>{item.stock} / {item.cap} units</span>
                          <span className="text-zinc-500">{percentFull}%</span>
                        </div>
                        <div className="w-full h-1.5 bg-zinc-100 dark:bg-zinc-900 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${stockColor.bar}`}
                            style={{ width: `${percentFull}%` }}
                          />
                        </div>
                      </div>
                    </td>

                    <td className="p-3 text-right">₹{item.price}</td>
                    <td className="p-3 text-right">{(item.margin * 100).toFixed(0)}%</td>
                    <td className="p-3 text-right font-medium text-zinc-500 dark:text-zinc-400">{item.minStock}</td>
                    <td className="p-3 text-right font-medium text-zinc-500 dark:text-zinc-400">{item.rop}</td>
                    <td className="p-3 text-right font-bold text-emerald-600 dark:text-emerald-400">{item.eoq}</td>
                    <td className="p-3 text-right">
                      <span className={`inline-block px-2 py-0.5 rounded text-[9px] font-bold border uppercase tracking-wider ${getRiskBadgeColor(item.risk)}`}>
                        {item.risk}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Informational Footer */}
        <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-850 text-[10px] text-zinc-500 dark:text-zinc-400 flex items-start gap-2 leading-relaxed">
          <Info className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-zinc-700 dark:text-zinc-300 block text-[11px] mb-0.5">Safety stock and ROP calculations details:</span>
            <p>
              Reorder Point (ROP) = (Average Daily Demand × Lead Time) + Safety Stock. Safety Stock buffer protects against demand fluctuations ($\sigma$) at 95% confidence level ($Z=1.96$). Economic Order Quantity (EOQ) optimizes annual holding costs vs vendor ordering charges.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
