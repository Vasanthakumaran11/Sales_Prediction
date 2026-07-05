import React, { useState } from 'react';
import { 
  FileText, 
  PlusCircle, 
  Database, 
  RotateCw, 
  Search, 
  Check, 
  Clock, 
  Tag, 
  HelpCircle 
} from 'lucide-react';
import { PRODUCT_CATALOG } from './mockData';

export default function TransactionsLog() {
  // Extract all products into a flat array for search/selection
  const productsList = Object.entries(PRODUCT_CATALOG).flatMap(([category, items]) => {
    return Object.entries(items).map(([name, detail]) => ({
      name,
      category,
      price: detail.price,
      margin: detail.margin
    }));
  });

  const [selectedProduct, setSelectedProduct] = useState(productsList[0]);
  const [unitsSold, setUnitsSold] = useState('5');
  const [discountPercent, setDiscountPercent] = useState('0');
  const [promoFlag, setPromoFlag] = useState(false);
  const [holidayFlag, setHolidayFlag] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchDropdown, setShowSearchDropdown] = useState(false);
  
  // Completed transaction logs list
  const [logs, setLogs] = useState([
    { id: 104, timestamp: '19:42:10', name: 'Curd', qty: 2, price: 45, discount: 0, total: 90, syncState: 'synced', promo: false },
    { id: 103, timestamp: '18:15:32', name: 'Aashirvaad Shudh Chakki Atta 5kg', qty: 1, price: 45, discount: 5, total: 43, syncState: 'synced', promo: true },
    { id: 102, timestamp: '15:22:18', name: 'Tata Tea Premium 250g', qty: 3, price: 250, discount: 0, total: 750, syncState: 'synced', promo: false },
    { id: 101, timestamp: '11:05:44', name: 'Eggs', qty: 12, price: 45, discount: 10, total: 486, syncState: 'synced', promo: false }
  ]);

  const [isSubmitting, setIsSubmitting] = useState(false);

  // Filtered product selection
  const filteredSearchList = productsList.filter(p => 
    p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelectProduct = (product) => {
    setSelectedProduct(product);
    setSearchQuery(product.name);
    setShowSearchDropdown(false);
  };

  const handleLogSubmit = (e) => {
    e.preventDefault();
    if (!unitsSold || parseInt(unitsSold) <= 0) return;

    setIsSubmitting(true);

    const price = selectedProduct.price;
    const qty = parseInt(unitsSold);
    const disc = parseFloat(discountPercent) || 0;
    const totalAmount = Math.round((price * qty) * (1 - disc / 100));

    // Simulate database transaction delay
    setTimeout(() => {
      const now = new Date();
      const timestamp = now.toTimeString().split(' ')[0];
      
      const newLog = {
        id: Math.floor(Math.random() * 1000) + 200,
        timestamp,
        name: selectedProduct.name,
        qty,
        price,
        discount: disc,
        total: totalAmount,
        syncState: 'syncing', // Starts as syncing then updates to synced
        promo: promoFlag
      };

      setLogs(prev => [newLog, ...prev]);
      setIsSubmitting(false);

      // Reset form variables
      setUnitsSold('5');
      setDiscountPercent('0');
      setPromoFlag(false);
      setHolidayFlag(false);

      // Transition log to synced after 3 seconds
      setTimeout(() => {
        setLogs(currentLogs => 
          currentLogs.map(l => l.id === newLog.id ? { ...l, syncState: 'synced' } : l)
        );
      }, 3000);

    }, 800);
  };

  return (
    <div className="space-y-6 font-sans">
      {/* Top Header Row */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <FileText className="w-5 h-5 text-emerald-500" /> Daily Transactions Log
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Log daily transactions below to ingest active sales data into the postgresql database cache.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        
        {/* Left Side: Data Entry Panel */}
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 space-y-5">
          <div className="border-b border-zinc-800 pb-3 flex items-center gap-2 text-white">
            <PlusCircle className="w-4 h-4 text-emerald-500" />
            <h3 className="text-sm font-bold uppercase tracking-wider">Record Sales Entry</h3>
          </div>

          <form onSubmit={handleLogSubmit} className="space-y-4">
            {/* Search autocomplete input */}
            <div className="space-y-1.5 relative">
              <label className="text-xs font-semibold text-zinc-450 uppercase tracking-wider block">Product Name / Item Code</label>
              <div className="relative">
                <input
                  id="input-product-search"
                  type="text"
                  placeholder="Type product name, e.g. Atta, Oil..."
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setShowSearchDropdown(true);
                  }}
                  onFocus={() => setShowSearchDropdown(true)}
                  className="w-full bg-zinc-950 border border-zinc-850 rounded-lg pl-9 pr-4 py-2.5 text-xs text-white focus:outline-none focus:border-emerald-500"
                  required
                />
                <Search className="absolute left-3 top-3 w-4.5 h-4.5 text-zinc-500" />
              </div>

              {/* Autocomplete Dropdown list */}
              {showSearchDropdown && searchQuery && (
                <div className="absolute z-30 left-0 w-full mt-1 bg-zinc-900 border border-zinc-800 rounded-lg shadow-2xl max-h-48 overflow-y-auto divide-y divide-zinc-850">
                  {filteredSearchList.length > 0 ? (
                    filteredSearchList.map(p => (
                      <button
                        key={p.name}
                        type="button"
                        onClick={() => handleSelectProduct(p)}
                        className="w-full text-left px-4 py-2 hover:bg-zinc-850 text-xs text-zinc-300 flex justify-between items-center"
                      >
                        <span className="font-semibold text-white">{p.name}</span>
                        <span className="text-[10px] text-zinc-500 uppercase">{p.category} (₹{p.price})</span>
                      </button>
                    ))
                  ) : (
                    <div className="px-4 py-3 text-xs text-zinc-500 text-center">No products found.</div>
                  )}
                </div>
              )}
            </div>

            {/* Read-Only Selected Summary */}
            {selectedProduct && (
              <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-900 flex justify-between items-center text-xs text-zinc-400">
                <div>
                  <span className="text-[10px] uppercase font-semibold text-zinc-500">Selected SKU Details</span>
                  <span className="block font-bold text-white text-xs">{selectedProduct.name}</span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] uppercase font-semibold text-zinc-500 block">Unit Price</span>
                  <span className="font-bold text-emerald-400">₹{selectedProduct.price}</span>
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider block">Units Sold</label>
                <input
                  id="input-units-sold"
                  type="number"
                  value={unitsSold}
                  onChange={(e) => setUnitsSold(e.target.value)}
                  min="1"
                  className="w-full bg-zinc-950 border border-zinc-850 rounded-lg px-4 py-2.5 text-xs text-white focus:outline-none focus:border-emerald-500"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider block">Discount (%)</label>
                <input
                  id="input-discount"
                  type="number"
                  value={discountPercent}
                  onChange={(e) => setDiscountPercent(e.target.value)}
                  min="0"
                  max="100"
                  className="w-full bg-zinc-950 border border-zinc-850 rounded-lg px-4 py-2.5 text-xs text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>

            {/* Custom parameters flags */}
            <div className="grid grid-cols-2 gap-4 pt-2">
              <label className="flex items-center gap-3 p-3 bg-zinc-950 rounded-lg border border-zinc-850 cursor-pointer select-none">
                <input
                  id="checkbox-promo"
                  type="checkbox"
                  checked={promoFlag}
                  onChange={(e) => setPromoFlag(e.target.checked)}
                  className="w-4 h-4 rounded text-emerald-500 bg-zinc-900 border-zinc-800 focus:ring-emerald-500 focus:ring-offset-0 focus:ring-0 accent-emerald-500"
                />
                <div className="text-left">
                  <span className="block text-xs font-bold text-white">Promo Flag</span>
                  <span className="text-[9px] text-zinc-550 block">Mark as active discount</span>
                </div>
              </label>

              <label className="flex items-center gap-3 p-3 bg-zinc-950 rounded-lg border border-zinc-850 cursor-pointer select-none">
                <input
                  id="checkbox-holiday"
                  type="checkbox"
                  checked={holidayFlag}
                  onChange={(e) => setHolidayFlag(e.target.checked)}
                  className="w-4 h-4 rounded text-emerald-500 bg-zinc-900 border-zinc-800 focus:ring-emerald-500 focus:ring-offset-0 focus:ring-0 accent-emerald-500"
                />
                <div className="text-left">
                  <span className="block text-xs font-bold text-white">Holiday Flag</span>
                  <span className="text-[9px] text-zinc-550 block">School/Federal holiday</span>
                </div>
              </label>
            </div>

            {/* Submit btn */}
            <button
              id="btn-submit-sale"
              type="submit"
              disabled={isSubmitting || !selectedProduct}
              className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs rounded-lg transition-all flex items-center justify-center gap-1.5 shadow-lg shadow-emerald-950/20 disabled:opacity-40"
            >
              {isSubmitting ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" /> Ingesting to local cache...
                </>
              ) : (
                <>
                  <Database className="w-4 h-4 text-emerald-300" /> Submit Sale Transaction
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right Side: Ledger of Completed Logs */}
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 flex flex-col h-[460px] justify-between">
          <div className="space-y-1 border-b border-zinc-800 pb-3">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Completed Transaction Ledger</h3>
            <p className="text-[10px] text-zinc-500">Live scrolling audit. Badges track PostgreSQL relational cache state.</p>
          </div>

          <div className="flex-1 overflow-y-auto py-4 space-y-3 pr-1">
            {logs.map(log => (
              <div 
                key={log.id} 
                className="p-3.5 rounded-xl bg-zinc-950 border border-zinc-850/80 hover:border-zinc-800 transition-all flex justify-between items-center text-xs relative overflow-hidden"
              >
                {/* Visual state vertical border */}
                <div className={`absolute left-0 top-0 h-full w-1 ${log.syncState === 'synced' ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'}`} />

                <div className="space-y-1 pl-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-zinc-100">{log.name}</span>
                    {log.promo && (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 bg-purple-500/10 border border-purple-500/20 text-purple-400 text-[8px] font-bold uppercase rounded">
                        <Tag className="w-2 h-2" /> Promo
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5 text-[9px] text-zinc-500">
                    <Clock className="w-3.5 h-3.5" />
                    <span>Logged at {log.timestamp}</span>
                    <span>•</span>
                    <span>Qty: {log.qty} units</span>
                  </div>
                </div>

                <div className="text-right space-y-1.5">
                  <span className="font-black text-white block">₹{log.total.toLocaleString()}</span>
                  
                  {log.syncState === 'synced' ? (
                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[9px] border border-emerald-500/10 font-bold tracking-wide uppercase">
                      <Check className="w-2.5 h-2.5 text-emerald-500" /> Synced to PG Cache
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 text-[9px] border border-amber-500/15 font-bold tracking-wide uppercase animate-pulse">
                      <RotateCw className="w-2.5 h-2.5 text-amber-500 animate-spin" /> Syncing...
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-850 text-[10px] text-zinc-500 flex items-center justify-between mt-2">
            <span>Database cache active: <strong>PostgreSQL Cache</strong></span>
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          </div>
        </div>

      </div>
    </div>
  );
}
