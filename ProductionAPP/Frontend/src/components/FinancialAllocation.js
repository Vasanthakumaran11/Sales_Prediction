import React from 'react';
import { 
  DollarSign, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle, 
  Percent, 
  ArrowRight, 
  TrendingDown 
} from 'lucide-react';

export default function FinancialAllocation({ storeInfo }) {
  const store = storeInfo || {
    name: 'Executive Dashboard',
    type: 'Supermarket',
    location: 'Urban',
    investment: 850000,
    openingMonth: 'October'
  };

  // Capital Efficiency category allocations
  const categoriesAlloc = [
    { name: 'Staples & Grains', allocated: 250000, demandVal: 290000, efficiency: 'Optimal', pct: 95 },
    { name: 'Beverages', allocated: 180000, demandVal: 150000, efficiency: 'Optimal', pct: 83 },
    { name: 'Snacks & Biscuits', allocated: 190000, demandVal: 180000, efficiency: 'Optimal', pct: 94 },
    { name: 'Perishables', allocated: 150000, demandVal: 90000, efficiency: 'Over-allocated', pct: 60, isLeak: true },
    { name: 'Personal Care', allocated: 80000, demandVal: 25000, efficiency: 'Critical Surplus', pct: 31, isLeak: true },
  ];

  // Calculations for ROI Maximizer
  const weeklySales = Math.round(store.investment * 0.08); // Estimate weekly demand
  const marginAmt = Math.round(weeklySales * 0.22); // average 22% margin
  const weeklyCosts = weeklySales - marginAmt;
  const roi = ((marginAmt / weeklyCosts) * 100).toFixed(1);

  // Total cash lock estimation
  const totalLockedCash = categoriesAlloc
    .filter(c => c.isLeak)
    .reduce((acc, curr) => acc + (curr.allocated - curr.demandVal), 0);

  return (
    <div className="space-y-6 font-sans">
      {/* Top Header Row */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-zinc-900 dark:text-white tracking-tight flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-emerald-500" /> Financial ROI & Allocation Tracker
          </h2>
          <p className="text-xs text-zinc-550 dark:text-zinc-400 mt-0.5">
            Evaluate investment returns, projected cash flows, and dead stock capital lock-up metrics.
          </p>
        </div>
      </div>

      {/* Financial Overview Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Profit Maximization Engine */}
        <div className="lg:col-span-1 bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 flex flex-col justify-between gap-5 shadow-sm">
          <div className="space-y-1">
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white uppercase tracking-wider">Profit Maximizer</h3>
            <p className="text-[10px] text-zinc-550">Projected weekly financial returns based on current demand forecasts.</p>
          </div>

          <div className="space-y-4">
            {/* ROI Metric Badge */}
            <div className="p-4 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 flex justify-between items-center">
              <div>
                <span className="block text-[10px] text-zinc-450 dark:text-zinc-500 uppercase font-semibold">Weekly ROI Estimate</span>
                <span className="text-2xl font-black text-zinc-900 dark:text-white">+{roi}%</span>
              </div>
              <span className="p-2 py-1 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-bold flex items-center gap-1">
                <TrendingUp className="w-3.5 h-3.5" /> High Margin
              </span>
            </div>

            {/* Calculations breakdown */}
            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between border-b border-zinc-150 dark:border-zinc-850 pb-2">
                <span className="text-zinc-500 dark:text-zinc-405">Projected Demand Intake</span>
                <span className="font-semibold text-zinc-900 dark:text-white">₹{weeklySales.toLocaleString()}</span>
              </div>
              <div className="flex justify-between border-b border-zinc-150 dark:border-zinc-850 pb-2">
                <span className="text-zinc-500 dark:text-zinc-405">Procurement Cost Basis</span>
                <span className="font-semibold text-zinc-700 dark:text-zinc-350">₹{weeklyCosts.toLocaleString()}</span>
              </div>
              <div className="flex justify-between border-b border-zinc-150 dark:border-zinc-850 pb-2">
                <span className="text-zinc-500 dark:text-zinc-405">Projected Weekly Profit</span>
                <span className="font-semibold text-emerald-605 dark:text-emerald-400">₹{marginAmt.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500 dark:text-zinc-405">Average Net margin</span>
                <span className="font-bold text-zinc-900 dark:text-white">22.0%</span>
              </div>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/15 border border-emerald-100 dark:border-emerald-900/30 text-[10px] text-emerald-800 dark:text-emerald-300">
            💡 Demand forecasting suggests optimization of Beverages can bump margins by an additional 1.8% this week.
          </div>
        </div>

        {/* Right 2 Columns: Capital Allocation Efficiency */}
        <div className="lg:col-span-2 bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 flex flex-col gap-5 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-150 dark:border-zinc-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white uppercase tracking-wider">Capital Allocation efficiency</h3>
              <p className="text-[10px] text-zinc-500">Real-time allocation tracker. Alerts for dead stock locking cash-flow.</p>
            </div>
            {totalLockedCash > 0 && (
              <span className="px-2.5 py-0.5 rounded-full bg-rose-50 dark:bg-rose-500/10 border border-rose-100 dark:border-rose-500/20 text-rose-600 dark:text-rose-400 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1">
                <AlertTriangle className="w-3.5 h-3.5 text-rose-500" /> ₹{totalLockedCash.toLocaleString()} Locked Capital
              </span>
            )}
          </div>

          {/* Allocation bars matrix */}
          <div className="space-y-4">
            {categoriesAlloc.map(cat => {
              const isLeak = cat.isLeak;
              const surplusVal = Math.max(0, cat.allocated - cat.demandVal);
              
              return (
                <div key={cat.name} className="p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-850/80 grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
                  {/* Category Name & Status */}
                  <div className="md:col-span-1">
                    <span className="font-bold text-zinc-900 dark:text-white block text-xs">{cat.name}</span>
                    <span className={`inline-block px-1.5 py-0.5 rounded text-[8px] font-bold border uppercase tracking-wider mt-1.5 ${
                      cat.efficiency === 'Optimal' 
                        ? 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-100 dark:border-emerald-500/20' 
                        : isLeak && cat.efficiency === 'Critical Surplus'
                          ? 'bg-rose-50 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-100 dark:border-rose-500/25 animate-pulse'
                          : 'bg-amber-50 dark:bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-100 dark:border-amber-500/20'
                    }`}>
                      {cat.efficiency}
                    </span>
                  </div>

                  {/* Allocation Level Slider/Bar */}
                  <div className="md:col-span-2 space-y-1">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="text-zinc-500">Allocated: <span className="font-semibold text-zinc-700 dark:text-zinc-350">₹{cat.allocated.toLocaleString()}</span></span>
                      <span className="text-zinc-500">Demand: <span className="font-semibold text-zinc-700 dark:text-zinc-350">₹{cat.demandVal.toLocaleString()}</span></span>
                    </div>
                    
                    {/* Visual overlap bar */}
                    <div className="w-full h-2 bg-zinc-200 dark:bg-zinc-900 rounded-full overflow-hidden relative">
                      <div 
                        className={`h-full rounded-full ${isLeak ? 'bg-amber-500' : 'bg-emerald-500'}`}
                        style={{ width: `${cat.pct}%` }}
                      />
                    </div>
                  </div>

                  {/* Efficiency Warning Detail */}
                  <div className="md:col-span-1 text-right text-xs">
                    {isLeak ? (
                      <div className="space-y-0.5">
                        <span className="text-rose-600 dark:text-rose-400 font-bold block">- ₹{surplusVal.toLocaleString()}</span>
                        <span className="text-[9px] text-zinc-500">Cash Flow Leakage</span>
                      </div>
                    ) : (
                      <div className="space-y-0.5">
                        <span className="text-emerald-600 dark:text-emerald-400 font-bold block flex items-center justify-end gap-1">
                          <CheckCircle className="w-3.5 h-3.5" /> 100%
                        </span>
                        <span className="text-[9px] text-zinc-500">Allocation Utility</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="p-3.5 rounded-lg bg-rose-50 dark:bg-rose-950/15 border border-rose-100 dark:border-rose-900/30 text-[10px] text-rose-800 dark:text-zinc-400 leading-relaxed flex items-start gap-2">
            <TrendingDown className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-rose-600 dark:text-rose-400 block uppercase tracking-wider mb-0.5">Over-allocation Risk Alert:</span>
              Excess allocation locks valuable investment capital into dead stocks, blocking capital turnover. Demand planning algorithms recommend reducing Personal Care orders and redirecting ₹55,000 back to high-turnover Staples.
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
