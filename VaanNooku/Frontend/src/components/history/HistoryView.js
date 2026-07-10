"use client";

import React, { useState } from "react";
import { History, Calendar, Calculator, Check, ArrowRight, DollarSign, BarChart3, Layers, CheckCircle } from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";

export default function HistoryView() {
  // Mock daily history ledger logs
  const [logs, setLogs] = useState([
    { date: "May 17, 2026", transactions: 1320, gross: 24567.70, discount: 1122.20, net: 23445.50, checked: true },
    { date: "May 16, 2026", transactions: 1285, gross: 22890.00, discount: 890.00, net: 22000.00, checked: true },
    { date: "May 15, 2026", transactions: 1410, gross: 27120.00, discount: 1540.00, net: 25580.00, checked: false },
    { date: "May 14, 2026", transactions: 1150, gross: 19850.50, discount: 450.50, net: 19400.00, checked: false },
    { date: "May 13, 2026", transactions: 1290, gross: 23410.00, discount: 1010.00, net: 22400.00, checked: false },
    { date: "May 12, 2026", transactions: 1380, gross: 26180.00, discount: 1200.00, net: 24980.00, checked: false },
    { date: "May 11, 2026", transactions: 1210, gross: 21050.00, discount: 850.00, net: 20200.00, checked: false },
  ]);

  const [allChecked, setAllChecked] = useState(false);

  const handleToggleCheck = (date) => {
    setLogs((prev) =>
      prev.map((log) => (log.date === date ? { ...log, checked: !log.checked } : log))
    );
  };

  const handleToggleAll = () => {
    const nextVal = !allChecked;
    setAllChecked(nextVal);
    setLogs((prev) => prev.map((log) => ({ ...log, checked: nextVal })));
  };

  // Perform financial calculations based on CHECKED records
  const checkedLogs = logs.filter((log) => log.checked);
  const totalTransactions = checkedLogs.reduce((sum, log) => sum + log.transactions, 0);
  const totalGross = checkedLogs.reduce((sum, log) => sum + log.gross, 0);
  const totalDiscount = checkedLogs.reduce((sum, log) => sum + log.discount, 0);
  const totalNet = totalGross - totalDiscount;
  const avgMarginRate = 0.2285; // 22.85% margin
  const projectedProfit = totalNet * avgMarginRate;

  return (
    <div className="space-y-6 font-sans px-6">
      <PageHeader
        title="Historical Sales "
        icon={History}
      />

      {/* Financial Calculator Box */}
      <Card className="bg-sky-50/20 border-sky-100 p-5 space-y-4 flex flex-col justify-between">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-2">
          <Calculator className="w-4 h-4 text-blue-600" />
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-serif">
            Financial Audit Calculator
          </h3>
          <span className="ml-auto text-[9px] font-bold text-slate-400 bg-slate-100 px-2 py-0.5 rounded uppercase font-sans">
            Selected: {checkedLogs.length} Days
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-sans">
          <div className="space-y-1">
            <span className="block text-[9px] text-slate-400 font-bold uppercase">Total Transactions</span>
            <span className="text-base font-black text-slate-900 leading-none">{totalTransactions.toLocaleString()}</span>
          </div>

          <div className="space-y-1">
            <span className="block text-[9px] text-slate-400 font-bold uppercase">Gross Revenue (₹)</span>
            <span className="text-base font-black text-slate-900 leading-none">₹{totalGross.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
          </div>

          <div className="space-y-1">
            <span className="block text-[9px] text-slate-450 font-bold uppercase">Total Discounts (-)</span>
            <span className="text-base font-black text-rose-600 leading-none">- ₹{totalDiscount.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
          </div>

          <div className="space-y-1">
            <span className="block text-[9px] text-slate-455 font-bold uppercase">Net Profits (₹)</span>
            <span className="text-base font-black text-emerald-600 leading-none">₹{projectedProfit.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
          </div>
        </div>

       
      </Card>

      {/* History Ledger Table */}
      <Card className="overflow-x-auto border border-sky-100 rounded-xl bg-white p-0">
        <table className="w-full border-collapse text-left text-xs">
          <thead>
            <tr className="bg-slate-50 border-b border-sky-100 text-slate-500 font-bold text-[9px] uppercase tracking-wider">
              <th className="p-3 w-10 text-center">
                <input
                  type="checkbox"
                  checked={allChecked}
                  onChange={handleToggleAll}
                  className="w-4 h-4 rounded text-blue-600 accent-blue-600"
                />
              </th>
              <th className="p-3">Saled Date</th>
              <th className="p-3 text-right">Transactions Count</th>
              <th className="p-3 text-right">Gross Sales (₹)</th>
              <th className="p-3 text-right">Discount Limit (₹)</th>
              <th className="p-3 text-right">Net Sales (₹)</th>
              <th className="p-3 text-center">Audit Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-sans text-slate-700">
            {logs.map((log) => (
              <tr key={log.date} className="hover:bg-sky-50/10 transition-colors">
                <td className="p-3 text-center">
                  <input
                    type="checkbox"
                    checked={log.checked}
                    onChange={() => handleToggleCheck(log.date)}
                    className="w-4 h-4 rounded text-blue-600 accent-blue-600"
                  />
                </td>
                <td className="p-3 font-semibold text-slate-800">{log.date}</td>
                <td className="p-3 text-right font-semibold text-slate-800">{log.transactions.toLocaleString()}</td>
                <td className="p-3 text-right font-semibold text-slate-800">
                  ₹{log.gross.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </td>
                <td className="p-3 text-right font-semibold text-rose-500">
                  - ₹{log.discount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </td>
                <td className="p-3 text-right font-bold text-slate-900">
                  ₹{log.net.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </td>
                <td className="p-3 text-center">
                  <span className="px-2 py-0.5 rounded bg-emerald-50 border border-emerald-100 text-emerald-600 text-[9px] font-bold uppercase tracking-wider">
                    Synced & Closed
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
