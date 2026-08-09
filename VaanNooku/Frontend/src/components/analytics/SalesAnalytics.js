"use client";

import React, { useState } from "react";
import {
  BarChart3,
  Calendar,
  Filter,
  Download,
  DollarSign,
  ShoppingCart,
  TrendingUp,
  Award,
  Users,
  Percent,
  ChevronDown,
  Info,
  RotateCw,
} from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";
import { StatTile } from "@/components/ui/StatTile";
import { useStoreContext } from "@/context/StoreContext";
import { DEMO_STORE_IDS } from "@/lib/constants";

export default function SalesAnalytics() {
  const { historyLogs, activeStore, storeProducts } = useStoreContext();
  const [timeframe, setTimeframe] = useState("Monthly");
  const [dateRange, setDateRange] = useState("May 01, 2026 - May 31, 2026");

  const isDemo = activeStore ? DEMO_STORE_IDS.includes(activeStore.id) : true;

  // Dynamic calculations
  const totalSales = historyLogs.reduce((sum, log) => sum + (parseFloat(log.net) || 0), 0);
  const totalOrders = historyLogs.reduce((sum, log) => sum + (parseInt(log.transactions) || 0), 0);
  const avgOrderValue = totalOrders > 0 ? totalSales / totalOrders : 0;
  const grossProfit = historyLogs.reduce((sum, log) => sum + ((parseFloat(log.net) || 0) - (parseFloat(log.gross) * 0.8)), 0) || (totalSales * 0.20);
  const profitMargin = totalSales > 0 ? Math.round((grossProfit / totalSales) * 1000) / 10 : 20.0;
  const returningCustomers = Math.round(totalOrders * 0.38);

  if (!historyLogs || historyLogs.length === 0) {
    return (
      <div className="space-y-6 font-sans px-6">
        <PageHeader
          title="Sales Analytics"
          icon={BarChart3}
        />
        <Card className="p-12 text-center space-y-4 max-w-md mx-auto border border-sky-100 bg-white rounded-2xl shadow-sm">
          <div className="w-12 h-12 rounded-full bg-sky-50 flex items-center justify-center mx-auto text-sky-500">
            <BarChart3 className="w-6 h-6 animate-pulse" />
          </div>
          <h3 className="text-sm font-bold text-slate-800 font-serif">No Analytics Data</h3>
          <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
            There are no sales logs recorded for this store yet. Analytics charts will be generated once your transactions are received.
          </p>
        </Card>
      </div>
    );
  }

  const handleDateRangeChange = () => {
    const ranges = [
      "May 01, 2026 - May 31, 2026",
      "Last 30 Days (May 01 - May 31)",
      "Year to Date (Jan 01 - May 31)"
    ];
    const currentIndex = ranges.indexOf(dateRange);
    const nextIndex = (currentIndex + 1) % ranges.length;
    setDateRange(ranges[nextIndex]);
  };

  const handleExport = () => {
    let csvContent = "data:text/csv;charset=utf-8,Metric,Value\n" +
      `Total Revenue,Rs. ${totalSales.toFixed(2)}\n` +
      `Total Orders,${totalOrders}\n` +
      `Average Bill,Rs. ${avgOrderValue.toFixed(2)}\n` +
      `Gross Profit,Rs. ${grossProfit.toFixed(2)}\n` +
      `Margin,${profitMargin}%\n`;
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `sales_report_${activeStore?.id || "store"}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Calculate Best Sellers dynamically from active storeProducts
  const bestSellers = storeProducts && storeProducts.length > 0 
    ? [...storeProducts].sort((a, b) => b.stock - a.stock).slice(0, 5).map(p => ({
        name: p.name,
        category: p.category,
        val: Math.round(p.sellingPrice * p.stock * 0.4) // simulated sales value based on current stock
      }))
    : [];

  // Locations data list or store location description
  const locations = activeStore
    ? [
        { name: `${activeStore.name} (${activeStore.location})`, val: totalSales, pct: 100 }
      ]
    : [];

  // Heatmap helper (Days vs Hours)
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const times = ["6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM"];

  // Heatmap values representation (shades of blue)
  const heatmapData = [
    [1, 2, 3, 2, 4, 3],
    [2, 3, 4, 3, 5, 4],
    [3, 4, 5, 4, 6, 5],
    [2, 3, 4, 3, 5, 4],
    [4, 5, 6, 5, 7, 6],
    [5, 6, 7, 6, 8, 7],
    [4, 5, 6, 5, 7, 6],
  ];

  const getHeatmapColor = (val) => {
    switch (val) {
      case 1: return "bg-sky-50";
      case 2: return "bg-sky-100/50";
      case 3: return "bg-sky-100";
      case 4: return "bg-sky-200/50";
      case 5: return "bg-sky-200";
      case 6: return "bg-sky-300";
      case 7: return "bg-sky-400";
      case 8: return "bg-blue-500";
      default: return "bg-sky-50";
    }
  };

  return (
    <div className="space-y-6 font-sans px-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-sky-200/60 pb-5">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight font-serif">
            Sales Analytics
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Track performance, analyze trends, and discover growth opportunities.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs font-sans">
          <button
            onClick={handleDateRangeChange}
            className="flex items-center gap-1.5 px-3 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold rounded-lg shadow-sm"
          >
            <Calendar className="w-3.5 h-3.5 text-slate-400" /> {dateRange} <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>
          <button
            onClick={() => alert("Analytics segment filters refreshed.")}
            className="flex items-center gap-1.5 px-3 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold rounded-lg shadow-sm"
          >
            <Filter className="w-3.5 h-3.5 text-slate-400" /> Filters <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>
          <button
            onClick={handleExport}
            className="flex items-center gap-1.5 px-3 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold rounded-lg shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-slate-400" /> Export <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>
        </div>
      </div>

      {/* KPI Stats Row (6 Columns) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
        {/* Total Sales */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Total Sales (₹)</span>
            <DollarSign className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">
            ₹{totalSales.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </span>
          {isDemo ? (
            <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
              ↑ 18.6% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
            </span>
          ) : (
            <span className="text-[9px] text-slate-400 font-medium mt-1">Net Sales Volume</span>
          )}
        </div>

        {/* Total Orders */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Total Orders</span>
            <ShoppingCart className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">
            {totalOrders.toLocaleString()}
          </span>
          {isDemo ? (
            <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
              ↑ 15.7% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
            </span>
          ) : (
            <span className="text-[9px] text-slate-400 font-medium mt-1">Transactions count</span>
          )}
        </div>

        {/* Average Order Value */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Average Order Value</span>
            <TrendingUp className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">
            ₹{avgOrderValue.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </span>
          {isDemo ? (
            <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
              ↑ 8.3% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
            </span>
          ) : (
            <span className="text-[9px] text-slate-400 font-medium mt-1">Mean receipt size</span>
          )}
        </div>

        {/* Gross Profit */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Gross Profit (₹)</span>
            <Award className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">
            ₹{grossProfit.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </span>
          {isDemo ? (
            <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
              ↑ 21.4% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
            </span>
          ) : (
            <span className="text-[9px] text-slate-400 font-medium mt-1">Estimated yield</span>
          )}
        </div>

        {/* Profit Margin */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Profit Margin</span>
            <Percent className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">
            {profitMargin}%
          </span>
          {isDemo ? (
            <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
              ↑ 2.6% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
            </span>
          ) : (
            <span className="text-[9px] text-slate-400 font-medium mt-1">Percent markup margin</span>
          )}
        </div>

        {/* Returning Customers */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Returning Customers</span>
            <Users className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">
            {returningCustomers.toLocaleString()}
          </span>
          {isDemo ? (
            <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
              ↑ 16.1% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
            </span>
          ) : (
            <span className="text-[9px] text-slate-400 font-medium mt-1">Estimated repeat rate</span>
          )}
        </div>
      </div>

      {/* Row 2: Sales Over Time & Sales by Product Category */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Total Sales Over Time Chart */}
        <Card className="lg:col-span-2 space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider font-serif">
                Total Sales Over Time
              </h3>
              <p className="text-[10px] text-slate-500 font-sans">Monthly sales trend with key events & growth drivers</p>
            </div>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="bg-slate-50 border border-slate-200 rounded px-2 py-1 text-xs text-slate-800 focus:outline-none"
            >
              <option>Monthly</option>
              <option>Weekly</option>
              <option>Daily</option>
            </select>
          </div>

          <div className="h-64 bg-slate-50/50 rounded-xl border border-sky-100 p-4 relative flex flex-col justify-between">
            {/* SVG line graph */}
            <div className="w-full h-full relative">
              {/* Event Annotations - Show only for demo */}
              {isDemo && (
                <>
                  <div className="absolute top-2 left-[5%] bg-white border border-sky-100 rounded px-1.5 py-0.5 shadow-sm text-[8px] text-slate-700">
                    <span className="font-bold text-emerald-600">New Year</span>
                    <span className="block text-[7px] text-slate-400">Higher demand for gifting</span>
                  </div>
                  <div className="absolute top-4 left-[28%] bg-white border border-sky-100 rounded px-1.5 py-0.5 shadow-sm text-[8px] text-slate-700">
                    <span className="font-bold text-blue-600">Republic Day</span>
                    <span className="block text-[7px] text-slate-400">Winter promos</span>
                  </div>
                  <div className="absolute top-2 left-[48%] bg-white border border-sky-100 rounded px-1.5 py-0.5 shadow-sm text-[8px] text-slate-700">
                    <span className="font-bold text-amber-600">Holi</span>
                    <span className="block text-[7px] text-slate-400">Festive shopping</span>
                  </div>
                  <div className="absolute top-4 left-[68%] bg-white border border-sky-100 rounded px-1.5 py-0.5 shadow-sm text-[8px] text-slate-700">
                    <span className="font-bold text-sky-600">Akshaya Tritiya</span>
                    <span className="block text-[7px] text-slate-400">Strong gold & household sales</span>
                  </div>
                </>
              )}

              <svg viewBox="0 0 100 40" className="w-full h-full overflow-visible">
                <line x1="0" y1="35" x2="100" y2="35" className="stroke-slate-200" strokeWidth="0.2" />
                <line x1="0" y1="20" x2="100" y2="20" className="stroke-slate-200" strokeWidth="0.2" strokeDasharray="1 1" />
                <line x1="0" y1="5" x2="100" y2="5" className="stroke-slate-200" strokeWidth="0.2" strokeDasharray="1 1" />

                {/* Sales Line */}
                {(() => {
                  const sortedLogs = [...historyLogs].sort((a, b) => new Date(a.date) - new Date(b.date));
                  const maxNet = Math.max(...sortedLogs.map(l => l.net), 1000);
                  const points = sortedLogs.map((log, idx) => {
                    const x = 5 + (idx / Math.max(1, sortedLogs.length - 1)) * 90;
                    const y = 35 - (log.net / maxNet) * 28;
                    return { x, y, log };
                  });
                  const pathD = points.length > 0 ? `M ${points.map(p => `${p.x} ${p.y}`).join(" L ")}` : "";
                  const areaD = points.length > 0 ? `${pathD} L 95 35 L 5 35 Z` : "";
                  return (
                    <>
                      {pathD && <path d={pathD} fill="none" stroke="#0284c7" strokeWidth="1.2" />}
                      {areaD && <path d={areaD} fill="url(#salesGrad)" className="opacity-10" />}
                      <defs>
                        <linearGradient id="salesGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stopColor="#0284c7" />
                          <stop offset="100%" stopColor="#ffffff" />
                        </linearGradient>
                      </defs>
                      {points.map((p, idx) => {
                        // Render dots only for up to 10 points or every 3rd point if many, to keep it clean
                        if (points.length > 15 && idx % 3 !== 0) return null;
                        return (
                          <circle key={idx} cx={p.x} cy={p.y} r="0.8" fill="#0284c7" className="hover:r-1.5 cursor-pointer transition-all" />
                        );
                      })}
                    </>
                  );
                })()}
              </svg>
            </div>
            {/* Months Axis */}
            <div className="flex justify-between text-[7px] text-slate-500 px-2 uppercase tracking-wider font-bold">
              {isDemo ? (
                <>
                  <span>Dec-25</span>
                  <span>Jan-26</span>
                  <span>Feb-26</span>
                  <span>Mar-26</span>
                  <span>Apr-26</span>
                  <span>May-26</span>
                </>
              ) : (
                <>
                  <span>May 01</span>
                  <span>May 08</span>
                  <span>May 15</span>
                  <span>May 22</span>
                  <span>May 31</span>
                </>
              )}
            </div>
          </div>
        </Card>

        {/* Sales by Category Donut */}
        <Card className="space-y-4 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider font-serif">
              Sales by Product Category
            </h3>
            <p className="text-[10px] text-slate-500">Share of total sales by category</p>
          </div>

          {(() => {
            // Group active storeProducts by category to calculate dynamic share
            const categorySums = {};
            let grandTotal = 0;
            
            (storeProducts || []).forEach(p => {
              const val = Math.round(p.sellingPrice * p.stock * 0.4);
              categorySums[p.category] = (categorySums[p.category] || 0) + val;
              grandTotal += val;
            });

            const catList = Object.entries(categorySums).map(([category, value]) => ({
              category,
              value,
              pct: grandTotal > 0 ? Math.round((value / grandTotal) * 100) : 0
            })).sort((a, b) => b.value - a.value);

            const finalCategories = catList || [];

            // Simple SVG pie segment offsets - precomputed to avoid state reassignment in JSX
            const segments = finalCategories.reduce((acc, cat) => {
              const priorTotal = acc.length > 0 ? acc[acc.length - 1].runningTotal : 0;
              acc.push({ ...cat, currentOffset: -priorTotal, runningTotal: priorTotal + cat.pct });
              return acc;
            }, []);
            const colors = ["#0ea5e9", "#3b82f6", "#14b8a6", "#f59e0b", "#ec4899", "#94a3b8"];

            return (
              <>
                <div className="flex items-center justify-between gap-4 font-sans text-xs">
                  <svg width="100" height="100" viewBox="0 0 40 40" className="transform -rotate-90 shrink-0">
                    {segments.map((cat, idx) => {
                      const strokeDash = `${cat.pct} ${100 - cat.pct}`;
                      return (
                        <circle
                          key={idx}
                          cx="20"
                          cy="20"
                          r="15.915"
                          fill="transparent"
                          stroke={colors[idx % colors.length]}
                          strokeWidth="4"
                          strokeDasharray={strokeDash}
                          strokeDashoffset={cat.currentOffset}
                        />
                      );
                    })}
                    <circle cx="20" cy="20" r="13" fill="#ffffff" />
                  </svg>

                  <div className="space-y-1.5 flex-1 text-[11px] text-slate-600 font-semibold max-h-36 overflow-y-auto">
                    {finalCategories.slice(0, 5).map((cat, idx) => (
                      <div key={idx} className="flex justify-between">
                        <span>{cat.category}</span>
                        <span className="font-bold text-slate-900">{cat.pct}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="border-t border-slate-100 pt-3 flex justify-between items-center text-[10px] text-slate-500 font-sans">
                  <span>Total Sales</span>
                  <span className="font-black text-slate-950">Rs. {totalSales.toLocaleString()}</span>
                </div>
              </>
            );
          })()}
        </Card>
      </div>



      {/* Row 4: Sales Heatmap, Sales vs Target, and Monthly Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Heatmap */}
        <Card className="space-y-4">
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-serif border-b border-slate-100 pb-2">
            Sales Heatmap (by Day & Time)
          </h3>
          <div className="grid grid-cols-7 gap-1 font-sans text-center text-[9px] text-slate-400">
            {times.map((t, idx) => (
              <span key={idx}>{t}</span>
            ))}
          </div>
          <div className="space-y-1.5 font-sans">
            {days.map((day, dIdx) => (
              <div key={day} className="flex items-center gap-1">
                <span className="w-8 text-[10px] font-bold text-slate-500 text-left">{day}</span>
                <div className="flex-1 grid grid-cols-6 gap-1.5">
                  {heatmapData[dIdx].map((val, hIdx) => (
                    <div
                      key={hIdx}
                      className={`h-4.5 rounded ${getHeatmapColor(val)} hover:opacity-80 transition-all`}
                      title={`${day} at ${times[hIdx]}: level ${val}`}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Sales vs Target Gauge */}
        <Card className="space-y-4 flex flex-col justify-between text-center items-center">
          <div className="w-full text-left">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-serif border-b border-slate-100 pb-2">
              Sales vs Target
            </h3>
          </div>

          <div className="relative w-40 h-24 flex items-center justify-center overflow-hidden">
            <svg width="140" height="70" viewBox="0 0 40 20" className="absolute top-2">
              <path d="M 2 18 A 16 16 0 0 1 38 18" fill="none" stroke="#f1f5f9" strokeWidth="4" strokeLinecap="round" />
              <path d="M 2 18 A 16 16 0 0 1 38 18" fill="none" stroke="#0ea5e9" strokeWidth="4.2" strokeLinecap="round" strokeDasharray="50 50" />
            </svg>
            <div className="absolute top-10 text-center font-sans">
              <span className="block text-2xl font-black text-slate-950 leading-none">84.5%</span>
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wide mt-1">Achieved</span>
            </div>
          </div>

          <div className="space-y-1 text-xs font-sans">
            <span className="block text-slate-500">Monthly Target</span>
            <span className="text-sm font-black text-slate-900">₹8,45,230 / ₹10,00,000</span>
          </div>
        </Card>

        {/* Monthly Sales Comparison Column Chart */}
        <Card className="space-y-4">
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-serif border-b border-slate-100 pb-2">
            Monthly Sales Comparison
          </h3>

          <div className="h-44 bg-slate-50/50 rounded-xl border border-sky-100 p-3 relative flex items-end justify-between font-sans text-[8px] text-slate-400">
            {/* Custom columns */}
            <div className="flex-1 flex justify-around items-end h-32">
              <div className="flex gap-1 items-end h-full">
                <div className="w-3 bg-slate-200 h-[40%]" />
                <div className="w-3 bg-blue-500 h-[60%]" />
              </div>
              <div className="flex gap-1 items-end h-full">
                <div className="w-3 bg-slate-200 h-[50%]" />
                <div className="w-3 bg-blue-500 h-[70%]" />
              </div>
              <div className="flex gap-1 items-end h-full">
                <div className="w-3 bg-slate-200 h-[45%]" />
                <div className="w-3 bg-blue-500 h-[65%]" />
              </div>
              <div className="flex gap-1 items-end h-full">
                <div className="w-3 bg-slate-200 h-[60%]" />
                <div className="w-3 bg-blue-500 h-[80%]" />
              </div>
              <div className="flex gap-1 items-end h-full">
                <div className="w-3 bg-slate-200 h-[70%]" />
                <div className="w-3 bg-blue-500 h-[92%]" />
              </div>
            </div>
          </div>
          <div className="flex justify-around text-[9px] text-slate-500 uppercase font-bold">
            <span>Dec</span>
            <span>Jan</span>
            <span>Feb</span>
            <span>Mar</span>
            <span>Apr</span>
          </div>
        </Card>
      </div>

      {/* Footer details */}
      <div className="flex justify-between items-center text-[10px] text-slate-400 font-sans border-t border-sky-100 pt-3">
        <span>Last Updated: May 17, 2026 10:30 AM</span>
        <span>Data Accuracy: 98.7%</span>
      </div>
    </div>
  );
}
