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

export default function SalesAnalytics() {
  const [timeframe, setTimeframe] = useState("Monthly");

  // Top 5 Products list
  const bestSellers = [
    { name: "Tata Tea Premium 250g", category: "Beverages", val: 25620 },
    { name: "Aashirvaad Atta 5kg", category: "Staples & Grains", val: 22480 },
    { name: "Amul Salted Butter 100g", category: "Dairy & Eggs", val: 18750 },
    { name: "Surf Excel Matic 1kg", category: "Household Essentials", val: 15320 },
    { name: "Namkeen", category: "Snacks & Branded Foods", val: 12980 },
  ];

  // Locations data list
  const locations = [
    { name: "T. Nagar, Chennai", val: 124560, pct: 100 },
    { name: "Indiranagar, Bangalore", val: 105230, pct: 85 },
    { name: "Bandra, Mumbai", val: 98750, pct: 80 },
    { name: "Koramangala, Bangalore", val: 87450, pct: 70 },
    { name: "Whitefield, Bangalore", val: 76320, pct: 60 },
  ];

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
          <button className="flex items-center gap-1.5 px-3 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold rounded-lg shadow-sm">
            <Calendar className="w-3.5 h-3.5 text-slate-400" /> Jan 01, 2026 - May 17, 2026 <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>
          <button className="flex items-center gap-1.5 px-3 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold rounded-lg shadow-sm">
            <Filter className="w-3.5 h-3.5 text-slate-400" /> Filters <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>
          <button className="flex items-center gap-1.5 px-3 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold rounded-lg shadow-sm">
            <Download className="w-3.5 h-3.5 text-slate-400" /> Export <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>
        </div>
      </div>

      {/* KPI Stats Row (6 Columns) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
        {/* Total Sales */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Total Sales (₹)</span>
            <DollarSign className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">₹8,45,230.50</span>
          <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
            ↑ 18.6% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
          </span>
        </div>

        {/* Total Orders */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Total Orders</span>
            <ShoppingCart className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">12,850</span>
          <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
            ↑ 15.7% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
          </span>
        </div>

        {/* Average Order Value */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Average Order Value</span>
            <TrendingUp className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">₹658.45</span>
          <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
            ↑ 8.3% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
          </span>
        </div>

        {/* Gross Profit */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Gross Profit (₹)</span>
            <Award className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">₹1,93,240.75</span>
          <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
            ↑ 21.4% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
          </span>
        </div>

        {/* Profit Margin */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Profit Margin</span>
            <Percent className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">22.85%</span>
          <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
            ↑ 2.6% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
          </span>
        </div>

        {/* Returning Customers */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex flex-col gap-1 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase">Returning Customers</span>
            <Users className="w-4 h-4 text-blue-600" />
          </div>
          <span className="text-base font-black text-slate-900 leading-none mt-1">4,235</span>
          <span className="text-[9px] text-emerald-600 font-extrabold flex items-center gap-0.5 mt-1">
            ↑ 16.1% <span className="text-slate-400 font-normal">vs Dec 01 - Apr 30</span>
          </span>
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
              {/* Event Annotations */}
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

              <svg viewBox="0 0 100 40" className="w-full h-full overflow-visible">
                <line x1="0" y1="35" x2="100" y2="35" className="stroke-slate-200" strokeWidth="0.2" />
                <line x1="0" y1="20" x2="100" y2="20" className="stroke-slate-200" strokeWidth="0.2" strokeDasharray="1 1" />
                <line x1="0" y1="5" x2="100" y2="5" className="stroke-slate-200" strokeWidth="0.2" strokeDasharray="1 1" />

                {/* Sales Line */}
                <path d="M 5 28 Q 23 23 45 27 T 85 15 T 95 10" fill="none" stroke="#0284c7" strokeWidth="1.2" />
                <path d="M 5 28 Q 23 23 45 27 T 85 15 T 95 10 L 95 35 L 5 35 Z" fill="url(#salesGrad)" className="opacity-10" />

                <defs>
                  <linearGradient id="salesGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#0284c7" />
                    <stop offset="100%" stopColor="#ffffff" />
                  </linearGradient>
                </defs>

                {/* Dots */}
                <circle cx="5" cy="28" r="0.8" fill="#0284c7" />
                <circle cx="23" cy="23" r="0.8" fill="#0284c7" />
                <circle cx="45" cy="27" r="0.8" fill="#0284c7" />
                <circle cx="68" cy="20" r="0.8" fill="#0284c7" />
                <circle cx="85" cy="15" r="0.8" fill="#0284c7" />
                <circle cx="95" cy="10" r="0.8" fill="#0284c7" />
              </svg>
            </div>
            {/* Months Axis */}
            <div className="flex justify-between text-[8px] text-slate-500 px-2 uppercase tracking-wider font-bold">
              <span>Dec-25</span>
              <span>Jan-26</span>
              <span>Feb-26</span>
              <span>Mar-26</span>
              <span>Apr-26</span>
              <span>May-26</span>
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

          <div className="flex items-center justify-between gap-4 font-sans text-xs">
            <svg width="100" height="100" viewBox="0 0 40 40" className="transform -rotate-90 shrink-0">
              <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#0ea5e9" strokeWidth="4" strokeDasharray="32 68" strokeDashoffset="0" />
              <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#3b82f6" strokeWidth="4" strokeDasharray="22 78" strokeDashoffset="-32" />
              <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#14b8a6" strokeWidth="4" strokeDasharray="15 85" strokeDashoffset="-54" />
              <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#f59e0b" strokeWidth="4" strokeDasharray="12 88" strokeDashoffset="-69" />
              <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#ec4899" strokeWidth="4" strokeDasharray="10 90" strokeDashoffset="-81" />
              <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#94a3b8" strokeWidth="4" strokeDasharray="9 91" strokeDashoffset="-91" />
              <circle cx="20" cy="20" r="13" fill="#ffffff" />
            </svg>

            <div className="space-y-1.5 flex-1 text-[11px] text-slate-600 font-semibold">
              <div className="flex justify-between">
                <span>Staples & Grocery</span>
                <span className="font-bold text-slate-900">32%</span>
              </div>
              <div className="flex justify-between">
                <span>Beverages</span>
                <span className="font-bold text-slate-900">22%</span>
              </div>
              <div className="flex justify-between">
                <span>Dairy & Eggs</span>
                <span className="font-bold text-slate-900">15%</span>
              </div>
              <div className="flex justify-between">
                <span>Snacks & Branded Foods</span>
                <span className="font-bold text-slate-900">12%</span>
              </div>
              <div className="flex justify-between">
                <span>Personal Care</span>
                <span className="font-bold text-slate-900">10%</span>
              </div>
            </div>
          </div>

          <div className="border-t border-slate-100 pt-3 flex justify-between items-center text-[10px] text-slate-500 font-sans">
            <span>Total Sales</span>
            <span className="font-black text-slate-950">₹8,45,230.50</span>
          </div>
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
