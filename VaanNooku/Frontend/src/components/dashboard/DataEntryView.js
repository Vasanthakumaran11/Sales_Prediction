"use client";

import React, { useState, useEffect } from "react";
import {
  FileText,
  Upload,
  Download,
  Save,
  Plus,
  Trash2,
  Calendar,
  Building2,
  DollarSign,
  Package,
  Layers,
  TrendingUp,
  Activity,
  ChevronDown,
  Info,
  CheckCircle,
} from "lucide-react";
import { flattenCatalog } from "@/lib/mock/catalog";
import { PageHeader, Card } from "@/components/ui/Card";
import { STORE_PROFILES } from "@/lib/mock/stores";
import { useStoreContext } from "@/context/StoreContext";

export default function DataEntryView() {
  const { setHistoryLogs } = useStoreContext();
  const catalog = flattenCatalog();

  // Sales Information Form State
  const [salesInfo, setSalesInfo] = useState({
    storeId: STORE_PROFILES[0].id,
    date: new Date().toISOString().split("T")[0],
    dayOfWeek: new Date().toLocaleDateString("en-US", { weekday: "long" }),
    businessType: "Super Market",
    dataSource: "Manual Entry",
    uploadedBy: "Arjun Sharma",
    notes: "",
    paymentMode: "Cash: 50%, UPI: 40%, Card: 10%",
  });

  // Rows of products in the table
  const [rows, setRows] = useState([
    { id: 1, name: "Tata Tea Premium 250g", category: "Beverages", unit: "pcs", price: 150, qty: 25, discount: 5, checked: false },
    { id: 2, name: "Aashirvaad Atta 5kg", category: "Staples & Grains", unit: "pcs", price: 270, qty: 40, discount: 2, checked: false },
    { id: 3, name: "Amul Salted Butter 100g", category: "Perishables", unit: "pcs", price: 68, qty: 60, discount: 0, checked: false },
    { id: 4, name: "Namkeen", category: "Snacks & Biscuits", unit: "pcs", price: 14, qty: 120, discount: 0, checked: false },
    { id: 5, name: "Soap", category: "Personal Care", unit: "pcs", price: 210, qty: 18, discount: 3, checked: false },
  ]);

  const [allChecked, setAllChecked] = useState(false);
  const [showSaveMessage, setShowSaveMessage] = useState(false);

  // Update day of week on date change
  const handleDateChange = (e) => {
    const selectedDate = e.target.value;
    const dateObj = new Date(selectedDate);
    const day = dateObj.toLocaleDateString("en-US", { weekday: "long" });
    setSalesInfo((prev) => ({ ...prev, date: selectedDate, dayOfWeek: day }));
  };

  // Row update handlers
  const handleRowChange = (id, field, value) => {
    setRows((prevRows) =>
      prevRows.map((row) => {
        if (row.id === id) {
          const updated = { ...row, [field]: value };
          // If product name changes, update category and price automatically
          if (field === "name") {
            const matched = catalog.find((p) => p.name === value);
            if (matched) {
              updated.category = matched.category;
              updated.price = matched.price;
            }
          }
          return updated;
        }
        return row;
      })
    );
  };

  const handleAddProduct = () => {
    const defaultProduct = catalog[0] || { name: "New SKU", category: "Staples", price: 100 };
    setRows((prev) => [
      ...prev,
      {
        id: Date.now(),
        name: defaultProduct.name,
        category: defaultProduct.category,
        unit: "pcs",
        price: defaultProduct.price,
        qty: 10,
        discount: 0,
        checked: false,
      },
    ]);
  };

  const handleRemoveSelected = () => {
    setRows((prev) => prev.filter((r) => !r.checked));
    setAllChecked(false);
  };

  const handleToggleCheck = (id) => {
    setRows((prev) =>
      prev.map((r) => (r.id === id ? { ...r, checked: !r.checked } : r))
    );
  };

  const handleToggleAllChecked = () => {
    const nextVal = !allChecked;
    setAllChecked(nextVal);
    setRows((prev) => prev.map((r) => ({ ...r, checked: nextVal })));
  };

  // Calculations
  const calculateRowAmount = (row) => {
    const gross = (parseFloat(row.price) || 0) * (parseInt(row.qty) || 0);
    const disc = (parseFloat(row.discount) || 0) / 100;
    return gross * (1 - disc);
  };

  const grossSales = rows.reduce((sum, r) => sum + (parseFloat(r.price) || 0) * (parseInt(r.qty) || 0), 0);
  const totalDiscount = rows.reduce(
    (sum, r) => sum + (parseFloat(r.price) || 0) * (parseInt(r.qty) || 0) * ((parseFloat(r.discount) || 0) / 100),
    0
  );
  const netSales = grossSales - totalDiscount;
  const taxAmount = netSales * 0.18;
  const rounding = Math.round(netSales + taxAmount) - (netSales + taxAmount);
  const totalPayable = netSales + taxAmount + rounding;

  const totalItemsSold = rows.length;
  const totalQtySold = rows.reduce((sum, r) => sum + (parseInt(r.qty) || 0), 0);
  const avgBillValue = totalItemsSold > 0 ? totalPayable / totalItemsSold : 0;

  const handleSaveAllData = () => {
    // Parse date into readable form (e.g. May 18, 2026)
    const formattedDate = new Date(salesInfo.date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric"
    });

    const newLog = {
      date: formattedDate,
      transactions: totalItemsSold,
      gross: grossSales,
      discount: totalDiscount,
      net: netSales,
      checked: true
    };

    setHistoryLogs((prev) => [newLog, ...prev]);
    setShowSaveMessage(true);
    setTimeout(() => setShowSaveMessage(false), 3000);
  };

  const handleDownloadTemplate = () => {
    const csvContent = "data:text/csv;charset=utf-8,Product Name,Category,Unit,Price,Quantity,Discount (%)\nTata Tea Premium 250g,Beverages,pcs,150,25,5\nAashirvaad Atta 5kg,Staples & Grains,pcs,270,40,2\n";
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "daily_sales_template.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleImportExcel = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".csv,.xlsx,.xls";
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        setRows([
          { id: 101, name: "Nescafe Gold 100g", category: "Beverages", unit: "pcs", price: 320, qty: 50, discount: 10, checked: false },
          { id: 102, name: "Amul Taaza Milk 1L", category: "Dairy & Bakery", unit: "pcs", price: 68, qty: 100, discount: 0, checked: false },
          { id: 103, name: "Surf Excel Matic 1kg", category: "Household Essentials", unit: "pcs", price: 210, qty: 30, discount: 5, checked: false },
        ]);
        alert(`Successfully imported sales logs from file: ${file.name}`);
      }
    };
    input.click();
  };

  return (
    <div className="space-y-6 font-sans px-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-sky-200/60 pb-5 px-2">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight font-serif">
            Daily Sales Data Entry
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Enter and manage your daily sales data for accurate forecasting.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={handleImportExcel}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold rounded-lg transition-all shadow-sm"
          >
            <Upload className="w-3.5 h-3.5 text-slate-500" /> Import from Excel
          </button>
          <button
            onClick={handleDownloadTemplate}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold rounded-lg transition-all shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" /> Download Template <ChevronDown className="w-3 h-3 text-slate-400" />
          </button>
          <button
            onClick={handleSaveAllData}
            className="flex items-center gap-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold rounded-lg transition-all shadow-md"
          >
            <Save className="w-3.5 h-3.5" /> Save All Data
          </button>
        </div>
      </div>

      {showSaveMessage && (
        <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center gap-2 text-xs text-emerald-800 animate-fade-in shadow-sm">
          <CheckCircle className="w-4 h-4 text-emerald-500" />
          <span>All daily sales parameters successfully stored and ingested into the model retraining cache.</span>
        </div>
      )}

      {/* Section 1: Sales Information */}
      <Card className="space-y-4">
        <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider font-serif pb-2 border-b border-slate-100">
          Sales Information
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4 text-xs font-sans">
          <div className="space-y-1">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
              Select Store *
            </label>
            <select
              value={salesInfo.storeId}
              onChange={(e) => setSalesInfo((prev) => ({ ...prev, storeId: e.target.value }))}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-800 focus:outline-none focus:border-sky-500"
            >
              {STORE_PROFILES.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.location})
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
              Select Date *
            </label>
            <div className="relative">
              <input
                type="date"
                value={salesInfo.date}
                onChange={handleDateChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg pl-3 pr-8 py-2 text-xs text-slate-850 focus:outline-none focus:border-sky-500"
              />
              <Calendar className="absolute right-2.5 top-2.5 w-4 h-4 text-slate-400 pointer-events-none" />
            </div>
          </div>

          <div className="space-y-1">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
              Day of Week
            </label>
            <input
              type="text"
              value={salesInfo.dayOfWeek}
              readOnly
              className="w-full bg-slate-100 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-500 cursor-not-allowed"
            />
          </div>

          <div className="space-y-1">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
              Business Type
            </label>
            <select
              value={salesInfo.businessType}
              onChange={(e) => setSalesInfo((prev) => ({ ...prev, businessType: e.target.value }))}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-800 focus:outline-none focus:border-sky-500"
            >
              <option>Super Market</option>
              <option>Grocer</option>
              <option>Hypermarket</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
              Data Source
            </label>
            <select
              value={salesInfo.dataSource}
              onChange={(e) => setSalesInfo((prev) => ({ ...prev, dataSource: e.target.value }))}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-800 focus:outline-none focus:border-sky-500"
            >
              <option>Manual Entry</option>
              <option>POS Sync</option>
              <option>Bulk Import</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Section 2: Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {/* Total Sales */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex items-center gap-3.5 shadow-sm">
          <div className="w-10 h-10 rounded-full bg-blue-50 flex items-center justify-center text-blue-600">
            <DollarSign className="w-5 h-5" />
          </div>
          <div>
            <span className="block text-[9px] text-slate-450 uppercase font-bold tracking-wider">Total Sales (₹)</span>
            <span className="text-sm font-black text-slate-900 leading-none">₹{totalPayable.toLocaleString()}</span>
          </div>
        </div>

        {/* Total Items Sold */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex items-center gap-3.5 shadow-sm">
          <div className="w-10 h-10 rounded-full bg-emerald-50 flex items-center justify-center text-emerald-600">
            <Package className="w-5 h-5" />
          </div>
          <div>
            <span className="block text-[9px] text-slate-450 uppercase font-bold tracking-wider">Total Items Sold</span>
            <span className="text-sm font-black text-slate-900 leading-none">{totalItemsSold}</span>
          </div>
        </div>

        {/* Total Quantity */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex items-center gap-3.5 shadow-sm">
          <div className="w-10 h-10 rounded-full bg-indigo-50 flex items-center justify-center text-indigo-600">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <span className="block text-[9px] text-slate-450 uppercase font-bold tracking-wider">Total Quantity</span>
            <span className="text-sm font-black text-slate-900 leading-none">{totalQtySold}</span>
          </div>
        </div>

        {/* Average Bill Value */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex items-center gap-3.5 shadow-sm">
          <div className="w-10 h-10 rounded-full bg-amber-50 flex items-center justify-center text-amber-605">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <span className="block text-[9px] text-slate-500 uppercase font-bold tracking-wider">Average Bill Value</span>
            <span className="text-sm font-black text-slate-900 leading-none">₹{Math.round(avgBillValue).toLocaleString()}</span>
          </div>
        </div>

        {/* Total Transactions */}
        <div className="bg-white border border-sky-100 rounded-xl p-4 flex items-center gap-3.5 shadow-sm">
          <div className="w-10 h-10 rounded-full bg-sky-50 flex items-center justify-center text-sky-600">
            <Info className="w-5 h-5" />
          </div>
          <div>
            <span className="block text-[9px] text-slate-500 uppercase font-bold tracking-wider">Total Transactions</span>
            <span className="text-sm font-black text-slate-900 leading-none">1,320</span>
          </div>
        </div>
      </div>

      {/* Section 3: Product Sales Details */}
      <Card className="space-y-4">
        <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3 pb-3 border-b border-slate-100">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider font-serif">
            Product Sales Details
          </h3>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleAddProduct}
              className="flex items-center gap-1 px-3 py-1.5 border border-blue-500 text-blue-650 hover:bg-blue-50 text-xs font-bold rounded-lg transition-all"
            >
              <Plus className="w-3.5 h-3.5" /> Add Product
            </button>
            <button
              type="button"
              onClick={handleRemoveSelected}
              disabled={!rows.some((r) => r.checked)}
              className="flex items-center gap-1 px-3 py-1.5 border border-rose-500 text-rose-600 hover:bg-rose-50 text-xs font-bold rounded-lg transition-all disabled:opacity-40"
            >
              <Trash2 className="w-3.5 h-3.5" /> Remove Selected
            </button>
          </div>
        </div>

        {/* Interactive Data entry table */}
        <div className="overflow-x-auto border border-sky-100 rounded-xl bg-white">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-sky-100 text-slate-500 font-bold text-[9px] uppercase tracking-wider">
                <th className="p-3 w-10 text-center">
                  <input
                    type="checkbox"
                    checked={allChecked}
                    onChange={handleToggleAllChecked}
                    className="w-4 h-4 rounded text-blue-600 border-slate-300 focus:ring-blue-500 accent-blue-600"
                  />
                </th>
                <th className="p-3 min-w-[200px]">Product Name / SKU</th>
                <th className="p-3">Category</th>
                <th className="p-3 w-20">Unit</th>
                <th className="p-3 w-28 text-right">Selling Price (₹)</th>
                <th className="p-3 w-24 text-right">Quantity Sold</th>
                <th className="p-3 w-24 text-right">Discount (%)</th>
                <th className="p-3 w-28 text-right">Sales Amount (₹)</th>
                <th className="p-3 w-16 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-sans">
              {rows.map((row) => (
                <tr key={row.id} className="hover:bg-sky-50/10 text-slate-700 transition-colors">
                  <td className="p-3 text-center">
                    <input
                      type="checkbox"
                      checked={row.checked}
                      onChange={() => handleToggleCheck(row.id)}
                      className="w-4 h-4 rounded text-blue-600 border-slate-300 focus:ring-blue-500 accent-blue-600"
                    />
                  </td>
                  <td className="p-3">
                    <select
                      value={row.name}
                      onChange={(e) => handleRowChange(row.id, "name", e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 text-xs text-slate-800 focus:outline-none focus:border-sky-500 font-serif font-semibold"
                    >
                      {catalog.map((catItem) => (
                        <option key={catItem.name} value={catItem.name}>
                          {catItem.name}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="p-3">
                    <input
                      type="text"
                      value={row.category}
                      readOnly
                      className="w-full bg-slate-100 border border-slate-200 rounded px-2.5 py-1.5 text-xs text-slate-500 cursor-not-allowed"
                    />
                  </td>
                  <td className="p-3">
                    <select
                      value={row.unit}
                      onChange={(e) => handleRowChange(row.id, "unit", e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded px-2 py-1.5 text-xs text-slate-800 focus:outline-none"
                    >
                      <option>pcs</option>
                      <option>kg</option>
                      <option>pkts</option>
                      <option>liters</option>
                    </select>
                  </td>
                  <td className="p-3 text-right">
                    <input
                      type="number"
                      value={row.price}
                      onChange={(e) => handleRowChange(row.id, "price", parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 rounded px-2 py-1.5 text-xs text-slate-850 text-right focus:outline-none focus:border-sky-550"
                    />
                  </td>
                  <td className="p-3 text-right">
                    <input
                      type="number"
                      value={row.qty}
                      onChange={(e) => handleRowChange(row.id, "qty", parseInt(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 rounded px-2 py-1.5 text-xs text-slate-850 text-right focus:outline-none focus:border-sky-550"
                    />
                  </td>
                  <td className="p-3 text-right">
                    <input
                      type="number"
                      value={row.discount}
                      onChange={(e) => handleRowChange(row.id, "discount", parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 rounded px-2 py-1.5 text-xs text-slate-850 text-right focus:outline-none focus:border-sky-550"
                    />
                  </td>
                  <td className="p-3 text-right font-semibold text-slate-900">
                    ₹{calculateRowAmount(row).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td className="p-3 text-center">
                    <button
                      type="button"
                      onClick={() => setRows((prev) => prev.filter((r) => r.id !== row.id))}
                      className="p-1 text-slate-400 hover:text-rose-500 rounded hover:bg-rose-50"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Section 4: Bottom Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start font-sans">
        {/* Additional Information */}
        <Card className="space-y-4">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider font-serif pb-2 border-b border-slate-100">
            Additional Information
          </h3>

          <div className="space-y-4 text-xs">
            <div className="space-y-1.5">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                Payment Mode Breakup
              </label>
              <input
                type="text"
                value={salesInfo.paymentMode}
                onChange={(e) => setSalesInfo((prev) => ({ ...prev, paymentMode: e.target.value }))}
                placeholder="e.g. Cash: 70%, Card: 20%, UPI: 10%"
                className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                Notes (Optional)
              </label>
              <textarea
                value={salesInfo.notes}
                onChange={(e) => setSalesInfo((prev) => ({ ...prev, notes: e.target.value }))}
                placeholder="Enter any additional notes about today's sales..."
                className="w-full h-20 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                  Uploaded By
                </label>
                <input
                  type="text"
                  value={salesInfo.uploadedBy}
                  readOnly
                  className="w-full bg-slate-100 border border-slate-200 rounded-lg px-3 py-2 text-slate-500 cursor-not-allowed"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                  Entry Timestamp
                </label>
                <input
                  type="text"
                  value={`${salesInfo.date} 10:30 AM`}
                  readOnly
                  className="w-full bg-slate-100 border border-slate-200 rounded-lg px-3 py-2 text-slate-500 cursor-not-allowed"
                />
              </div>
            </div>
          </div>
        </Card>

        {/* Sales Summary Card */}
        <Card className="space-y-4">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider font-serif pb-2 border-b border-slate-100">
            Sales Summary
          </h3>

          <div className="space-y-3.5 text-xs">
            <div className="flex justify-between items-center py-0.5">
              <span className="text-slate-500">Gross Sales</span>
              <span className="font-semibold text-slate-850">
                ₹{grossSales.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>

            <div className="flex justify-between items-center py-0.5">
              <span className="text-slate-500">Total Discount</span>
              <span className="font-semibold text-rose-600">
                - ₹{totalDiscount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>

            <div className="flex justify-between items-center py-0.5 border-t border-slate-100 pt-2.5">
              <span className="text-slate-505 font-bold">Net Sales</span>
              <span className="font-black text-emerald-600 text-sm">
                ₹{netSales.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>

            <div className="flex justify-between items-center py-0.5">
              <span className="text-slate-505">Tax Amount (18%)</span>
              <span className="font-semibold text-slate-800">
                ₹{taxAmount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>

            <div className="flex justify-between items-center py-0.5">
              <span className="text-slate-505">Rounding Adjustment</span>
              <span className="font-semibold text-slate-800">
                ₹{rounding.toFixed(2)}
              </span>
            </div>

            <div className="flex justify-between items-center py-3 border-t border-sky-100 pt-3">
              <span className="text-slate-900 font-extrabold uppercase tracking-wider text-[11px] font-serif">
                Total Payable
              </span>
              <span className="font-black text-blue-600 text-lg">
                ₹{totalPayable.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
          </div>
        </Card>
      </div>

      {/* Footer controls */}
      <div className="border-t border-sky-200/60 pt-5 flex items-center justify-between font-sans">
        <button
          type="button"
          onClick={() => setRows([])}
          className="px-4 py-2 border border-slate-200 text-slate-600 hover:bg-slate-50 text-xs font-bold rounded-lg transition-all flex items-center gap-1"
        >
          Clear All <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
        </button>
        <button
          onClick={handleSaveAllData}
          className="flex items-center gap-1.5 px-6 py-2.5 bg-blue-600 hover:bg-blue-505 text-white text-xs font-extrabold rounded-lg transition-all shadow-md"
        >
          <Save className="w-4 h-4" /> Save All Data
        </button>
      </div>
    </div>
  );
}
