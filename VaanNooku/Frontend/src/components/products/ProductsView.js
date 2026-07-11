"use client";

import React, { useState } from "react";
import {
  Package,
  Layers,
  TrendingUp,
  AlertTriangle,
  FileSpreadsheet,
  Plus,
  Search,
  Filter,
  Eye,
  Edit2,
  Copy,
  MoreVertical,
  ChevronDown,
  Sparkles,
  ArrowUpRight,
  TrendingDown,
  XCircle,
  HelpCircle,
  ArrowRight,
  CheckCircle,
} from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";
import { flattenCatalog } from "@/lib/mock/catalog";

import { useStoreContext } from "@/context/StoreContext";

export default function ProductsView() {
  const { storeProducts, setStoreProducts, activeStore } = useStoreContext();
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All Categories");
  const [selectedBrand, setSelectedBrand] = useState("All Brands");
  const [currentPage, setCurrentPage] = useState(1);

  // Link to context products state
  const products = storeProducts || [];
  const setProducts = setStoreProducts;

  // Mock catalog loaded
  const baseCatalog = flattenCatalog();

  if (products.length === 0) {
    return (
      <div className="space-y-6 font-sans px-6">
        <PageHeader
          title="Products Catalog"
          icon={Package}
        />
        <Card className="p-12 text-center space-y-4 max-w-md mx-auto border border-sky-100 bg-white rounded-2xl shadow-sm">
          <div className="w-12 h-12 rounded-full bg-sky-50 flex items-center justify-center mx-auto text-sky-500">
            <Package className="w-6 h-6 animate-pulse" />
          </div>
          <h3 className="text-sm font-bold text-slate-850 font-serif">No Products in Catalog</h3>
          <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
            Your store catalog is empty. Place a replenishment order in the <strong>Inventory</strong> panel to receive your first product delivery.
          </p>
        </Card>
      </div>
    );
  }

  const demoIds = ["balaji-store", "shiva-stores", "surya-markets"];
  const isDemo = activeStore ? demoIds.includes(activeStore.id) : true;

  // Dynamic metrics computations
  const totalProducts = products.length;
  const categoriesCount = new Set(products.map((p) => p.category)).size;
  const totalInventoryValue = products.reduce((sum, p) => sum + (parseFloat(p.buyingPrice) || 0) * (parseInt(p.stock) || 0), 0);
  const avgMargin = products.length > 0 ? products.reduce((sum, p) => sum + (parseFloat(p.margin) || 0), 0) / products.length : 0;
  const lowStockCount = products.filter((p) => p.stock > 0 && p.stock < 20).length;
  const outOfStockCount = products.filter((p) => p.stock === 0).length;
  const discontinuedCount = 0;

  // Recently updated items list
  const updates = [
    { name: "Amul Butter 100g", desc: "Price Updated", time: "2 min ago" },
    { name: "Aashirvaad Atta 5kg", desc: "Stock Updated", time: "15 min ago" },
    { name: "Colgate MaxFresh 150g", desc: "Details Updated", time: "1 hour ago" },
    { name: "Tata Salt 1kg", desc: "Price Updated", time: "2 hours ago" },
  ];

  const filtered = products.filter((p) => {
    const matchesSearch = p.name.toLowerCase().includes(searchTerm.toLowerCase()) || p.sku.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCat = selectedCategory === "All Categories" || p.category === selectedCategory;
    const matchesBrand = selectedBrand === "All Brands" || p.brand === selectedBrand;
    return matchesSearch && matchesCat && matchesBrand;
  });

  const handleExportExcel = () => {
    let csvContent = "data:text/csv;charset=utf-8,Product Name,Category,Brand,SKU,Barcode,Buying Price,Selling Price,Margin (%),Stock,Status\n";
    products.forEach(p => {
      csvContent += `"${p.name}","${p.category}","${p.brand}","${p.sku}","${p.barcode}",${p.buyingPrice},${p.sellingPrice},${p.margin},${p.stock},"${p.status}"\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "products_catalog_export.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleImportProducts = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".csv,.xlsx,.xls";
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        setProducts((prev) => [
          ...prev,
          {
            name: "Cadbury Celebrations Gift Pack",
            category: "Snacks & Biscuits",
            brand: "Cadbury",
            sku: "CAD-CELEB-GP",
            barcode: "8901234567890",
            buyingPrice: 150.0,
            sellingPrice: 200.0,
            margin: 25.0,
            stock: 80,
            status: "Healthy",
            updated: "Just Now",
            img: "🎁",
          },
          {
            name: "Haldiram's Bhujia 400g",
            category: "Snacks & Biscuits",
            brand: "Haldiram",
            sku: "HAL-BHUJIA-400",
            barcode: "8901030112233",
            buyingPrice: 85.0,
            sellingPrice: 110.0,
            margin: 22.7,
            stock: 150,
            status: "Healthy",
            updated: "Just Now",
            img: "🍿",
          }
        ]);
        alert(`Successfully imported 2 products from file: ${file.name}`);
      }
    };
    input.click();
  };

  const handleAddProduct = async () => {
    const name = prompt("Enter Product Name:", "Annapoorna Ghee 1L");
    if (!name) return;
    const category = prompt("Enter Category (Staples & Grains / Dairy & Bakery / Beverages / Snacks & Biscuits):", "Dairy & Bakery");
    const brand = prompt("Enter Brand Name:", "Annapoorna");
    const buyingPrice = parseFloat(prompt("Enter Buying Price (₹):", "450")) || 0;
    const sellingPrice = parseFloat(prompt("Enter Selling Price (₹):", "550")) || 0;
    const stock = parseInt(prompt("Enter Initial Stock Qty:", "50")) || 0;
    
    const margin = buyingPrice > 0 ? Math.round(((sellingPrice - buyingPrice) / sellingPrice) * 1000) / 10 : 0;
    const sku = `${brand.slice(0, 3).toUpperCase()}-${name.split(" ")[0].toUpperCase()}-${Date.now().toString().slice(-4)}`;
    
    const newProduct = {
      name,
      category,
      brand,
      sku,
      barcode: sku,
      buyingPrice,
      sellingPrice,
      margin,
      stock,
      status: stock === 0 ? "Out of Stock" : stock < 20 ? "Low Stock" : "Healthy",
      updated: "Just Now",
      img: "📦"
    };

    // Save product to database if dynamic store
    if (activeStore) {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || "";
      if (apiBase) {
        try {
          const response = await fetch(`${apiBase}/api/stores/${activeStore.id}/products`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              sku: sku,
              name: name,
              category: category,
              costPrice: buyingPrice,
              sellingPrice: sellingPrice
            })
          });
          if (!response.ok) {
            console.error("Failed to add product to database.");
          } else {
            console.log("Product successfully saved to database!");
          }
        } catch (err) {
          console.error("Connection failure adding product to DB:", err);
        }
      }
    }
    
    setProducts((prev) => [newProduct, ...prev]);
  };

  return (
    <div className="space-y-6 font-sans px-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-sky-200/60 pb-5">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight font-serif">
            Products Catalog
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Manage stock items, pricing, margins, and catalog categories.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs font-sans">
          <button
            onClick={handleExportExcel}
            className="flex items-center gap-1.5 px-3 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold rounded-lg shadow-sm"
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-slate-400" /> Export to Excel
          </button>
          <button
            onClick={handleImportProducts}
            className="flex items-center gap-1.5 px-3 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold rounded-lg shadow-sm"
          >
            Import Products <ChevronDown className="w-3.5 h-3.5 text-slate-405" />
          </button>
          <button
            onClick={handleAddProduct}
            className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg shadow-md"
          >
            <Plus className="w-3.5 h-3.5" /> Add New Product <ChevronDown className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Top Row - KPI Stats Cards (7 Columns) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-3">
        {/* Total Products */}
        <div className="bg-white border border-sky-100 rounded-xl p-3.5 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Total Products</span>
            <Package className="w-3.5 h-3.5 text-blue-505" />
          </div>
          <span className="text-base font-black text-slate-900 mt-1">{totalProducts}</span>
          {isDemo ? (
            <span className="text-[8px] text-emerald-600 font-bold">↑ 12.5% vs last month</span>
          ) : (
            <span className="text-[8px] text-slate-400 font-medium">Active Items</span>
          )}
        </div>

        {/* Categories */}
        <div className="bg-white border border-sky-100 rounded-xl p-3.5 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Categories</span>
            <Layers className="w-3.5 h-3.5 text-blue-505" />
          </div>
          <span className="text-base font-black text-slate-900 mt-1">{categoriesCount}</span>
          {isDemo ? (
            <span className="text-[8px] text-emerald-600 font-bold">↑ 5.6% vs last month</span>
          ) : (
            <span className="text-[8px] text-slate-400 font-medium">Product Groups</span>
          )}
        </div>

        {/* Inventory Value */}
        <div className="bg-white border border-sky-100 rounded-xl p-3.5 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Inventory Value</span>
            <TrendingUp className="w-3.5 h-3.5 text-emerald-505" />
          </div>
          <span className="text-base font-black text-slate-900 mt-1">
            {totalInventoryValue > 10000000 
              ? '₹' + (totalInventoryValue / 10000000).toFixed(2) + ' Cr' 
              : '₹' + totalInventoryValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
          {isDemo ? (
            <span className="text-[8px] text-emerald-600 font-bold">↑ 18.3% vs last month</span>
          ) : (
            <span className="text-[8px] text-slate-400 font-medium">Total Cost Asset</span>
          )}
        </div>

        {/* Average Margin */}
        <div className="bg-white border border-sky-100 rounded-xl p-3.5 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Average Margin</span>
            <TrendingUp className="w-3.5 h-3.5 text-emerald-505" />
          </div>
          <span className="text-base font-black text-slate-900 mt-1">{avgMargin.toFixed(1)}%</span>
          {isDemo ? (
            <span className="text-[8px] text-emerald-600 font-bold">↑ 2.4% vs last month</span>
          ) : (
            <span className="text-[8px] text-slate-400 font-medium">Markup Yield</span>
          )}
        </div>

        {/* Low Stock */}
        <div className="bg-white border border-sky-100 rounded-xl p-3.5 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Low Stock</span>
            <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
          </div>
          <span className="text-base font-black text-slate-900 mt-1">{lowStockCount}</span>
          {isDemo ? (
            <span className="text-[8px] text-rose-600 font-bold">↓ 8.7% vs last month</span>
          ) : (
            <span className="text-[8px] text-slate-400 font-medium">Replenish Urgently</span>
          )}
        </div>

        {/* Out of Stock */}
        <div className="bg-white border border-sky-100 rounded-xl p-3.5 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Out of Stock</span>
            <XCircle className="w-3.5 h-3.5 text-rose-500" />
          </div>
          <span className="text-base font-black text-slate-900 mt-1">{outOfStockCount}</span>
          {isDemo ? (
            <span className="text-[8px] text-rose-600 font-bold">↑ 15.2% vs last month</span>
          ) : (
            <span className="text-[8px] text-slate-400 font-medium">Deficit Count</span>
          )}
        </div>

        {/* Discontinued */}
        <div className="bg-white border border-sky-100 rounded-xl p-3.5 flex flex-col gap-1 shadow-sm font-sans">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Discontinued</span>
            <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
          </div>
          <span className="text-base font-black text-slate-900 mt-1">{discontinuedCount}</span>
          {isDemo ? (
            <span className="text-[8px] text-rose-600 font-bold">↓ 25.0% vs last month</span>
          ) : (
            <span className="text-[8px] text-slate-400 font-medium">Inactive SKUs</span>
          )}
        </div>
      </div>

      {/* Row 3 - Mid Charts & Updates (Shown only on mock demo stores) */}
      {isDemo && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Category Share Donut */}
          <Card className="space-y-4">
            <div>
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-serif">
                Products by Category
              </h3>
            </div>
            <div className="flex items-center justify-between gap-3 text-[10px] text-slate-600 font-sans">
              <svg width="70" height="70" viewBox="0 0 42 42" className="transform -rotate-90 shrink-0">
                <circle cx="21" cy="21" r="15.915" fill="transparent" stroke="#3b82f6" strokeWidth="6.5" strokeDasharray="38 62" strokeDashoffset="0" />
                <circle cx="21" cy="21" r="15.915" fill="transparent" stroke="#10b981" strokeWidth="6.5" strokeDasharray="20 80" strokeDashoffset="-38" />
                <circle cx="21" cy="21" r="15.915" fill="transparent" stroke="#f59e0b" strokeWidth="6.5" strokeDasharray="18 82" strokeDashoffset="-58" />
                <circle cx="21" cy="21" r="15.915" fill="transparent" stroke="#a78bfa" strokeWidth="6.5" strokeDasharray="14 86" strokeDashoffset="-76" />
                <circle cx="21" cy="21" r="15.915" fill="transparent" stroke="#e2e8f0" strokeWidth="6.5" strokeDasharray="10 90" strokeDashoffset="-90" />
                <circle cx="21" cy="21" r="13" fill="#ffffff" />
              </svg>
              <div className="space-y-1 flex-1 font-semibold text-[10px]">
                <div className="flex justify-between">
                  <span>Staples & Grains</span>
                  <span className="font-bold text-slate-800">38%</span>
                </div>
                <div className="flex justify-between">
                  <span>Dairy & Bakery</span>
                  <span className="font-bold text-slate-800">20%</span>
                </div>
                <div className="flex justify-between">
                  <span>Snacks & Drinks</span>
                  <span className="font-bold text-slate-800">18%</span>
                </div>
                <div className="flex justify-between">
                  <span>Household</span>
                  <span className="font-bold text-slate-800">14%</span>
                </div>
              </div>
            </div>
          </Card>

          {/* Products Added Trend */}
          <Card className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-serif">
                Products Added Trend
              </h3>
              <span className="text-[9px] font-bold text-slate-400">Monthly</span>
            </div>
            <div className="h-28 flex items-end justify-between gap-1 text-[8px] text-slate-400 font-sans">
              <div className="flex flex-col items-center flex-1">
                <div className="w-3 bg-blue-500 rounded-t h-[40%]" />
                <span className="mt-1">Dec</span>
              </div>
              <div className="flex flex-col items-center flex-1">
                <div className="w-3 bg-blue-500 rounded-t h-[55%]" />
                <span className="mt-1">Jan</span>
              </div>
              <div className="flex flex-col items-center flex-1">
                <div className="w-3 bg-blue-500 rounded-t h-[65%]" />
                <span className="mt-1">Feb</span>
              </div>
              <div className="flex flex-col items-center flex-1">
                <div className="w-3 bg-blue-500 rounded-t h-[50%]" />
                <span className="mt-1">Mar</span>
              </div>
              <div className="flex flex-col items-center flex-1">
                <div className="w-3 bg-blue-500 rounded-t h-[80%]" />
                <span className="mt-1">Apr</span>
              </div>
              <div className="flex flex-col items-center flex-1">
                <div className="w-3 bg-blue-500 rounded-t h-[60%]" />
                <span className="mt-1">May</span>
              </div>
            </div>
          </Card>

          {/* Margin Distribution */}
          <Card className="space-y-4">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-serif">
              Margin Distribution
            </h3>
            <div className="space-y-2 text-[10px] text-slate-600 font-sans">
              <div className="space-y-0.5">
                <div className="flex justify-between">
                  <span>30% and above</span>
                  <span className="font-bold text-slate-900">25% (1,062)</span>
                </div>
                <div className="w-full bg-slate-100 h-1.5 rounded overflow-hidden">
                  <div className="bg-blue-600 h-full rounded" style={{ width: "25%" }} />
                </div>
              </div>
              <div className="space-y-0.5">
                <div className="flex justify-between">
                  <span>20% - 30%</span>
                  <span className="font-bold text-slate-900">32% (1,360)</span>
                </div>
                <div className="w-full bg-slate-100 h-1.5 rounded overflow-hidden">
                  <div className="bg-blue-600 h-full rounded" style={{ width: "32%" }} />
                </div>
              </div>
              <div className="space-y-0.5">
                <div className="flex justify-between">
                  <span>10% - 20%</span>
                  <span className="font-bold text-slate-900">25% (1,062)</span>
                </div>
                <div className="w-full bg-slate-100 h-1.5 rounded overflow-hidden">
                  <div className="bg-blue-600 h-full rounded" style={{ width: "25%" }} />
                </div>
              </div>
            </div>
          </Card>

          {/* Recently Updated */}
          <Card className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-serif">
                Recently Updated
              </h3>
              <span className="text-[9px] font-bold text-blue-600 cursor-pointer">View All</span>
            </div>
            <div className="space-y-2.5 font-sans text-xs">
              {updates.map((up, idx) => (
                <div key={idx} className="flex justify-between items-center text-slate-700">
                  <div>
                    <span className="font-bold text-slate-900 block font-serif leading-tight">{up.name}</span>
                    <span className="text-[9px] text-slate-450 uppercase">{up.desc}</span>
                  </div>
                  <span className="text-[9px] text-slate-400">{up.time}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* Row 4 - Filtering & Search Bar */}
      <Card className="p-4 flex flex-col md:flex-row flex-wrap items-center gap-3 text-xs bg-white border border-sky-100">
        <div className="relative flex-1 min-w-[200px]">
          <input
            type="text"
            placeholder="Search by product name, SKU, barcode..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 border border-slate-200 rounded-lg text-slate-800 focus:outline-none focus:border-sky-500 font-sans"
          />
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
        </div>

        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-700 focus:outline-none"
        >
          <option>All Categories</option>
          <option>Dairy & Bakery</option>
          <option>Staples & Grains</option>
          <option>Beverages</option>
        </select>

        <select
          value={selectedBrand}
          onChange={(e) => setSelectedBrand(e.target.value)}
          className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-700 focus:outline-none"
        >
          <option>All Brands</option>
          <option>Amul</option>
          <option>India Gate</option>
          <option>Fortune</option>
          <option>Tata Tea</option>
        </select>

        <button className="flex items-center gap-1.5 px-3 py-2 border border-slate-200 hover:bg-slate-50 rounded-lg text-slate-705 font-bold shadow-sm">
          <Filter className="w-3.5 h-3.5" /> More Filters
        </button>
        <button
          onClick={() => {
            setSearchTerm("");
            setSelectedCategory("All Categories");
            setSelectedBrand("All Brands");
          }}
          className="text-slate-400 hover:text-slate-600 font-bold text-xs"
        >
          Reset
        </button>
      </Card>

      {/* Row 5 - Products Table */}
      <Card className="overflow-x-auto border border-sky-100 rounded-xl bg-white p-0">
        <table className="w-full border-collapse text-left text-xs">
          <thead>
            <tr className="bg-slate-50 border-b border-sky-100 text-slate-500 font-bold text-[9px] uppercase tracking-wider">
              <th className="p-3.5 w-10 text-center">
                <input type="checkbox" className="w-4 h-4 rounded text-blue-600 accent-blue-600" />
              </th>
              <th className="p-3.5 min-w-[200px]">Product</th>
              <th className="p-3.5">Category</th>
              <th className="p-3.5">Brand</th>
              <th className="p-3.5">SKU / Barcode</th>
              <th className="p-3.5 text-right">Buying Price</th>
              <th className="p-3.5 text-right">Selling Price</th>
              <th className="p-3.5">Gross Margin</th>
              <th className="p-3.5">Inventory</th>
              <th className="p-3.5">Stock Status</th>
              <th className="p-3.5">Last Updated</th>
              <th className="p-3.5 text-center">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-sans text-slate-700">
            {filtered.map((item, idx) => (
              <tr key={idx} className="hover:bg-sky-50/10 transition-colors">
                <td className="p-3.5 text-center">
                  <input type="checkbox" className="w-4 h-4 rounded text-blue-600 accent-blue-600" />
                </td>
                <td className="p-3.5">
                  <div className="flex items-center gap-3">
                    <span className="text-xl bg-slate-105/50 w-8 h-8 rounded-lg flex items-center justify-center border border-slate-100">
                      {item.img}
                    </span>
                    <div>
                      <span className="font-bold text-slate-900 block font-serif leading-tight">{item.name}</span>
                      <span className="text-[9px] text-slate-400 block mt-0.5">{item.category}</span>
                    </div>
                  </div>
                </td>
                <td className="p-3.5">
                  <span className="px-2 py-0.5 rounded-full bg-sky-50 border border-sky-100 text-sky-700 text-[10px] font-semibold">
                    {item.category}
                  </span>
                </td>
                <td className="p-3.5 font-semibold text-slate-800">{item.brand}</td>
                <td className="p-3.5">
                  <span className="font-bold block text-slate-800 text-[11px]">{item.sku}</span>
                  <span className="text-[9px] text-slate-400 block mt-0.5">{item.barcode}</span>
                </td>
                <td className="p-3.5 text-right font-semibold">₹{item.buyingPrice.toFixed(2)}</td>
                <td className="p-3.5 text-right font-semibold text-slate-900">₹{item.sellingPrice.toFixed(2)}</td>
                <td className="p-3.5">
                  <div className="space-y-1">
                    <span className="font-bold block text-[10px] text-slate-800">{item.margin}%</span>
                    <div className="w-20 bg-slate-100 h-1 rounded overflow-hidden">
                      <div className="bg-emerald-500 h-full" style={{ width: `${item.margin * 3}%` }} />
                    </div>
                  </div>
                </td>
                <td className="p-3.5">
                  <div className="space-y-1">
                    <span className="font-bold block text-slate-800">{item.stock} Units</span>
                    <div className="w-20 bg-slate-100 h-1 rounded overflow-hidden">
                      <div className="bg-blue-600 h-full" style={{ width: `${Math.min(item.stock / 3, 100)}%` }} />
                    </div>
                  </div>
                </td>
                <td className="p-3.5">
                  <span
                    className={`px-2.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider ${
                      item.status === "Healthy"
                        ? "bg-emerald-50 text-emerald-700 border border-emerald-100"
                        : item.status === "Low Stock"
                        ? "bg-amber-50 text-amber-700 border border-amber-100"
                        : "bg-rose-50 text-rose-700 border border-rose-100"
                    }`}
                  >
                    {item.status}
                  </span>
                </td>
                <td className="p-3.5 text-[10px] text-slate-450 leading-relaxed">{item.updated}</td>
                <td className="p-3.5 text-center">
                  <div className="flex items-center justify-center gap-1.5 text-slate-400">
                    <button className="p-1 hover:text-blue-600 hover:bg-blue-50 rounded">
                      <Eye className="w-3.5 h-3.5" />
                    </button>
                    <button className="p-1 hover:text-blue-600 hover:bg-blue-50 rounded">
                      <Edit2 className="w-3.5 h-3.5" />
                    </button>
                    <button className="p-1 hover:text-blue-600 hover:bg-blue-50 rounded">
                      <Copy className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Pagination Footer */}
      <div className="flex justify-between items-center text-xs text-slate-450 font-sans border-t border-sky-100 pt-4">
        <span>Showing 1 to 5 of 4,250 products</span>
        <div className="flex items-center gap-1">
          <button className="px-2 py-1 border border-slate-200 hover:bg-slate-50 rounded font-semibold">&lt;</button>
          <button className="px-2 py-1 bg-blue-600 text-white font-bold rounded">1</button>
          <button className="px-2 py-1 border border-slate-200 hover:bg-slate-50 rounded">2</button>
          <button className="px-2 py-1 border border-slate-200 hover:bg-slate-50 rounded">3</button>
          <span className="px-1 text-slate-300">...</span>
          <button className="px-2 py-1 border border-slate-200 hover:bg-slate-50 rounded">425</button>
          <button className="px-2 py-1 border border-slate-200 hover:bg-slate-50 rounded font-semibold">&gt;</button>
        </div>
      </div>
    </div>
  );
}
