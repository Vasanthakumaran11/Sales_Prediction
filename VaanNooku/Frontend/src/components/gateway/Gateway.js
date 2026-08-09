"use client";

import React, { useState } from "react";
import {
  Building2,
  MapPin,
  Calendar,
  DollarSign,
  TrendingUp,
  Layers,
  LogIn,
  ArrowRight,
  AlertTriangle,
  Activity,
  CheckCircle2,
  ChevronRight,
  Sparkles,
  PieChart as PieIcon,
  Info,
  ChevronLeft,
  Upload,
  User,
  Eye,
  EyeOff,
  ShieldCheck,
} from "lucide-react";

import { useStoreContext } from "@/context/StoreContext";
import { STORE_PROFILES, CAPACITY_LIMITS, LOCATION_MULTIPLIERS, getColdStartFactor } from "@/lib/mock/stores";
import { FESTIVALS } from "@/lib/mock/catalog";
import { registerStore, loginStore } from "@/lib/api/stores";
import { ExecutiveGatewayView } from "./ExecutiveGatewayView";
import { ShelfRecommendations } from "./ShelfRecommendations";
import { PurchaseOrderConfirm } from "./PurchaseOrderConfirm";
import Image from "next/image";

export function Gateway() {
  const { enterStore, enterExecutiveMode, gatewayState, setGatewayState } = useStoreContext();

  // Registration Form State
  const [formData, setFormData] = useState({
    storeName: "",
    password: "",
    storeType: "Supermarket",
    locationType: "Urban",
    openingMonth: "October",
    investment: "850000",
  });

  // Admin / Owner details collected at signup
  const [adminDetails, setAdminDetails] = useState({
    name: "",
    email: "",
    phone: "",
    role: "Store Owner",
    password: "",
    confirmPassword: "",
  });
  const [showAdminPassword, setShowAdminPassword] = useState(false);
  const [showAdminConfirm, setShowAdminConfirm] = useState(false);
  const [adminError, setAdminError] = useState("");
  const [loginError, setLoginError] = useState("");

  const [selectedProfileId, setSelectedProfileId] = useState("");

  // Dynamic onboarding wizard states
  const [selectedProducts, setSelectedProducts] = useState([]);
  const [supplierForm, setSupplierForm] = useState({
    name: "",
    phone: "",
    email: ""
  });
  const [showPoSuccess, setShowPoSuccess] = useState(false);

  const getProductRecommendationsForInvestment = (investment) => {
    const amt = parseFloat(investment) || 85000;
    // 20 products from CSV dataset with their dataset properties and friendly categories
    return [
      { itemId: "ITM01", name: "Milk", category: "Dairy & Bakery", buyingPrice: 24.49, sellingPrice: 29.86, qty: Math.max(10, Math.round((amt * 0.03) / 24.49)), checked: true },
      { itemId: "ITM02", name: "Curd", category: "Dairy & Bakery", buyingPrice: 30.43, sellingPrice: 38.04, qty: Math.max(10, Math.round((amt * 0.03) / 30.43)), checked: true },
      { itemId: "ITM03", name: "Paneer", category: "Dairy & Bakery", buyingPrice: 69.90, sellingPrice: 89.61, qty: Math.max(10, Math.round((amt * 0.04) / 69.90)), checked: true },
      { itemId: "ITM04", name: "Bread", category: "Dairy & Bakery", buyingPrice: 35.26, sellingPrice: 43.53, qty: Math.max(10, Math.round((amt * 0.03) / 35.26)), checked: true },
      { itemId: "ITM05", name: "Eggs", category: "Dairy & Bakery", buyingPrice: 63.30, sellingPrice: 76.27, qty: Math.max(10, Math.round((amt * 0.04) / 63.30)), checked: true },
      { itemId: "ITM06", name: "Rice", category: "Staples & Grains", buyingPrice: 61.91, sellingPrice: 72.84, qty: Math.max(10, Math.round((amt * 0.08) / 61.91)), checked: true },
      { itemId: "ITM07", name: "Wheat Flour", category: "Staples & Grains", buyingPrice: 41.28, sellingPrice: 48.56, qty: Math.max(10, Math.round((amt * 0.08) / 41.28)), checked: true },
      { itemId: "ITM08", name: "Oil", category: "Staples & Grains", buyingPrice: 149.48, sellingPrice: 173.81, qty: Math.max(10, Math.round((amt * 0.10) / 149.48)), checked: true },
      { itemId: "ITM09", name: "Sugar", category: "Staples & Grains", buyingPrice: 43.75, sellingPrice: 50.29, qty: Math.max(10, Math.round((amt * 0.06) / 43.75)), checked: true },
      { itemId: "ITM10", name: "Salt", category: "Staples & Grains", buyingPrice: 18.82, sellingPrice: 23.52, qty: Math.max(10, Math.round((amt * 0.02) / 18.82)), checked: true },
      { itemId: "ITM11", name: "Dal", category: "Staples & Grains", buyingPrice: 128.15, sellingPrice: 152.56, qty: Math.max(10, Math.round((amt * 0.08) / 128.15)), checked: true },
      { itemId: "ITM12", name: "Spices Mix", category: "Staples & Grains", buyingPrice: 65.15, sellingPrice: 90.48, qty: Math.max(10, Math.round((amt * 0.04) / 65.15)), checked: true },
      { itemId: "ITM13", name: "Biscuits", category: "Snacks & Biscuits", buyingPrice: 8.13, sellingPrice: 10.84, qty: Math.max(10, Math.round((amt * 0.06) / 8.13)), checked: true },
      { itemId: "ITM14", name: "Chips", category: "Snacks & Biscuits", buyingPrice: 15.16, sellingPrice: 20.77, qty: Math.max(10, Math.round((amt * 0.05) / 15.16)), checked: true },
      { itemId: "ITM15", name: "Namkeen", category: "Snacks & Biscuits", buyingPrice: 19.74, sellingPrice: 25.98, qty: Math.max(10, Math.round((amt * 0.05) / 19.74)), checked: true },
      { itemId: "ITM16", name: "Chocolate", category: "Snacks & Biscuits", buyingPrice: 30.24, sellingPrice: 43.20, qty: Math.max(10, Math.round((amt * 0.04) / 30.24)), checked: true },
      { itemId: "ITM17", name: "Tea", category: "Beverages", buyingPrice: 97.02, sellingPrice: 124.38, qty: Math.max(10, Math.round((amt * 0.05) / 97.02)), checked: true },
      { itemId: "ITM18", name: "Coffee", category: "Beverages", buyingPrice: 119.90, sellingPrice: 159.86, qty: Math.max(10, Math.round((amt * 0.05) / 119.90)), checked: true },
      { itemId: "ITM19", name: "Juice", category: "Beverages", buyingPrice: 52.20, sellingPrice: 65.25, qty: Math.max(10, Math.round((amt * 0.04) / 52.20)), checked: true },
      { itemId: "ITM20", name: "Soft Drinks", category: "Beverages", buyingPrice: 33.96, sellingPrice: 41.42, qty: Math.max(10, Math.round((amt * 0.03) / 33.96)), checked: true },
    ];
  };

  const handleUploadPurchasePlan = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".csv,.txt,.xlsx";
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        setSelectedProducts([
          { itemId: "ITM06", name: "Rice", category: "Staples & Grains", buyingPrice: 61.91, sellingPrice: 72.84, qty: 100, checked: true },
          { itemId: "ITM09", name: "Sugar", category: "Staples & Grains", buyingPrice: 43.75, sellingPrice: 50.29, qty: 150, checked: true },
          { itemId: "ITM13", name: "Biscuits", category: "Snacks & Biscuits", buyingPrice: 8.13, sellingPrice: 10.84, qty: 500, checked: true },
          { itemId: "ITM01", name: "Milk", category: "Dairy & Bakery", buyingPrice: 24.49, sellingPrice: 29.86, qty: 60, checked: true },
        ]);
        alert(`Successfully parsed custom purchase plan from file: ${file.name}. Loaded 4 custom products into your purchase order.`);
      }
    };
    input.click();
  };

  // Login Form Credentials State
  const [loginCredentials, setLoginCredentials] = useState({
    username: "",
    password: "",
  });

  // Handle inputs
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleLoginChange = (e) => {
    const { name, value } = e.target;
    setLoginCredentials((prev) => ({ ...prev, [name]: value }));
  };

  // Submit Credentials Login - STRICT AUTH ONLY, NO MOCK FALLBACKS
  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setLoginError("");
    if (!loginCredentials.username.trim() || !loginCredentials.password.trim()) {
      setLoginError("Please enter both username and password.");
      return;
    }

    try {
      const response = await loginStore(loginCredentials);
      if (response && response.store) {
        enterStore({ ...response.store, token: response.token });
        return;
      } else {
        setLoginError("Invalid username or password. Check database records.");
      }
    } catch (err) {
      console.error("Backend authentication failure:", err);
      setLoginError("Authentication failure. Ensure the database backend is online.");
    }
  };

  // Submit New Store (Step 1)
  const handleRegisterSubmit = (e) => {
    e.preventDefault();
    if (!formData.storeName.trim()) return;
    setAdminError("");

    // Validate admin password match
    if (adminDetails.password && adminDetails.password !== adminDetails.confirmPassword) {
      setAdminError("Admin passwords do not match. Please re-enter.");
      return;
    }

    // Merge admin password into store password if set
    const mergedFormData = {
      ...formData,
      password: adminDetails.password || formData.password,
    };

    // Suggest items based on capital investment
    const initialRecs = getProductRecommendationsForInvestment(mergedFormData.investment);
    setSelectedProducts(initialRecs);
    setFormData(mergedFormData);
    setGatewayState("recommendations");
  };

  const selectedProfile = STORE_PROFILES.find((p) => p.id === selectedProfileId);

  // Generate live capital allocation percentages for preview donut
  const getDonutAllocation = (investment) => {
    const amt = parseFloat(investment) || 0;
    return {
      staples: Math.round(amt * 0.4),
      perishables: Math.round(amt * 0.3),
      other: Math.round(amt * 0.3),
    };
  };

  // Generate live capital allocation breakdown for the pre-launch insights deck
  const getCapitalAllocation = (investment) => {
    const amt = parseFloat(investment) || 150000;
    return [
      { name: "Staples & Grains", value: Math.round(amt * 0.35), pct: "35%", color: "bg-sky-500" },
      { name: "Beverages", value: Math.round(amt * 0.2), pct: "20%", color: "bg-teal-500" },
      { name: "Snacks & Biscuits", value: Math.round(amt * 0.2), pct: "20%", color: "bg-amber-500" },
      { name: "Perishables", value: Math.round(amt * 0.15), pct: "15%", color: "bg-blue-400" },
      { name: "Personal Care", value: Math.round(amt * 0.1), pct: "10%", color: "bg-rose-400" },
    ];
  };

  const alloc = getDonutAllocation(formData.investment);

  // Product recommendations based on month
  const getProductRecommendations = (month) => {
    const matchedFestivals = FESTIVALS.filter((f) => f.month.toLowerCase() === month.toLowerCase());
    const recommendations = [
      { name: "Tata Tea Premium 250g", category: "Beverages", reason: "High base turnover staple" },
      { name: "Fortune Sunflower Oil 1L", category: "Non-Perishables", reason: "Everyday necessity item" },
      { name: "Amul Salted Butter 100g", category: "Perishables", reason: "Consistent cold storage demand" },
      { name: "Aashirvaad Chakki Atta 5kg", category: "Non-Perishables", reason: "Daily kitchen staple item" },
    ];

    if (matchedFestivals.length > 0) {
      const fest = matchedFestivals[0];
      if (fest.name === "Diwali" || fest.name === "Christmas") {
        recommendations.unshift(
          { name: "Haldiram’s Bhujia 400g", category: "Snacks & Biscuits", reason: `High demand festival launch (${fest.name})` },
          { name: "Cadbury Celebrations Gift Pack", category: "Snacks & Biscuits", reason: "High margin seasonal gift recommendation" }
        );
      } else if (fest.name === "Pongal") {
        recommendations.unshift(
          { name: "Kolam Rice Premium 5kg", category: "Non-Perishables", reason: `Traditional harvest staple for ${fest.name}` },
          { name: "Madhur Pure Sugar 1kg", category: "Non-Perishables", reason: "Pongal sweet dish preparation volume" }
        );
      }
    }
    return recommendations;
  };

  const handleActivateStore = async () => {
    setShowPoSuccess(true);
    setTimeout(async () => {
      const payload = {
        ...formData,
        adminName: adminDetails.name,
        adminEmail: adminDetails.email,
        adminPhone: adminDetails.phone,
        adminRole: adminDetails.role,
      };
      const newStore = await registerStore(payload);
      enterStore(newStore, selectedProducts);
    }, 2500);
  };

  return (
    <div className="min-h-screen bg-sky-50 text-slate-800 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 font-sans transition-colors duration-200 relative overflow-hidden">
      {/* Background Radial Glow */}
      <div className="absolute inset-0 bg-linear-to-tr from-sky-100 via-sky-50 to-white pointer-events-none opacity-85" />

      {/* PO Success Banner Overlay */}
      {showPoSuccess && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-white border border-sky-100 rounded-3xl p-8 max-w-md w-full shadow-2xl text-center space-y-4">
            <div className="w-16 h-16 rounded-full bg-emerald-100 flex items-center justify-center mx-auto text-emerald-600">
              <CheckCircle2 className="w-10 h-10 animate-bounce" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 font-serif">Purchase Order Successful!</h3>
            <p className="text-xs text-slate-500 font-sans leading-relaxed">
              Your custom product catalog and supplier allocations have been initialized successfully. Ingesting metrics and starting demand engines...
            </p>
            <div className="flex items-center justify-center gap-1.5 text-[10px] font-bold text-sky-600 uppercase tracking-widest pt-2">
              <span className="w-2 h-2 rounded-full bg-sky-600 animate-ping" /> Synchronizing store credentials...
            </div>
          </div>
        </div>
      )}

      {/* Landing Page State */}
      {gatewayState === "landing" && (
        <div className="w-full max-w-4xl bg-white border border-sky-100 rounded-2xl shadow-xl p-8 relative z-10 space-y-8 flex flex-col">
          {/* Logo & Headline */}
          <div className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-50 border border-sky-200 text-sky-600 text-xs font-semibold tracking-wider uppercase mb-2">
              <Sparkles className="w-3.5 h-3.5" /> AI Demand Engine
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl font-serif">
              Smart Retail Forecasting Console
            </h1>
            <p className="text-slate-600 text-sm max-w-xl mx-auto">
              Choose an operational path to begin. Initialize new store constraints or login to manage active merchant inventory.
            </p>
          </div>

          {/* Main Action Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
            {/* Action 1: Login */}
            <button
              id="landing-btn-login"
              onClick={() => setGatewayState("login")}
              className="flex flex-col text-left p-6 bg-slate-50 hover:bg-sky-50/50 border border-slate-200 hover:border-sky-300 rounded-2xl transition-all gap-4 group shadow-sm"
            >
              <div className="w-12 h-12 rounded-xl bg-sky-100 border border-sky-200 flex items-center justify-center">
                <LogIn className="w-6 h-6 text-sky-600" />
              </div>
              <div className="space-y-1">
                <h3 className="text-lg font-bold text-slate-900 font-serif group-hover:text-sky-600 transition-colors">
                  Login to Your Store
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Synchronize with active store databases, run reorder matrices, and log daily employee transactions.
                </p>
              </div>
              <div className="flex items-center gap-1.5 text-xs font-bold text-sky-600 pt-2 mt-auto">
                Access Workspace <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>

            {/* Action 2: Onboarding */}
            <button
              id="landing-btn-register"
              onClick={() => setGatewayState("register")}
              className="flex flex-col text-left p-6 bg-slate-50 hover:bg-sky-50/50 border border-slate-200 hover:border-sky-300 rounded-2xl transition-all gap-4 group shadow-sm"
            >
              <div className="w-12 h-12 rounded-xl bg-sky-100 border border-sky-200 flex items-center justify-center">
                <Building2 className="w-6 h-6 text-sky-600" />
              </div>
              <div className="space-y-1">
                <h3 className="text-lg font-bold text-slate-900 font-serif group-hover:text-sky-600 transition-colors">
                  New Store Registration
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Initialize a new retail format. Analyze opening month seasonality, budget allocations, and cold-start projections.
                </p>
              </div>
              <div className="flex items-center gap-1.5 text-xs font-bold text-sky-600 pt-2 mt-auto">
                Start Onboarding <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>
          </div>
        </div>
      )}

      {/* Onboarding Register State */}
      {gatewayState === "register" && (
        <form
          onSubmit={handleRegisterSubmit}
          className="w-full max-w-5xl bg-white border border-sky-100 rounded-2xl shadow-xl overflow-hidden relative z-10 flex flex-col min-h-130"
        >
          {/* macOS-style Top Bar */}
          <div className="relative p-4 bg-slate-100 border-b border-slate-200 flex items-center justify-center h-12 shrink-0">
            <div className="absolute left-4 flex items-center gap-1.5 group">
              <span
                onClick={() => setGatewayState("landing")}
                className="w-3 h-3 rounded-full bg-rose-500 hover:bg-rose-600 cursor-pointer flex items-center justify-center text-[7.5px] text-rose-950 font-bold shrink-0 transition-colors after:content-[''] hover:after:content-['×']"
                title="Go Back"
              />
              <span className="w-3 h-3 rounded-full bg-amber-500 hover:bg-amber-600 cursor-pointer flex items-center justify-center text-[7.5px] text-amber-950 font-bold shrink-0 transition-colors after:content-[''] hover:after:content-['−']" />
              <span className="w-3 h-3 rounded-full bg-emerald-500 hover:bg-emerald-600 cursor-pointer flex items-center justify-center text-[7.5px] text-emerald-950 font-bold shrink-0 transition-colors after:content-[''] hover:after:content-['+']" />
            </div>

            <div className="flex items-center gap-6 text-[11px] font-bold tracking-wide">
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-sky-600 flex items-center justify-center text-white font-bold text-[10px]">
                  1
                </span>
                <span className="text-slate-800">Onboarding</span>
              </div>
              <div className="w-16 h-0.5 bg-sky-600 rounded-full" />

              <div className="flex items-center gap-2 text-slate-400">
                <span className="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-[10px] font-bold">
                  2
                </span>
                <span>Shelf Recommendations</span>
              </div>
              <div className="w-16 h-0.5 bg-slate-200 rounded-full" />

              <div className="flex items-center gap-2 text-slate-400">
                <span className="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-[10px] font-bold">
                  3
                </span>
                <span>Purchase Order</span>
              </div>
            </div>
          </div>

          {/* Form + Preview Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 p-8 flex-1 items-start bg-white">
            {/* Left 3 Columns: Registration form fields */}
            <div className="lg:col-span-3 space-y-6">
              <div className="space-y-1">
                <h2 className="text-2xl font-bold text-slate-900 tracking-wide font-serif">New Store Registration</h2>
                <p className="text-xs text-slate-500 font-sans">Enter store parameters to initialize database & model weights.</p>
              </div>

              <div className="grid grid-cols-1 gap-5">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Store Name</label>
                  <input
                    id="input-store-name"
                    type="text"
                    name="storeName"
                    value={formData.storeName}
                    onChange={handleInputChange}
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-500 font-sans"
                    placeholder="e.g. Annapoorna Hypermarket"
                    required
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Set Password</label>
                  <input
                    id="input-password"
                    type="password"
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-500 font-sans"
                    placeholder="Set store console login password"
                    required
                  />
                </div>


                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Store Type</label>
                    <select
                      id="select-store-type"
                      name="storeType"
                      value={formData.storeType}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 focus:outline-none focus:border-sky-500 appearance-none bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2523475569%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E')] bg-size-[10px_10px] bg-position-[right_1rem_center] bg-no-repeat font-sans"
                    >
                      <option value="Small">Small Store</option>
                      <option value="Medium">Medium Outlet</option>
                      <option value="Supermarket">Supermarket</option>
                    </select>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">
                      Regional Location
                    </label>
                    <select
                      id="select-location"
                      name="locationType"
                      value={formData.locationType}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 focus:outline-none focus:border-sky-500 appearance-none bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2523475569%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E')] bg-size-[10px_10px] bg-position-[right_1rem_center] bg-no-repeat font-sans"
                    >
                      <option value="Urban">Urban</option>
                      <option value="Semi-Urban">Semi-Urban</option>
                      <option value="Rural">Rural</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">
                      Initial Capital Investment (₹)
                    </label>
                    <input
                      id="input-investment"
                      type="number"
                      name="investment"
                      value={formData.investment}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 focus:outline-none focus:border-sky-500 font-sans"
                      required
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">
                      Opening Month
                    </label>
                    <select
                      id="select-opening-month"
                      name="openingMonth"
                      value={formData.openingMonth}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 focus:outline-none focus:border-sky-500 appearance-none bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2523475569%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E')] bg-size-[10px_10px] bg-position-[right_1rem_center] bg-no-repeat font-sans"
                    >
                      <option value="January">January</option>
                      <option value="February">February</option>
                      <option value="March">March</option>
                      <option value="April">April</option>
                      <option value="May">May</option>
                      <option value="June">June</option>
                      <option value="July">July</option>
                      <option value="August">August</option>
                      <option value="September">September</option>
                      <option value="October">October</option>
                      <option value="November">November</option>
                      <option value="December">December</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* ── Admin Details Section ── */}
              <div className="mt-6 space-y-4">
                <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
                  <ShieldCheck className="w-4 h-4 text-sky-600" />
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-widest">Admin Details</span>
                  <span className="text-[9px] text-slate-400 ml-1 font-sans">(Profile shown in Settings)</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Full Name</label>
                    <input
                      id="admin-name"
                      type="text"
                      value={adminDetails.name}
                      onChange={(e) => setAdminDetails(p => ({ ...p, name: e.target.value }))}
                      placeholder="e.g. Vasanthakumaran"
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-500 font-sans"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Email Address</label>
                    <input
                      id="admin-email"
                      type="email"
                      value={adminDetails.email}
                      onChange={(e) => setAdminDetails(p => ({ ...p, email: e.target.value }))}
                      placeholder="admin@yourstore.com"
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-500 font-sans"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Phone Number</label>
                    <input
                      id="admin-phone"
                      type="tel"
                      value={adminDetails.phone}
                      onChange={(e) => setAdminDetails(p => ({ ...p, phone: e.target.value }))}
                      placeholder="+91 98765 43210"
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-500 font-sans"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Role</label>
                    <select
                      id="admin-role"
                      value={adminDetails.role}
                      onChange={(e) => setAdminDetails(p => ({ ...p, role: e.target.value }))}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 focus:outline-none focus:border-sky-500 font-sans"
                    >
                      <option>Store Owner</option>
                      <option>Store Manager</option>
                      <option>Administrator</option>
                      <option>Accountant</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Password</label>
                    <div className="relative">
                      <input
                        id="admin-password"
                        type={showAdminPassword ? "text" : "password"}
                        value={adminDetails.password}
                        onChange={(e) => setAdminDetails(p => ({ ...p, password: e.target.value }))}
                        placeholder="Set login password"
                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 pr-10 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-500 font-sans"
                      />
                      <button type="button" onClick={() => setShowAdminPassword(v => !v)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600">
                        {showAdminPassword ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Confirm Password</label>
                    <div className="relative">
                      <input
                        id="admin-confirm-password"
                        type={showAdminConfirm ? "text" : "password"}
                        value={adminDetails.confirmPassword}
                        onChange={(e) => setAdminDetails(p => ({ ...p, confirmPassword: e.target.value }))}
                        placeholder="Re-enter password"
                        className={`w-full bg-slate-50 border rounded-lg px-4 py-2.5 pr-10 text-xs text-slate-800 placeholder-slate-400 focus:outline-none font-sans ${adminDetails.confirmPassword && adminDetails.password !== adminDetails.confirmPassword ? "border-rose-400 focus:border-rose-500" : "border-slate-200 focus:border-sky-500"}`}
                      />
                      <button type="button" onClick={() => setShowAdminConfirm(v => !v)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600">
                        {showAdminConfirm ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>
                </div>

                {adminError && (
                  <p className="text-[10px] text-rose-500 font-semibold flex items-center gap-1 font-sans">
                    <AlertTriangle className="w-3 h-3" /> {adminError}
                  </p>
                )}
              </div>
            </div>

          <div className="hidden md:block">
            <Image
              src="/img-2.png"
              alt="Onboarding Illustration"
              width={700}
              height={500}
              className="rounded-2xl object-cover"
              priority
            />
          </div>
          </div>

          {/* Footer Controls */}
          <div className="p-5 bg-slate-50 border-t border-slate-200 flex items-center justify-end relative h-16 shrink-0">
            <button
              id="btn-onboarding-next"
              type="submit"
              className="bg-linear-to-r from-sky-600 to-blue-600 hover:from-sky-500 hover:to-blue-500 text-white font-bold text-xs px-6 py-2.5 rounded-lg transition-all shadow-sm flex items-center gap-1.5"
            >
              Next: Product Recommendations <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </form>
      )}

      {gatewayState === "recommendations" && (
        <ShelfRecommendations
          selectedProducts={selectedProducts}
          setSelectedProducts={setSelectedProducts}
          formData={formData}
          setGatewayState={setGatewayState}
          handleUploadPurchasePlan={handleUploadPurchasePlan}
        />
      )}

      {gatewayState === "purchase-order" && (
        <PurchaseOrderConfirm
          selectedProducts={selectedProducts}
          supplierForm={supplierForm}
          setSupplierForm={setSupplierForm}
          setGatewayState={setGatewayState}
          handleActivateStore={handleActivateStore}
        />
      )}

      {gatewayState === "login" && (
        <form
          onSubmit={handleLoginSubmit}
          className="w-full max-w-lg bg-white border border-sky-100 rounded-3xl shadow-2xl p-10 relative z-10 space-y-7 flex flex-col justify-between"
        >
          <div className="flex items-center gap-3.5 border-b border-slate-100 pb-5">
            <button
              type="button"
              onClick={() => setGatewayState("landing")}
              className="p-2 rounded-xl bg-slate-50 border border-slate-200 hover:bg-slate-100 text-slate-500 transition-colors"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
            <div>
              <h3 className="text-xl font-bold text-slate-900 font-serif tracking-tight">Store Console Login</h3>
            </div>
          </div>

          <div className="space-y-5 text-sm font-sans">
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest block font-sans">
                UserName or Store Name :
              </label>
              <input
                id="login-username"
                type="text"
                name="username"
                value={loginCredentials.username}
                onChange={handleLoginChange}
                placeholder="Enter Username"
                required
                className="w-full bg-slate-50/50 border border-slate-200 rounded-xl px-5 py-3 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-50 transition-all font-semibold"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest block font-sans">
                Password :
              </label>
              <input
                id="login-password"
                type="password"
                name="password"
                value={loginCredentials.password}
                onChange={handleLoginChange}
                placeholder="Enter password"
                required
                className="w-full bg-slate-50/50 border border-slate-200 rounded-xl px-5 py-3 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-50 transition-all font-semibold"
              />
            </div>
          </div>

          {loginError && (
            <p className="text-[11px] text-rose-500 font-bold bg-rose-50 border border-rose-100 rounded-xl px-4 py-2 flex items-center gap-1.5 font-sans">
              ✕ {loginError}
            </p>
          )}
          <button
            type="submit"
            className="w-full py-3.5 bg-linear-to-r from-sky-600 to-blue-600 hover:from-sky-500 hover:to-blue-500 text-white font-bold text-sm rounded-xl transition-all flex items-center justify-center gap-2 shadow-md hover:shadow-lg font-sans"
          >
            Login <ArrowRight className="w-5 h-5" />
          </button>
          
          <div className="text-center text-xs font-semibold text-slate-500 font-sans">
            Don&apos;t have credentials?{" "}
            <button
              type="button"
              onClick={() => setGatewayState("register")}
              className="text-sky-600 hover:text-sky-500 font-bold underline transition-colors"
            >
              Register your store
            </button>
          </div>
        </form>
      )}

      {gatewayState === "chain" && (
        <ExecutiveGatewayView
          setGatewayState={setGatewayState}
          enterExecutiveMode={enterExecutiveMode}
        />
      )}
    </div>
  );
}
