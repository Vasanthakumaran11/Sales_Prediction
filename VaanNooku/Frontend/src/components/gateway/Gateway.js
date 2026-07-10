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
} from "lucide-react";
import { useStoreContext } from "@/context/StoreContext";
import { STORE_PROFILES, CAPACITY_LIMITS, LOCATION_MULTIPLIERS, getColdStartFactor } from "@/lib/mock/stores";
import { FESTIVALS } from "@/lib/mock/catalog";
import { registerStore } from "@/lib/api/stores";

export function Gateway() {
  const { enterStore, enterExecutiveMode } = useStoreContext();

  // gatewayState: 'landing', 'register', 'login', 'chain', 'insights'
  const [gatewayState, setGatewayState] = useState("landing");

  // Registration Form State
  const [formData, setFormData] = useState({
    storeName: "",
    storeType: "Supermarket",
    locationType: "Urban",
    openingMonth: "October",
    investment: "850000",
  });

  const [selectedProfileId, setSelectedProfileId] = useState(STORE_PROFILES[0].id);

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

  // Submit Credentials Login
  const handleLoginSubmit = (e) => {
    e.preventDefault();
    if (!loginCredentials.username.trim()) return;

    const searchName = loginCredentials.username.toLowerCase();
    const matched = STORE_PROFILES.find(
      (p) =>
        p.name.toLowerCase().includes(searchName) ||
        p.id.toLowerCase().includes(searchName)
    ) || STORE_PROFILES[0];

    enterStore(matched);
  };

  // Submit New Store
  const handleRegisterSubmit = (e) => {
    e.preventDefault();
    if (!formData.storeName.trim()) return;
    setGatewayState("insights");
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
    const newStore = await registerStore(formData);
    enterStore(newStore);
  };

  return (
    <div className="min-h-screen bg-sky-50 text-slate-800 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 font-sans transition-colors duration-200 relative overflow-hidden">
      {/* Background Radial Glow */}
      <div className="absolute inset-0 bg-gradient-to-tr from-sky-100 via-sky-50 to-white pointer-events-none opacity-85" />

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
                <p className="text-xs text-slate-550 leading-relaxed">
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
                <p className="text-xs text-slate-550 leading-relaxed">
                  Initialize a new retail format. Analyze opening month seasonality, budget allocations, and cold-start projections.
                </p>
              </div>
              <div className="flex items-center gap-1.5 text-xs font-bold text-sky-600 pt-2 mt-auto">
                Start Onboarding <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>
          </div>

          {/* Executive Link Footer */}
          <div className="flex justify-center border-t border-slate-100 pt-6">
            <button
              id="landing-btn-chain"
              onClick={() => setGatewayState("chain")}
              className="flex items-center gap-2 text-xs font-semibold text-slate-500 hover:text-sky-600 transition-colors"
            >
              <Layers className="w-4 h-4 text-sky-500" />
              <span>Multi-Store Executive Control Mode</span>
            </button>
          </div>
        </div>
      )}

      {/* Onboarding Register State */}
      {gatewayState === "register" && (
        <form
          onSubmit={handleRegisterSubmit}
          className="w-full max-w-5xl bg-white border border-sky-100 rounded-2xl shadow-xl overflow-hidden relative z-10 flex flex-col min-h-[520px]"
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
                <span>Store Registration</span>
              </div>
              <div className="w-16 h-0.5 bg-slate-200 rounded-full" />

              <div className="flex items-center gap-2 text-slate-400">
                <span className="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-[10px] font-bold">
                  3
                </span>
                <span>First Context</span>
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

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block font-sans">Store Type</label>
                    <select
                      id="select-store-type"
                      name="storeType"
                      value={formData.storeType}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 focus:outline-none focus:border-sky-500 appearance-none bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2523475569%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E')] bg-[length:10px_10px] bg-[right_1rem_center] bg-no-repeat font-sans"
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
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-xs text-slate-800 focus:outline-none focus:border-sky-500 appearance-none bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2523475569%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E')] bg-[length:10px_10px] bg-[right_1rem_center] bg-no-repeat font-sans"
                    >
                      <option value="Urban">Urban</option>
                      <option value="Semi-Urban">Semi-Urban</option>
                      <option value="Rural">Rural</option>
                    </select>
                  </div>
                </div>

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
              </div>
            </div>

            {/* Right 2 Columns: Live Preview Card */}
            <div className="lg:col-span-2 h-full flex flex-col justify-center">
              <div className="w-full bg-sky-50/50 border border-sky-100 rounded-2xl p-6 shadow-sm flex flex-col justify-between gap-5 min-h-[280px]">
                <div className="space-y-1.5">
                  <h3 className="text-sm font-bold text-slate-900 tracking-wide font-serif">Capital Allocation Blueprint</h3>
                  <p className="text-[10px] text-slate-500 font-sans leading-normal">
                    AI-driven recommendation blueprint based on historical data.
                  </p>
                </div>

                <div className="flex items-center justify-center gap-5 my-2">
                  <div className="text-right font-sans">
                    <span className="block text-base font-black text-slate-950">30%</span>
                    <span className="block text-[8px] text-slate-500 uppercase font-bold tracking-wider">Perishables</span>
                  </div>

                  <svg width="90" height="90" viewBox="0 0 40 40" className="transform -rotate-90 shrink-0">
                    <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#e2e8f0" strokeWidth="4" />
                    <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#0ea5e9" strokeWidth="4.2" strokeDasharray="40 60" strokeDashoffset="0" />
                    <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#64748b" strokeWidth="4.2" strokeDasharray="30 70" strokeDashoffset="-40" />
                    <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#94a3b8" strokeWidth="4" strokeDasharray="30 70" strokeDashoffset="-70" />
                    <circle cx="20" cy="20" r="13" fill="#ffffff" />
                  </svg>

                  <div className="text-left font-sans">
                    <span className="block text-base font-black text-slate-950">40%</span>
                    <span className="block text-[8px] text-slate-500 uppercase font-bold tracking-wider">Staples</span>
                  </div>
                </div>

                <div className="border-t border-sky-100 pt-3 flex justify-between items-center text-[9px] text-slate-500 font-sans">
                  <div>
                    <span className="block font-semibold">Staples (40%):</span>
                    <span className="font-bold text-sky-600">₹{alloc.staples.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="block font-semibold">Perishables (30%):</span>
                    <span className="font-bold text-slate-650">₹{alloc.perishables.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Footer Controls */}
          <div className="p-5 bg-slate-50 border-t border-slate-200 flex items-center justify-end relative h-16 shrink-0">
            <button
              id="btn-initialize-engine"
              type="submit"
              className="bg-gradient-to-r from-sky-600 to-blue-600 hover:from-sky-500 hover:to-blue-500 text-white font-bold text-xs px-6 py-2.5 rounded-lg transition-all shadow-sm"
            >
              Initialize Store Engine
            </button>
          </div>
        </form>
      )}

      {/* Existing Store Login State */}
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
          <button
            type="submit"
            className="w-full py-3.5 bg-gradient-to-r from-sky-600 to-blue-600 hover:from-sky-500 hover:to-blue-500 text-white font-bold text-sm rounded-xl transition-all flex items-center justify-center gap-2 shadow-md hover:shadow-lg font-sans"
          >
            Login <ArrowRight className="w-5 h-5" />
          </button>
        </form>
      )}

      {/* Executive Control State */}
      {gatewayState === "chain" && (
        <div className="w-full max-w-4xl bg-white border border-sky-100 rounded-2xl shadow-xl p-8 relative z-10 space-y-6">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <button
              onClick={() => setGatewayState("landing")}
              className="p-1 rounded bg-slate-50 border border-slate-200 hover:bg-slate-105 text-slate-500"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <div>
              <h3 className="text-base font-bold text-slate-900 font-serif">Multi-Store Executive Control Tower</h3>
              <p className="text-[10px] text-slate-550 font-sans">
                Aggregated operations and performance benchmarking across all registered nodes.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 font-sans">
            <div className="bg-sky-50/50 border border-sky-100 p-4 rounded-xl">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">Total Chain Revenue</span>
              <span className="text-lg font-bold text-slate-900">₹455,900</span>
              <span className="text-[10px] text-sky-600 block mt-1">+14.2% projected MoM</span>
            </div>
            <div className="bg-sky-50/50 border border-sky-100 p-4 rounded-xl">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">Active Asset Value</span>
              <span className="text-lg font-bold text-teal-600">₹305,400</span>
              <span className="text-[10px] text-slate-500 block mt-1">Spread across 3 active store formats</span>
            </div>
            <div className="bg-sky-50/50 border border-sky-100 p-4 rounded-xl">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">Global Stockout Alerts</span>
              <span className="text-lg font-bold text-rose-600">17 Items</span>
              <span className="text-[10px] text-rose-500 block mt-1">Requires immediate bulk purchasing</span>
            </div>
          </div>

          {/* Table */}
          <div className="overflow-x-auto border border-sky-100 rounded-xl bg-white">
            <table className="w-full border-collapse text-left text-xs font-sans">
              <thead>
                <tr className="bg-slate-50 border-b border-sky-100 text-slate-500 font-bold text-[9px] uppercase tracking-wider">
                  <th className="p-3">Store Node</th>
                  <th className="p-3">Format Type / Location</th>
                  <th className="p-3 text-right">R² Accuracy Score</th>
                  <th className="p-3 text-right">Waste / Expiry Margin</th>
                  <th className="p-3 text-right">Active Inventory Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {STORE_PROFILES.map((store) => (
                  <tr key={store.id} className="hover:bg-sky-50/30 text-slate-700">
                    <td className="p-3 font-semibold text-slate-900">{store.name}</td>
                    <td className="p-3">
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 text-[9px] text-slate-550 border border-slate-200">
                        {store.type} - {store.location}
                      </span>
                    </td>
                    <td className="p-3 text-right font-medium text-sky-600">
                      {parseFloat(store.metrics.forecastR2) * 100}%
                    </td>
                    <td className="p-3 text-right font-medium text-amber-600">
                      {parseFloat(store.metrics.wasteMargin) * 100}%
                    </td>
                    <td className="p-3 text-right text-slate-800">₹{store.metrics.inventoryValue.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex justify-end">
            <button
              id="btn-launch-chain"
              onClick={enterExecutiveMode}
              className="flex items-center gap-2 bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs px-6 py-2.5 rounded-lg transition-all shadow"
            >
              Access Executive Control Tower <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Pre-Launch Insights Deck State */}
      {gatewayState === "insights" && (
        <div className="w-full max-w-5xl bg-white border border-sky-100 rounded-2xl shadow-xl p-8 relative z-10 space-y-8 flex flex-col">
          {/* Deck Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-100 pb-6">
            <div>
              <div className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-sky-50 border border-sky-200 text-[10px] text-sky-600 font-bold uppercase tracking-wider mb-2 font-sans">
                Predictive Analytics Report
              </div>
              <h2 className="text-2xl font-bold text-slate-900 flex items-center gap-2 font-serif">
                <Building2 className="w-5 h-5 text-sky-600" /> Pre-Launch Insights Deck: {formData.storeName}
              </h2>
              <p className="text-slate-500 text-xs mt-1 font-sans">
                Forecasting model outputs matching a{" "}
                <span className="font-semibold text-slate-800">{formData.storeType}</span> store launch in{" "}
                <span className="font-semibold text-slate-800">{formData.openingMonth}</span> at{" "}
                <span className="font-semibold text-slate-800">{formData.locationType}</span> format.
              </p>
            </div>
            <div className="flex items-center gap-3 font-sans">
              <button
                type="button"
                onClick={() => setGatewayState("register")}
                className="px-4 py-2 rounded-lg border border-slate-200 text-slate-500 hover:text-slate-800 hover:bg-slate-50 text-xs font-bold font-sans"
              >
                Modify Inputs
              </button>
              <button
                id="btn-activate-console"
                type="button"
                onClick={handleActivateStore}
                className="bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs px-5 py-2.5 rounded-lg shadow flex items-center gap-1.5"
              >
                Activate Store Console <CheckCircle2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Insights Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 font-sans">
            {/* Box 1: Month recommendations */}
            <div className="bg-slate-50/50 border border-slate-200 p-5 rounded-xl flex flex-col gap-4">
              <div>
                <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 font-serif">
                  <Sparkles className="w-4 h-4 text-sky-600" /> Seasonal Shelf Strategy
                </h4>
                <p className="text-[10px] text-slate-500 mt-1">
                  Recommended launch list aligned with {formData.openingMonth} demand patterns.
                </p>
              </div>

              <div className="space-y-3 flex-1">
                {getProductRecommendations(formData.openingMonth).map((rec, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg bg-white border border-slate-200 hover:border-sky-300 transition-all flex justify-between items-center text-xs"
                  >
                    <div>
                      <span className="font-bold text-slate-900 block">{rec.name}</span>
                      <span className="text-[10px] text-slate-550">{rec.category}</span>
                    </div>
                    <span className="inline-block px-2 py-0.5 rounded bg-sky-50 text-sky-600 text-[9px] font-bold border border-sky-100 text-right max-w-[120px] truncate">
                      {rec.reason}
                    </span>
                  </div>
                ))}
              </div>
              <div className="p-3 rounded-lg bg-sky-50/80 border border-sky-100 text-[10px] text-sky-750">
                🚀 Models recommend prioritizing staples to establish a reliable launch customer buffer.
              </div>
            </div>

            {/* Box 2: Capital Allocation */}
            <div className="bg-slate-50/50 border border-slate-200 p-5 rounded-xl flex flex-col gap-4">
              <div>
                <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 font-serif">
                  <PieIcon className="w-4 h-4 text-sky-600" /> Capital Allocation Blueprint
                </h4>
                <p className="text-[10px] text-slate-505 mt-1">
                  Optimal category investment split matching local seasonality weights.
                </p>
              </div>

              {/* Simple Donut Chart */}
              <div className="flex justify-center items-center h-32 relative">
                <svg width="120" height="120" viewBox="0 0 40 40" className="transform -rotate-90">
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#0ea5e9" strokeWidth="4" strokeDasharray="35 65" strokeDashoffset="0" />
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#14b8a6" strokeWidth="4" strokeDasharray="20 80" strokeDashoffset="-35" />
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#f59e0b" strokeWidth="4" strokeDasharray="20 80" strokeDashoffset="-55" />
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#3b82f6" strokeWidth="4" strokeDasharray="15 85" strokeDashoffset="-75" />
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#f43f5e" strokeWidth="4" strokeDasharray="10 90" strokeDashoffset="-90" />
                  <circle cx="20" cy="20" r="13" fill="#ffffff" />
                </svg>
                <div className="absolute text-center">
                  <span className="block text-xs font-bold text-slate-500">Total</span>
                  <span className="text-sm font-black text-slate-950">
                    ₹{(parseInt(formData.investment) || 150000).toLocaleString()}
                  </span>
                </div>
              </div>

              <div className="space-y-2 flex-1 text-xs">
                {getCapitalAllocation(formData.investment).map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`w-2.5 h-2.5 rounded-full ${item.color}`} />
                      <span className="text-slate-700 text-[11px] font-semibold">{item.name}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-bold text-slate-900 block">₹{item.value.toLocaleString()}</span>
                      <span className="text-[9px] text-slate-500 font-bold uppercase">{item.pct}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Box 3: 3-Month Risk Matrix */}
            <div className="bg-slate-50/50 border border-slate-200 p-5 rounded-xl flex flex-col gap-4">
              <div>
                <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 font-serif">
                  <TrendingUp className="w-4 h-4 text-sky-600" /> 3-Month Risk Matrix
                </h4>
                <p className="text-[10px] text-slate-500 mt-1">
                  90-day forward demand path showing Cold Start and stabilization factors.
                </p>
              </div>

              {/* Custom Line Chart */}
              <div className="h-32 bg-white rounded-lg p-2 border border-slate-200 flex flex-col justify-between">
                <div className="w-full h-full relative">
                  <svg viewBox="0 0 100 40" className="w-full h-full">
                    <line x1="0" y1="35" x2="100" y2="35" className="stroke-slate-200" strokeWidth="0.5" />
                    <line x1="0" y1="20" x2="100" y2="20" className="stroke-slate-200" strokeWidth="0.5" strokeDasharray="1 1" />
                    <line x1="0" y1="5" x2="100" y2="5" className="stroke-slate-200" strokeWidth="0.5" strokeDasharray="1 1" />
                    <path d="M 5 28 Q 20 22 45 17 T 95 6" fill="none" stroke="#0284c7" strokeWidth="2" />
                    <circle cx="5" cy="28" r="1.5" fill="#ef4444" />
                    <circle cx="45" cy="17" r="1.5" fill="#f59e0b" />
                    <circle cx="95" cy="6" r="1.5" fill="#0284c7" />
                    <text x="5" y="32" className="fill-slate-500" fontSize="3">
                      Month 1
                    </text>
                    <text x="45" y="22" className="fill-slate-500" fontSize="3">
                      Month 2
                    </text>
                    <text x="80" y="11" className="fill-slate-500" fontSize="3">
                      Month 3
                    </text>
                  </svg>
                </div>
              </div>

              <div className="space-y-3 flex-1 text-xs">
                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-rose-100 flex items-center justify-center text-[10px] font-bold text-rose-600 mt-0.5">
                    1
                  </div>
                  <div>
                    <span className="font-bold text-slate-800 block">Month 1: Cold Start Period (40% discount)</span>
                    <span className="text-[10px] text-slate-500">
                      Scaling applied to guard against early stock buffer bloating.
                    </span>
                  </div>
                </div>

                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-amber-100 flex items-center justify-center text-[10px] font-bold text-amber-600 mt-0.5">
                    2
                  </div>
                  <div>
                    <span className="font-bold text-slate-800 block">Month 2-3: Ramping up (70% scaling)</span>
                    <span className="text-[10px] text-slate-500">
                      Market awareness and footfall begins to match local average.
                    </span>
                  </div>
                </div>

                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-sky-100 flex items-center justify-center text-[10px] font-bold text-sky-600 mt-0.5">
                    3
                  </div>
                  <div>
                    <span className="font-bold text-slate-800 block">Month 4+: Stabilized forecasting</span>
                    <span className="text-[10px] text-slate-505">
                      ML models shift from cold start modes to full demand signal tracking.
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
