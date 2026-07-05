import React, { useState } from 'react';
import { 
  Building2, 
  MapPin, 
  Calendar, 
  DollarSign, 
  TrendingUp, 
  Layers, 
  Users, 
  ArrowRight, 
  AlertTriangle, 
  Activity, 
  CheckCircle2, 
  ChevronRight, 
  Sparkles, 
  PieChart as PieIcon, 
  Info 
} from 'lucide-react';
import { STORE_PROFILES, PRODUCT_CATALOG, FESTIVALS } from './mockData';

export default function Gateway({ onSelectStore, onSelectChain }) {
  const [activeTab, setActiveTab] = useState('new'); // 'new', 'existing', 'chain'
  
  // New Store Form State
  const [wizardStep, setWizardStep] = useState(1);
  const [formData, setFormData] = useState({
    storeName: '',
    storeType: 'Medium', // Small, Medium, Supermarket
    locationType: 'Urban', // Urban, Semi-Urban, Rural
    openingMonth: 'October',
    investment: '150000'
  });
  
  const [showInsightsDeck, setShowInsightsDeck] = useState(false);
  const [selectedProfileId, setSelectedProfileId] = useState(STORE_PROFILES[0].id);

  // Handle inputs
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  // Submit New Store Wizard
  const handleWizardSubmit = (e) => {
    e.preventDefault();
    if (!formData.storeName.trim()) return;
    setShowInsightsDeck(true);
  };

  const selectedProfile = STORE_PROFILES.find(p => p.id === selectedProfileId);

  // Generate capital allocation for donut chart
  const getCapitalAllocation = (investment) => {
    const amt = parseFloat(investment) || 100000;
    return [
      { name: 'Staples & Grain', value: amt * 0.35, pct: '35%', color: 'bg-emerald-500', fill: '#10b981' },
      { name: 'Beverages', value: amt * 0.20, pct: '20%', color: 'bg-teal-500', fill: '#14b8a6' },
      { name: 'Snacks & Sweets', value: amt * 0.20, pct: '20%', color: 'bg-amber-500', fill: '#f59e0b' },
      { name: 'Perishables', value: amt * 0.15, pct: '15%', color: 'bg-blue-500', fill: '#3b82f6' },
      { name: 'Personal Care & Hygiene', value: amt * 0.10, pct: '10%', color: 'bg-rose-500', fill: '#f43f5e' },
    ];
  };

  // Product recommendations based on month
  const getProductRecommendations = (month) => {
    const matchedFestivals = FESTIVALS.filter(f => f.month.toLowerCase() === month.toLowerCase());
    const recommendations = [
      { name: 'Tata Tea Premium 250g', category: 'Beverages', reason: 'High base turnover staple' },
      { name: 'Fortune Sunflower Oil 1L', category: 'Non-Perishables', reason: 'Everyday household necessity' },
      { name: 'Amul Salted Butter 100g', category: 'Perishables', reason: 'Consistent cold storage demand' },
      { name: 'Aashirvaad Shudh Chakki Atta 5kg', category: 'Non-Perishables', reason: 'Daily kitchen staple item' },
    ];

    if (matchedFestivals.length > 0) {
      const fest = matchedFestivals[0];
      if (fest.name === 'Diwali' || fest.name === 'Christmas') {
        recommendations.unshift(
          { name: 'Haldiram’s Bhujia 400g', category: 'Snacks & Biscuits', reason: `High demand festival launch (${fest.name})` },
          { name: 'Cadbury Celebrations Gift Pack', category: 'Snacks & Biscuits', reason: 'High margin seasonal gift recommendation' }
        );
      } else if (fest.name === 'Pongal') {
        recommendations.unshift(
          { name: 'Kolam Rice Premium 5kg', category: 'Non-Perishables', reason: `Traditional harvest staple for ${fest.name}` },
          { name: 'Madhur Pure Sugar 1kg', category: 'Non-Perishables', reason: 'Pongal sweet dish preparation volume' }
        );
      }
    }
    return recommendations;
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 font-sans">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-zinc-900 via-zinc-950 to-black pointer-events-none opacity-80" />
      
      {!showInsightsDeck ? (
        <div className="w-full max-w-4xl bg-zinc-900/40 backdrop-blur-xl border border-zinc-800/80 rounded-2xl shadow-2xl p-8 relative z-10 flex flex-col gap-8">
          {/* Header */}
          <div className="text-center space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold tracking-wider uppercase mb-2">
              <Sparkles className="w-3.5 h-3.5" /> Workspace Initializer
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Retail Forecasting Console
            </h1>
            <p className="text-zinc-400 text-sm max-w-xl mx-auto">
              Configure your predictive modeling workspace. Access individual store dashboards, run optimization matrices, or monitor multi-store retail operations.
            </p>
          </div>

          {/* Mode Selector Tabs */}
          <div className="grid grid-cols-3 gap-2 p-1 bg-zinc-950 rounded-xl border border-zinc-800">
            <button
              id="tab-new-store"
              onClick={() => setActiveTab('new')}
              className={`flex flex-col sm:flex-row items-center justify-center gap-2 py-3 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'new' 
                  ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700/50' 
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Building2 className="w-4 h-4 text-emerald-500" />
              <span>New Store Setup</span>
            </button>
            
            <button
              id="tab-existing-store"
              onClick={() => setActiveTab('existing')}
              className={`flex flex-col sm:flex-row items-center justify-center gap-2 py-3 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'existing' 
                  ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700/50' 
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Activity className="w-4 h-4 text-amber-500" />
              <span>Optimize Store</span>
            </button>

            <button
              id="tab-chain-store"
              onClick={() => setActiveTab('chain')}
              className={`flex flex-col sm:flex-row items-center justify-center gap-2 py-3 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'chain' 
                  ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700/50' 
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Layers className="w-4 h-4 text-blue-500" />
              <span>Executive Control</span>
            </button>
          </div>

          {/* Tab Content */}
          <div className="mt-2 min-h-[350px]">
            {/* Tab 1: New Store Wizard */}
            {activeTab === 'new' && (
              <form onSubmit={handleWizardSubmit} className="space-y-6">
                {wizardStep === 1 ? (
                  <div className="space-y-6">
                    <div className="border-b border-zinc-800 pb-4">
                      <h3 className="text-lg font-semibold text-white">Step 1: Store Demographics</h3>
                      <p className="text-xs text-zinc-400">Specify store dimensions and location scaling multipliers.</p>
                    </div>
                    
                    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Store Name</label>
                        <input
                          id="input-store-name"
                          type="text"
                          name="storeName"
                          value={formData.storeName}
                          onChange={handleInputChange}
                          placeholder="e.g. Balaji Groceries"
                          className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-2.5 text-white placeholder-zinc-600 focus:outline-none focus:border-emerald-500 text-sm"
                          required
                        />
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Physical Format Size</label>
                        <select
                          id="select-store-type"
                          name="storeType"
                          value={formData.storeType}
                          onChange={handleInputChange}
                          className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-emerald-500 text-sm"
                        >
                          <option value="Small">Small (Intake Limit: 400 items/day)</option>
                          <option value="Medium">Medium (Intake Limit: 800 items/day)</option>
                          <option value="Supermarket">Supermarket (No Capacity Cap)</option>
                        </select>
                      </div>

                      <div className="space-y-2 sm:col-span-2">
                        <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider block mb-1">Location Profile</label>
                        <div className="grid grid-cols-3 gap-3">
                          {[
                            { value: 'Urban', multiplier: '1.0x demand scaling', desc: 'Metropolitan center' },
                            { value: 'Semi-Urban', multiplier: '0.8x demand scaling', desc: 'Township / Suburbs' },
                            { value: 'Rural', multiplier: '0.6x demand scaling', desc: 'Rural community' }
                          ].map(item => (
                            <label
                              key={item.value}
                              className={`flex flex-col justify-between p-3 rounded-lg border text-left cursor-pointer transition-all ${
                                formData.locationType === item.value 
                                  ? 'bg-emerald-500/5 border-emerald-500 text-white' 
                                  : 'bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700'
                              }`}
                            >
                              <input
                                type="radio"
                                name="locationType"
                                value={item.value}
                                checked={formData.locationType === item.value}
                                onChange={handleInputChange}
                                className="sr-only"
                              />
                              <div>
                                <span className="block text-sm font-semibold text-white">{item.value}</span>
                                <span className="block text-[10px] text-zinc-500 mt-1">{item.desc}</span>
                              </div>
                              <span className={`inline-block text-[10px] px-2 py-0.5 rounded-full mt-3 font-semibold ${
                                formData.locationType === item.value 
                                  ? 'bg-emerald-500/20 text-emerald-400' 
                                  : 'bg-zinc-900 text-zinc-400'
                              }`}>
                                {item.multiplier}
                              </span>
                            </label>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex justify-end pt-4">
                      <button
                        id="btn-next-step"
                        type="button"
                        onClick={() => setWizardStep(2)}
                        disabled={!formData.storeName.trim()}
                        className="flex items-center gap-2 bg-zinc-800 hover:bg-zinc-700 text-white font-medium text-sm px-6 py-2.5 rounded-lg border border-zinc-700/50 transition-all disabled:opacity-40"
                      >
                        Next Step <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="border-b border-zinc-800 pb-4">
                      <h3 className="text-lg font-semibold text-white">Step 2: Capital & Launch Calendar</h3>
                      <p className="text-xs text-zinc-400">Establish opening Month seasonality and initial inventory capital.</p>
                    </div>
                    
                    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider block">Opening Month</label>
                        <div className="relative">
                          <select
                            id="select-opening-month"
                            name="openingMonth"
                            value={formData.openingMonth}
                            onChange={handleInputChange}
                            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-emerald-500 text-sm appearance-none"
                          >
                            {['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'].map(m => (
                              <option key={m} value={m}>{m}</option>
                            ))}
                          </select>
                          <Calendar className="absolute right-3.5 top-3 w-4 h-4 text-zinc-500 pointer-events-none" />
                        </div>
                        <p className="text-[10px] text-zinc-500">
                          Used to compute starting seasonal demand factors (e.g., Diwali or Pongal overlays).
                        </p>
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider block">Initial Inventory Budget (INR)</label>
                        <div className="relative">
                          <input
                            id="input-investment"
                            type="number"
                            name="investment"
                            value={formData.investment}
                            onChange={handleInputChange}
                            placeholder="e.g. 200000"
                            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-8 py-2.5 text-white focus:outline-none focus:border-emerald-500 text-sm"
                            required
                          />
                          <DollarSign className="absolute left-3 top-3 w-4 h-4 text-zinc-500" />
                        </div>
                        <p className="text-[10px] text-zinc-500">
                          Initial stock recommendation logic scales to fit this capital allocation.
                        </p>
                      </div>
                    </div>

                    <div className="flex justify-between pt-4">
                      <button
                        type="button"
                        onClick={() => setWizardStep(1)}
                        className="text-zinc-400 hover:text-zinc-200 text-sm font-medium"
                      >
                        Back to Demographics
                      </button>
                      <button
                        id="btn-run-prelaunch"
                        type="submit"
                        className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-sm px-6 py-2.5 rounded-lg transition-all"
                      >
                        Run Pre-Launch Engine <ArrowRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                )}
              </form>
            )}

            {/* Tab 2: Existing Store Sync */}
            {activeTab === 'existing' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-white">Optimize Active Store</h3>
                  <p className="text-xs text-zinc-400">Sync with an active registered merchant profile database.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider block">Select Active Store Profile</label>
                      <select
                        id="select-store-profile"
                        value={selectedProfileId}
                        onChange={(e) => setSelectedProfileId(e.target.value)}
                        className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-amber-500 text-sm"
                      >
                        {STORE_PROFILES.map(p => (
                          <option key={p.id} value={p.id}>{p.name} ({p.type} - {p.location})</option>
                        ))}
                      </select>
                    </div>

                    <div className="p-4 rounded-xl bg-zinc-950 border border-zinc-800/80 space-y-4 text-xs">
                      <div className="flex items-center gap-2 text-white font-medium">
                        <Info className="w-3.5 h-3.5 text-amber-500" />
                        <span>Profile Integration Info</span>
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-zinc-400">
                        <div>
                          <span className="block text-[10px] text-zinc-500 uppercase font-semibold">Active Period</span>
                          <span className="text-sm font-semibold text-zinc-300">{selectedProfile.activeDays} Days Logged</span>
                        </div>
                        <div>
                          <span className="block text-[10px] text-zinc-500 uppercase font-semibold">Predictive Model Accuracy</span>
                          <span className="text-sm font-semibold text-emerald-400">{parseFloat(selectedProfile.metrics.r2) * 100}% R²</span>
                        </div>
                        <div>
                          <span className="block text-[10px] text-zinc-500 uppercase font-semibold">Opening Month</span>
                          <span className="text-sm font-semibold text-zinc-300">{selectedProfile.openingMonth}</span>
                        </div>
                        <div>
                          <span className="block text-[10px] text-zinc-500 uppercase font-semibold">Initial Capital Outlay</span>
                          <span className="text-sm font-semibold text-zinc-300">₹{selectedProfile.investment.toLocaleString()}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Operational Health Summary */}
                  <div className="space-y-4">
                    <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider block">Operational Health Summary</label>
                    
                    <div className="space-y-3">
                      {/* Deficit Tracker */}
                      <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-900/30 flex items-start gap-3">
                        <AlertTriangle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
                        <div className="space-y-1">
                          <span className="text-xs font-bold text-rose-400 uppercase block tracking-wider">Deficit Tracker</span>
                          <p className="text-xs text-zinc-300">
                            Flashed <span className="font-semibold text-rose-300">{selectedProfile.metrics.deficitCount} critical supply shortages</span>. High probability of stockouts within 48h.
                          </p>
                        </div>
                      </div>

                      {/* Efficiency Metric */}
                      <div className="p-4 rounded-xl bg-amber-950/10 border border-amber-900/30 flex items-start gap-3">
                        <TrendingUp className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                        <div className="space-y-1">
                          <span className="text-xs font-bold text-amber-400 uppercase block tracking-wider">Efficiency Tracker</span>
                          <p className="text-xs text-zinc-300">
                            Historical waste margin at <span className="font-semibold text-amber-300">{selectedProfile.metrics.wasteMargin}</span>. Dead stock over-allocation leaks <span className="font-semibold text-red-400">{selectedProfile.metrics.leakingMargin}</span> of net profit margins.
                          </p>
                        </div>
                      </div>
                    </div>

                    <button
                      id="btn-sync-store"
                      onClick={() => onSelectStore(selectedProfile)}
                      className="w-full py-3 bg-amber-600 hover:bg-emerald-600 text-white font-semibold text-sm rounded-lg transition-all flex items-center justify-center gap-2 group border border-amber-500/20 hover:border-emerald-500/20"
                    >
                      Sync & Launch Operations Center <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-all" />
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Tab 3: Multi-Store Executive */}
            {activeTab === 'chain' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-white">Multi-Store Chain Executive Control Tower</h3>
                  <p className="text-xs text-zinc-400">Aggregated operations and performance benchmarking across all registered nodes.</p>
                </div>

                {/* Macro metrics */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-zinc-950 border border-zinc-800 p-4 rounded-xl">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold block">Total Chain Revenue</span>
                    <span className="text-lg sm:text-xl font-bold text-white tracking-tight">₹455,900</span>
                    <span className="text-[10px] text-emerald-400 block mt-1">+14.2% projected MoM</span>
                  </div>
                  <div className="bg-zinc-950 border border-zinc-800 p-4 rounded-xl">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold block">Active Inventory Asset Value</span>
                    <span className="text-lg sm:text-xl font-bold text-teal-400 tracking-tight">₹305,400</span>
                    <span className="text-[10px] text-zinc-500 block mt-1">Spread across 3 active format stores</span>
                  </div>
                  <div className="bg-zinc-950 border border-zinc-800 p-4 rounded-xl">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold block">Global Stockout Alerts</span>
                    <span className="text-lg sm:text-xl font-bold text-rose-500 tracking-tight">17 Items</span>
                    <span className="text-[10px] text-rose-400/80 block mt-1">Requires immediate bulk purchasing</span>
                  </div>
                </div>

                {/* Store comparison matrix */}
                <div className="overflow-x-auto border border-zinc-800 rounded-xl bg-zinc-950">
                  <table className="w-full border-collapse text-left text-xs">
                    <thead>
                      <tr className="bg-zinc-900 border-b border-zinc-800 text-zinc-400 font-semibold text-[10px] uppercase tracking-wider">
                        <th className="p-3">Store Node</th>
                        <th className="p-3">Format Type / Location</th>
                        <th className="p-3 text-right">R² Accuracy Score</th>
                        <th className="p-3 text-right">Waste / Expiry Margin</th>
                        <th className="p-3 text-right">Active Inventory Value</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-900">
                      {STORE_PROFILES.map((store) => (
                        <tr key={store.id} className="hover:bg-zinc-900/40 text-zinc-300">
                          <td className="p-3 font-semibold text-white">{store.name}</td>
                          <td className="p-3">
                            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-zinc-800 text-[10px] text-zinc-300 font-medium">
                              {store.type} - {store.location}
                            </span>
                          </td>
                          <td className="p-3 text-right font-medium text-emerald-400">{store.metrics.r2}</td>
                          <td className={`p-3 text-right font-medium ${
                            parseFloat(store.metrics.wasteMargin) > 5 ? 'text-rose-400' : 'text-amber-400'
                          }`}>
                            {store.metrics.wasteMargin}
                          </td>
                          <td className="p-3 text-right text-zinc-300">₹{store.metrics.inventoryValue.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="flex justify-end">
                  <button
                    id="btn-launch-chain"
                    onClick={() => onSelectChain()}
                    className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm px-6 py-2.5 rounded-lg transition-all"
                  >
                    Enter Executive Control Tower <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Stage 1 Option: Pre-Launch Insights Deck */
        <div className="w-full max-w-5xl bg-zinc-900/60 backdrop-blur-xl border border-zinc-800/80 rounded-2xl shadow-2xl p-8 relative z-10 space-y-8 flex flex-col">
          {/* Deck Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-800 pb-6">
            <div>
              <div className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-[10px] text-emerald-400 font-bold uppercase tracking-wider mb-2">
                Predictive Analytics Report
              </div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <Building2 className="w-5 h-5 text-emerald-500" /> Pre-Launch Insights Deck: {formData.storeName}
              </h2>
              <p className="text-zinc-400 text-xs mt-1">
                Forecasting model outputs matching a <span className="font-semibold text-zinc-300">{formData.storeType}</span> store launch in <span className="font-semibold text-zinc-300">{formData.openingMonth}</span> at <span className="font-semibold text-zinc-300">{formData.locationType}</span> format.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowInsightsDeck(false)}
                className="px-4 py-2 rounded-lg border border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 text-xs font-semibold"
              >
                Modify Inputs
              </button>
              <button
                id="btn-activate-console"
                onClick={() => onSelectStore({
                  id: 'new_store',
                  name: formData.storeName,
                  type: formData.storeType,
                  location: formData.locationType,
                  investment: parseFloat(formData.investment) || 150000,
                  openingMonth: formData.openingMonth,
                  activeDays: 0,
                  metrics: {
                    r2: "0.923 (Base Model)",
                    wasteMargin: "0.0%",
                    stockouts: 0,
                    leakingMargin: "0.0%",
                    deficitCount: 0,
                    revenue: 0,
                    inventoryValue: parseFloat(formData.investment) || 150000
                  }
                })}
                className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs px-5 py-2.5 rounded-lg shadow-lg flex items-center gap-1.5"
              >
                Activate Store Console <CheckCircle2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Insights Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Box 1: Month-Specific Product Recommendations */}
            <div className="bg-zinc-950/60 border border-zinc-800/80 p-5 rounded-xl flex flex-col gap-4">
              <div>
                <h4 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-amber-400" /> Seasonal Shelf Strategy
                </h4>
                <p className="text-[10px] text-zinc-500 mt-1">Recommended launch list aligned with {formData.openingMonth} demand patterns.</p>
              </div>

              <div className="space-y-3 flex-1">
                {getProductRecommendations(formData.openingMonth).map((rec, idx) => (
                  <div key={idx} className="p-3 rounded-lg bg-zinc-900 border border-zinc-800/80 hover:border-zinc-700/80 transition-all flex justify-between items-center text-xs">
                    <div>
                      <span className="font-semibold text-zinc-100 block">{rec.name}</span>
                      <span className="text-[10px] text-zinc-500">{rec.category}</span>
                    </div>
                    <span className="inline-block px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[9px] font-medium border border-emerald-500/10 text-right max-w-[120px] truncate">
                      {rec.reason}
                    </span>
                  </div>
                ))}
              </div>
              <div className="p-3 rounded-lg bg-emerald-950/15 border border-emerald-900/30 text-[10px] text-emerald-300">
                🚀 Models recommend prioritizing staples to establish a reliable launch customer buffer.
              </div>
            </div>

            {/* Box 2: Capital Allocation Blueprint */}
            <div className="bg-zinc-950/60 border border-zinc-800/80 p-5 rounded-xl flex flex-col gap-4">
              <div>
                <h4 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <PieIcon className="w-4 h-4 text-emerald-400" /> Capital Allocation Blueprint
                </h4>
                <p className="text-[10px] text-zinc-500 mt-1">Optimal category investment split matching local seasonality weights.</p>
              </div>

              {/* Simple Custom SVG Donut Chart */}
              <div className="flex justify-center items-center h-32 relative">
                <svg width="120" height="120" viewBox="0 0 40 40" className="transform -rotate-90">
                  {/* Segment 1: Staples 35% */}
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#10b981" strokeWidth="4" strokeDasharray="35 65" strokeDashoffset="0" />
                  {/* Segment 2: Beverages 20% */}
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#14b8a6" strokeWidth="4" strokeDasharray="20 80" strokeDashoffset="-35" />
                  {/* Segment 3: Snacks 20% */}
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#f59e0b" strokeWidth="4" strokeDasharray="20 80" strokeDashoffset="-55" />
                  {/* Segment 4: Perishables 15% */}
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#3b82f6" strokeWidth="4" strokeDasharray="15 85" strokeDashoffset="-75" />
                  {/* Segment 5: Personal Care 10% */}
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#f43f5e" strokeWidth="4" strokeDasharray="10 90" strokeDashoffset="-90" />
                  
                  {/* Inner Hole */}
                  <circle cx="20" cy="20" r="13" fill="#18181b" />
                </svg>
                <div className="absolute text-center">
                  <span className="block text-xs font-semibold text-zinc-400">Total</span>
                  <span className="text-sm font-bold text-white">₹{(parseInt(formData.investment) || 150000).toLocaleString()}</span>
                </div>
              </div>

              <div className="space-y-2 flex-1 text-xs">
                {getCapitalAllocation(formData.investment).map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`w-2.5 h-2.5 rounded-full ${item.color}`} />
                      <span className="text-zinc-300 text-[11px]">{item.name}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-semibold text-zinc-100 block">₹{item.value.toLocaleString()}</span>
                      <span className="text-[9px] text-zinc-500 font-semibold uppercase">{item.pct}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Box 3: 3-Month Risk Matrix */}
            <div className="bg-zinc-950/60 border border-zinc-800/80 p-5 rounded-xl flex flex-col gap-4">
              <div>
                <h4 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-blue-400" /> 3-Month Risk Matrix
                </h4>
                <p className="text-[10px] text-zinc-500 mt-1">90-day forward demand path showing Cold Start and stabilization factors.</p>
              </div>

              {/* Custom SVG Line Chart */}
              <div className="h-32 bg-zinc-900/60 rounded-lg p-2 border border-zinc-800/50 flex flex-col justify-between">
                <div className="w-full h-full relative">
                  <svg viewBox="0 0 100 40" className="w-full h-full">
                    {/* Gridlines */}
                    <line x1="0" y1="35" x2="100" y2="35" stroke="#27272a" strokeWidth="0.5" />
                    <line x1="0" y1="20" x2="100" y2="20" stroke="#27272a" strokeWidth="0.5" strokeDasharray="1 1" />
                    <line x1="0" y1="5" x2="100" y2="5" stroke="#27272a" strokeWidth="0.5" strokeDasharray="1 1" />
                    
                    {/* Graph Line */}
                    {/* Path: Start at 40% (y = 26), Month 1 to 70% (y = 17) in Month 2, 100% (y = 5) in Month 3 */}
                    <path
                      d="M 5 28 Q 20 22 45 17 T 95 6"
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="2"
                    />

                    {/* Nodes */}
                    <circle cx="5" cy="28" r="1.5" fill="#f43f5e" /> {/* Cold Start Start */}
                    <circle cx="45" cy="17" r="1.5" fill="#f59e0b" /> {/* Scaling Up */}
                    <circle cx="95" cy="6" r="1.5" fill="#10b981" /> {/* Stabilized */}
                    
                    {/* Labels inside SVG */}
                    <text x="5" y="32" fill="#71717a" fontSize="3">Month 1</text>
                    <text x="45" y="22" fill="#71717a" fontSize="3">Month 2</text>
                    <text x="80" y="11" fill="#71717a" fontSize="3">Month 3</text>
                  </svg>
                </div>
              </div>

              <div className="space-y-3 flex-1 text-xs">
                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-rose-500/10 flex items-center justify-center text-[10px] font-bold text-rose-400 mt-0.5">1</div>
                  <div>
                    <span className="font-semibold text-zinc-200 block">Month 1: Cold Start Period (40% discount)</span>
                    <span className="text-[10px] text-zinc-500">Scaling applied to guard against early stock buffer bloating.</span>
                  </div>
                </div>
                
                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-amber-500/10 flex items-center justify-center text-[10px] font-bold text-amber-400 mt-0.5">2</div>
                  <div>
                    <span className="font-semibold text-zinc-200 block">Month 2-3: Ramping up (70% scaling)</span>
                    <span className="text-[10px] text-zinc-500">Market awareness and footfall begins to match local average.</span>
                  </div>
                </div>

                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-emerald-500/10 flex items-center justify-center text-[10px] font-bold text-emerald-400 mt-0.5">3</div>
                  <div>
                    <span className="font-semibold text-zinc-200 block">Month 4+: Stabilized forecasting</span>
                    <span className="text-[10px] text-zinc-500">ML models shift from cold start modes to full demand signal tracking.</span>
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
