import React, { useState } from 'react';
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
  ChevronLeft
} from 'lucide-react';
import { STORE_PROFILES, PRODUCT_CATALOG, FESTIVALS } from './mockData';

export default function Gateway({ onSelectStore, onSelectChain }) {
  // gatewayState: 'landing', 'register', 'login', 'chain', 'insights'
  const [gatewayState, setGatewayState] = useState('landing');
  
  // Registration Form State (Directly matching the image input fields)
  const [formData, setFormData] = useState({
    storeName: '',
    storeType: 'Supermarket',
    locationType: 'Urban',
    openingMonth: 'October',
    investment: '850000'
  });
  
  const [selectedProfileId, setSelectedProfileId] = useState(STORE_PROFILES[0].id);

  // Handle inputs
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  // Submit New Store
  const handleRegisterSubmit = (e) => {
    e.preventDefault();
    if (!formData.storeName.trim()) return;
    setGatewayState('insights');
  };

  const selectedProfile = STORE_PROFILES.find(p => p.id === selectedProfileId);

  // Generate live capital allocation percentages for preview donut
  const getDonutAllocation = (investment) => {
    const amt = parseFloat(investment) || 0;
    return {
      staples: Math.round(amt * 0.40),
      perishables: Math.round(amt * 0.30),
      other: Math.round(amt * 0.30)
    };
  };

  // Generate live capital allocation breakdown for the pre-launch insights deck
  const getCapitalAllocation = (investment) => {
    const amt = parseFloat(investment) || 150000;
    return [
      { name: 'Staples & Grains', value: Math.round(amt * 0.35), pct: '35%', color: 'bg-emerald-500' },
      { name: 'Beverages', value: Math.round(amt * 0.20), pct: '20%', color: 'bg-teal-500' },
      { name: 'Snacks & Biscuits', value: Math.round(amt * 0.20), pct: '20%', color: 'bg-amber-500' },
      { name: 'Perishables', value: Math.round(amt * 0.15), pct: '15%', color: 'bg-blue-500' },
      { name: 'Personal Care', value: Math.round(amt * 0.10), pct: '10%', color: 'bg-rose-500' }
    ];
  };

  const alloc = getDonutAllocation(formData.investment);

  // Product recommendations based on month
  const getProductRecommendations = (month) => {
    const matchedFestivals = FESTIVALS.filter(f => f.month.toLowerCase() === month.toLowerCase());
    const recommendations = [
      { name: 'Tata Tea Premium 250g', category: 'Beverages', reason: 'High base turnover staple' },
      { name: 'Fortune Sunflower Oil 1L', category: 'Non-Perishables', reason: 'Everyday necessity item' },
      { name: 'Amul Salted Butter 100g', category: 'Perishables', reason: 'Consistent cold storage demand' },
      { name: 'Aashirvaad Chakki Atta 5kg', category: 'Non-Perishables', reason: 'Daily kitchen staple item' },
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
    <div className="min-h-screen bg-[#131920] text-zinc-100 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 font-sans transition-colors duration-200 relative overflow-hidden">
      
      {/* Background Radial Glow */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-emerald-500/5 via-zinc-950/20 to-black pointer-events-none opacity-85" />
      
      {/* Floating Sparkle Icon on main page background (matching mockup image positioning) */}
      {gatewayState === 'register' && (
        <div className="absolute bottom-16 right-16 lg:right-32 pointer-events-none opacity-50 z-0">
          <svg viewBox="0 0 24 24" className="w-10 h-10 fill-zinc-650 animate-pulse">
            <path d="M12 0L14.6 9.4L24 12L14.6 14.6L12 24L9.4 14.6L0 12L9.4 9.4L12 0Z" />
          </svg>
        </div>
      )}

      {/* Landing Page State */}
      {gatewayState === 'landing' && (
        <div className="w-full max-w-4xl bg-[#1c232b] border border-zinc-800/80 rounded-2xl shadow-2xl p-8 relative z-10 space-y-8 flex flex-col">
          {/* Logo & Headline */}
          <div className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold tracking-wider uppercase mb-2">
              <Sparkles className="w-3.5 h-3.5 animate-pulse" /> AI Demand Engine
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
              Smart Retail forecasting console
            </h1>
            <p className="text-zinc-400 text-sm max-w-xl mx-auto">
              Choose an operational path to begin. Initialize new store constraints or login to manage active merchant inventory.
            </p>
          </div>

          {/* Main Action Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
            {/* Action 1: Login */}
            <button
              id="landing-btn-login"
              onClick={() => setGatewayState('login')}
              className="flex flex-col text-left p-6 bg-[#202730]/60 hover:bg-[#202730] border border-zinc-800 hover:border-zinc-700 rounded-2xl transition-all gap-4 group"
            >
              <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
                <LogIn className="w-6 h-6 text-amber-500" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-white group-hover:text-amber-400 transition-colors">Login to Your Store</h3>
                <p className="text-xs text-zinc-400 leading-relaxed">
                  Synchronize with active store databases, run reorder matrices, and log daily employee transactions.
                </p>
              </div>
              <div className="flex items-center gap-1.5 text-xs font-bold text-zinc-300 pt-2 mt-auto">
                Access Workspace <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>

            {/* Action 2: Onboarding */}
            <button
              id="landing-btn-register"
              onClick={() => setGatewayState('register')}
              className="flex flex-col text-left p-6 bg-[#202730]/60 hover:bg-[#202730] border border-zinc-800 hover:border-zinc-700 rounded-2xl transition-all gap-4 group"
            >
              <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                <Building2 className="w-6 h-6 text-emerald-500" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-white group-hover:text-emerald-400 transition-colors">New Store Registration</h3>
                <p className="text-xs text-zinc-400 leading-relaxed">
                  Initialize a new retail format. Analyze opening month seasonality, budget allocations, and cold-start projections.
                </p>
              </div>
              <div className="flex items-center gap-1.5 text-xs font-bold text-zinc-300 pt-2 mt-auto">
                Start Onboarding <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>
          </div>

          {/* Executive Link Footer */}
          <div className="flex justify-center border-t border-zinc-800/60 pt-6">
            <button
              id="landing-btn-chain"
              onClick={() => setGatewayState('chain')}
              className="flex items-center gap-2 text-xs font-semibold text-zinc-550 dark:text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              <Layers className="w-4 h-4 text-blue-500" />
              <span>Multi-Store Executive Control Mode</span>
            </button>
          </div>
        </div>
      )}

      {/* Onboarding Register State (Rebuilt to match mockup image exactly) */}
      {gatewayState === 'register' && (
        <form onSubmit={handleRegisterSubmit} className="w-full max-w-5xl bg-[#1c232b] border border-zinc-800/80 rounded-2xl shadow-2xl overflow-hidden relative z-10 flex flex-col min-h-[520px]">
          
          {/* macOS-style Top Bar */}
          <div className="relative p-4 bg-[#14191f] border-b border-zinc-800/60 flex items-center justify-center h-12 shrink-0">
            {/* Titlebar Dots (Clickable Red close button to return to menu) */}
            <div className="absolute left-4 flex items-center gap-1.5 group">
              <span 
                onClick={() => setGatewayState('landing')} 
                className="w-3 h-3 rounded-full bg-rose-500 hover:bg-rose-600 cursor-pointer flex items-center justify-center text-[7.5px] text-rose-950 font-bold shrink-0 transition-colors after:content-[''] hover:after:content-['×']"
                title="Go Back"
              />
              <span className="w-3 h-3 rounded-full bg-amber-500 hover:bg-amber-600 cursor-pointer flex items-center justify-center text-[7.5px] text-amber-950 font-bold shrink-0 transition-colors after:content-[''] hover:after:content-['−']" />
              <span className="w-3 h-3 rounded-full bg-emerald-500 hover:bg-emerald-600 cursor-pointer flex items-center justify-center text-[7.5px] text-emerald-950 font-bold shrink-0 transition-colors after:content-[''] hover:after:content-['+']" />
            </div>

            {/* Connected Progress steps bar (Centered) */}
            <div className="flex items-center gap-6 text-[11px] font-bold tracking-wide">
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-emerald-600 flex items-center justify-center text-white font-bold text-[10px]">1</span>
                <span className="text-zinc-100">Onboarding</span>
              </div>
              <div className="w-16 h-0.5 bg-emerald-600 rounded-full" />
              
              <div className="flex items-center gap-2 text-zinc-500">
                <span className="w-5 h-5 rounded-full border border-zinc-700 flex items-center justify-center text-[10px] font-bold">2</span>
                <span>Store Registration</span>
              </div>
              <div className="w-16 h-0.5 bg-zinc-800 rounded-full" />

              <div className="flex items-center gap-2 text-zinc-500">
                <span className="w-5 h-5 rounded-full border border-zinc-700 flex items-center justify-center text-[10px] font-bold">3</span>
                <span>First Context</span>
              </div>
            </div>
          </div>

          {/* Form + Preview Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 p-8 flex-1 items-start">
            {/* Left 3 Columns: Registration form fields */}
            <div className="lg:col-span-3 space-y-6">
              <div className="space-y-1">
                <h2 className="text-2xl font-bold text-white tracking-wide">New Store Registration</h2>
                <p className="text-xs text-zinc-400">Enter a retail owner to manage your shop details.</p>
              </div>

              <div className="grid grid-cols-1 gap-5">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest block">Store Name</label>
                  <input
                    id="input-store-name"
                    type="text"
                    name="storeName"
                    value={formData.storeName}
                    onChange={handleInputChange}
                    className="w-full bg-[#202730] border border-[#2e3743] rounded-lg px-4 py-2.5 text-xs text-white placeholder-zinc-550 focus:outline-none focus:border-emerald-500"
                    required
                  />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest block">Store Type</label>
                    <select
                      id="select-store-type"
                      name="storeType"
                      value={formData.storeType}
                      onChange={handleInputChange}
                      className="w-full bg-[#202730] border border-[#2e3743] rounded-lg px-4 py-2.5 text-xs text-white focus:outline-none focus:border-emerald-500 appearance-none bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2523a1a1aa%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E')] bg-[length:10px_10px] bg-[right_1rem_center] bg-no-repeat"
                    >
                      <option value="Small">Small</option>
                      <option value="Medium">Medium</option>
                      <option value="Supermarket">Supermarket</option>
                    </select>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest block">Regional Location</label>
                    <select
                      id="select-location"
                      name="locationType"
                      value={formData.locationType}
                      onChange={handleInputChange}
                      className="w-full bg-[#202730] border border-[#2e3743] rounded-lg px-4 py-2.5 text-xs text-white focus:outline-none focus:border-emerald-500 appearance-none bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2523a1a1aa%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E')] bg-[length:10px_10px] bg-[right_1rem_center] bg-no-repeat"
                    >
                      <option value="Urban">Urban</option>
                      <option value="Semi-Urban">Semi-Urban</option>
                      <option value="Rural">Rural</option>
                    </select>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest block">Initial Capital Investment</label>
                  <div className="relative">
                    <input
                      id="input-investment"
                      type="number"
                      name="investment"
                      value={formData.investment}
                      onChange={handleInputChange}
                      className="w-full bg-[#202730] border border-[#2e3743] rounded-lg px-4 py-2.5 text-xs text-white placeholder-zinc-550 focus:outline-none focus:border-emerald-500"
                      required
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Right 2 Columns: Live Preview Card (Capital Allocation Blueprint) */}
            <div className="lg:col-span-2 h-full flex flex-col justify-center">
              {/* Glassmorphism Card */}
              <div className="w-full bg-[#202730]/40 backdrop-blur-md border border-zinc-700/30 rounded-2xl p-6 shadow-2xl flex flex-col justify-between gap-5 min-h-[280px]">
                <div className="space-y-1.5">
                  <h3 className="text-sm font-bold text-white tracking-wide">Capital Allocation Blueprint</h3>
                  <p className="text-[10px] text-zinc-400 leading-normal">
                    AI-driven recommendation blueprint based on historical data.
                  </p>
                </div>

                {/* SVG Donut flanked exactly by the labels */}
                <div className="flex items-center justify-center gap-5 my-2">
                  <div className="text-right">
                    <span className="block text-base font-black text-white">30%</span>
                    <span className="block text-[8px] text-zinc-450 uppercase font-bold tracking-wider">Perishables</span>
                  </div>

                  <svg width="90" height="90" viewBox="0 0 40 40" className="transform -rotate-90 shrink-0">
                    <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#272f3a" strokeWidth="4" />
                    <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#10b981" strokeWidth="4.2" strokeDasharray="40 60" strokeDashoffset="0" />
                    <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#718096" strokeWidth="4.2" strokeDasharray="30 70" strokeDashoffset="-40" />
                    <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#4a5568" strokeWidth="4" strokeDasharray="30 70" strokeDashoffset="-70" />
                    <circle cx="20" cy="20" r="13" fill="#1b2229" />
                  </svg>

                  <div className="text-left">
                    <span className="block text-base font-black text-white">40%</span>
                    <span className="block text-[8px] text-zinc-450 uppercase font-bold tracking-wider">Staples</span>
                  </div>
                </div>

                <div className="border-t border-zinc-800/85 pt-3 flex justify-between items-center text-[9px] text-zinc-450">
                  <div>
                    <span className="block font-semibold">Staples (40%):</span>
                    <span className="font-bold text-emerald-400">₹{alloc.staples.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="block font-semibold">Perishables (30%):</span>
                    <span className="font-bold text-zinc-300">₹{alloc.perishables.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Clean Footer Controls matching image (No back button on left, button on right) */}
          <div className="p-5 bg-[#14191f] border-t border-zinc-800/40 flex items-center justify-end relative h-16 shrink-0">
            <button
              id="btn-initialize-engine"
              type="submit"
              className="bg-[#10b981] hover:bg-emerald-400 text-zinc-950 font-extrabold text-xs px-6 py-2.5 rounded-lg transition-all shadow-md shadow-emerald-950/20"
            >
              Initialize Store Engine
            </button>
          </div>

        </form>
      )}

      {/* Existing Store Login State */}
      {gatewayState === 'login' && (
        <div className="w-full max-w-xl bg-[#1c232b] border border-zinc-800/80 rounded-2xl shadow-2xl p-8 relative z-10 space-y-6">
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-3">
            <button
              onClick={() => setGatewayState('landing')}
              className="p-1 rounded bg-[#202730] border border-zinc-850 hover:bg-zinc-805 text-zinc-400"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <div>
              <h3 className="text-base font-bold text-white">Optimize Active Store</h3>
              <p className="text-[10px] text-zinc-500">Sync with an active registered merchant profile database.</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs font-semibold text-zinc-400 uppercase block">Select Active Store Profile</label>
              <select
                id="select-store-profile"
                value={selectedProfileId}
                onChange={(e) => setSelectedProfileId(e.target.value)}
                className="w-full bg-[#202730] border border-[#2e3743] rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-amber-500 text-sm"
              >
                {STORE_PROFILES.map(p => (
                  <option key={p.id} value={p.id}>{p.name} ({p.type} - {p.location})</option>
                ))}
              </select>
            </div>

            <div className="p-4 rounded-xl bg-zinc-950/40 border border-zinc-800 space-y-4 text-xs">
              <div className="flex items-center gap-2 text-white font-semibold">
                <Info className="w-3.5 h-3.5 text-amber-500" />
                <span>Profile Details</span>
              </div>
              <div className="grid grid-cols-2 gap-3 text-zinc-400">
                <div>
                  <span className="block text-[10px] text-zinc-500 font-bold uppercase">Active Period</span>
                  <span className="text-xs font-semibold text-zinc-300">{selectedProfile.activeDays} Days Logged</span>
                </div>
                <div>
                  <span className="block text-[10px] text-zinc-500 font-bold uppercase">Predictive Accuracy</span>
                  <span className="text-xs font-bold text-emerald-400">{parseFloat(selectedProfile.metrics.r2) * 100}% R²</span>
                </div>
              </div>
            </div>

            {/* Health indicators */}
            <div className="space-y-3 pt-2">
              <div className="p-3.5 rounded-xl bg-rose-950/20 border border-rose-900/30 flex items-start gap-2.5 text-[11px] text-rose-350 leading-relaxed">
                <AlertTriangle className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
                <p>
                  Found <span className="font-semibold text-rose-300">{selectedProfile.metrics.deficitCount} critical supply shortages</span> requiring immediate reorder action in replenishment view.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-amber-950/10 border border-amber-900/30 flex items-start gap-2.5 text-[11px] text-amber-350 leading-relaxed">
                <Activity className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                <p>
                  Historical waste margin at <span className="font-semibold text-amber-300">{selectedProfile.metrics.wasteMargin}</span> due to dead stocks locking capital turnover.
                </p>
              </div>
            </div>

            <button
              id="btn-sync-store"
              onClick={() => onSelectStore(selectedProfile)}
              className="w-full py-3 bg-[#10b981] hover:bg-emerald-400 text-zinc-950 font-bold text-xs rounded-lg transition-all flex items-center justify-center gap-1.5 shadow-lg"
            >
              Sync & Access Operations Center <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Executive Control State */}
      {gatewayState === 'chain' && (
        <div className="w-full max-w-4xl bg-[#1c232b] border border-zinc-800/80 rounded-2xl shadow-2xl p-8 relative z-10 space-y-6">
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-3">
            <button
              onClick={() => setGatewayState('landing')}
              className="p-1 rounded bg-[#202730] border border-zinc-850 hover:bg-zinc-805 text-zinc-400"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <div>
              <h3 className="text-base font-bold text-white">Multi-Store Executive Control Tower</h3>
              <p className="text-[10px] text-zinc-500">Aggregated operations and performance benchmarking across all registered nodes.</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-zinc-950/40 border border-zinc-800 p-4 rounded-xl">
              <span className="text-[10px] text-zinc-500 font-bold uppercase block">Total Chain Revenue</span>
              <span className="text-lg font-bold text-white">₹455,900</span>
              <span className="text-[10px] text-emerald-400 block mt-1">+14.2% projected MoM</span>
            </div>
            <div className="bg-zinc-950/40 border border-zinc-800 p-4 rounded-xl">
              <span className="text-[10px] text-zinc-500 font-bold uppercase block">Active Asset Value</span>
              <span className="text-lg font-bold text-teal-400">₹305,400</span>
              <span className="text-[10px] text-zinc-500 block mt-1">Spread across 3 active store formats</span>
            </div>
            <div className="bg-zinc-950/40 border border-zinc-800 p-4 rounded-xl">
              <span className="text-[10px] text-zinc-500 font-bold uppercase block">Global Stockout Alerts</span>
              <span className="text-lg font-bold text-rose-500">17 Items</span>
              <span className="text-[10px] text-rose-400 block mt-1">Requires immediate bulk purchasing</span>
            </div>
          </div>

          {/* Table */}
          <div className="overflow-x-auto border border-zinc-800 rounded-xl bg-zinc-950/50">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="bg-zinc-900/60 border-b border-zinc-800 text-zinc-500 font-bold text-[9px] uppercase tracking-wider">
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
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-zinc-900 text-[9px] text-zinc-400 border border-zinc-800">
                        {store.type} - {store.location}
                      </span>
                    </td>
                    <td className="p-3 text-right font-medium text-emerald-400">{store.metrics.r2}</td>
                    <td className="p-3 text-right font-medium text-amber-400">{store.metrics.wasteMargin}</td>
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
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs px-6 py-2.5 rounded-lg transition-all shadow"
            >
              Access Executive Control Tower <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Pre-Launch Insights Deck State */}
      {gatewayState === 'insights' && (
        <div className="w-full max-w-5xl bg-[#1c232b] border border-zinc-800/80 rounded-2xl shadow-2xl p-8 relative z-10 space-y-8 flex flex-col">
          {/* Deck Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-800 pb-6">
            <div>
              <div className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-[10px] text-emerald-400 font-bold uppercase tracking-wider mb-2">
                Predictive Analytics Report
              </div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <Building2 className="w-5 h-5 text-emerald-500" /> Pre-Launch Insights Deck: {formData.storeName}
              </h2>
              <p className="text-zinc-455 text-xs mt-1">
                Forecasting model outputs matching a <span className="font-semibold text-zinc-200">{formData.storeType}</span> store launch in <span className="font-semibold text-zinc-200">{formData.openingMonth}</span> at <span className="font-semibold text-zinc-200">{formData.locationType}</span> format.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setGatewayState('register')}
                className="px-4 py-2 rounded-lg border border-zinc-850 text-zinc-400 hover:text-zinc-250 hover:bg-zinc-800 text-xs font-bold"
              >
                Modify Inputs
              </button>
              <button
                id="btn-activate-console"
                type="button"
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
                className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs px-5 py-2.5 rounded-lg shadow-lg flex items-center gap-1.5"
              >
                Activate Store Console <CheckCircle2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Insights Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Box 1: Month recommendations */}
            <div className="bg-[#202730]/40 border border-zinc-800 p-5 rounded-xl flex flex-col gap-4">
              <div>
                <h4 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-amber-500" /> Seasonal Shelf Strategy
                </h4>
                <p className="text-[10px] text-zinc-500 mt-1">Recommended launch list aligned with {formData.openingMonth} demand patterns.</p>
              </div>

              <div className="space-y-3 flex-1">
                {getProductRecommendations(formData.openingMonth).map((rec, idx) => (
                  <div key={idx} className="p-3 rounded-lg bg-zinc-950/40 border border-zinc-800 hover:border-zinc-700 transition-all flex justify-between items-center text-xs">
                    <div>
                      <span className="font-bold text-zinc-100 block">{rec.name}</span>
                      <span className="text-[10px] text-zinc-500">{rec.category}</span>
                    </div>
                    <span className="inline-block px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[9px] font-bold border border-emerald-500/10 text-right max-w-[120px] truncate">
                      {rec.reason}
                    </span>
                  </div>
                ))}
              </div>
              <div className="p-3 rounded-lg bg-emerald-950/15 border border-emerald-900/30 text-[10px] text-emerald-300">
                🚀 Models recommend prioritizing staples to establish a reliable launch customer buffer.
              </div>
            </div>

            {/* Box 2: Capital Allocation */}
            <div className="bg-[#202730]/40 border border-zinc-800 p-5 rounded-xl flex flex-col gap-4">
              <div>
                <h4 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <PieIcon className="w-4 h-4 text-emerald-500" /> Capital Allocation Blueprint
                </h4>
                <p className="text-[10px] text-zinc-500 mt-1">Optimal category investment split matching local seasonality weights.</p>
              </div>

              {/* Simple Donut Chart */}
              <div className="flex justify-center items-center h-32 relative">
                <svg width="120" height="120" viewBox="0 0 40 40" className="transform -rotate-90">
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#10b981" strokeWidth="4" strokeDasharray="35 65" strokeDashoffset="0" />
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#14b8a6" strokeWidth="4" strokeDasharray="20 80" strokeDashoffset="-35" />
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#f59e0b" strokeWidth="4" strokeDasharray="20 80" strokeDashoffset="-55" />
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#3b82f6" strokeWidth="4" strokeDasharray="15 85" strokeDashoffset="-75" />
                  <circle cx="20" cy="20" r="15.915" fill="transparent" stroke="#f43f5e" strokeWidth="4" strokeDasharray="10 90" strokeDashoffset="-90" />
                  <circle cx="20" cy="20" r="13" fill="#1b2229" />
                </svg>
                <div className="absolute text-center">
                  <span className="block text-xs font-bold text-zinc-400">Total</span>
                  <span className="text-sm font-black text-white">₹{(parseInt(formData.investment) || 150000).toLocaleString()}</span>
                </div>
              </div>

              <div className="space-y-2 flex-1 text-xs">
                {getCapitalAllocation(formData.investment).map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`w-2.5 h-2.5 rounded-full ${item.color}`} />
                      <span className="text-zinc-300 text-[11px] font-semibold">{item.name}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-bold text-zinc-100 block">₹{item.value.toLocaleString()}</span>
                      <span className="text-[9px] text-zinc-400 dark:text-zinc-500 font-bold uppercase">{item.pct}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Box 3: 3-Month Risk Matrix */}
            <div className="bg-[#202730]/40 border border-zinc-800 p-5 rounded-xl flex flex-col gap-4">
              <div>
                <h4 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-blue-500" /> 3-Month Risk Matrix
                </h4>
                <p className="text-[10px] text-zinc-500 mt-1">90-day forward demand path showing Cold Start and stabilization factors.</p>
              </div>

              {/* Custom Line Chart */}
              <div className="h-32 bg-[#202730]/60 rounded-lg p-2 border border-zinc-800/50 flex flex-col justify-between">
                <div className="w-full h-full relative">
                  <svg viewBox="0 0 100 40" className="w-full h-full">
                    <line x1="0" y1="35" x2="100" y2="35" className="stroke-zinc-800" strokeWidth="0.5" />
                    <line x1="0" y1="20" x2="100" y2="20" className="stroke-zinc-800" strokeWidth="0.5" strokeDasharray="1 1" />
                    <line x1="0" y1="5" x2="100" y2="5" className="stroke-zinc-800" strokeWidth="0.5" strokeDasharray="1 1" />
                    <path
                      d="M 5 28 Q 20 22 45 17 T 95 6"
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="2"
                    />
                    <circle cx="5" cy="28" r="1.5" fill="#f43f5e" />
                    <circle cx="45" cy="17" r="1.5" fill="#f59e0b" />
                    <circle cx="95" cy="6" r="1.5" fill="#10b981" />
                    <text x="5" y="32" className="fill-zinc-550" fontSize="3">Month 1</text>
                    <text x="45" y="22" className="fill-zinc-550" fontSize="3">Month 2</text>
                    <text x="80" y="11" className="fill-zinc-550" fontSize="3">Month 3</text>
                  </svg>
                </div>
              </div>

              <div className="space-y-3 flex-1 text-xs">
                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-rose-500/10 flex items-center justify-center text-[10px] font-bold text-rose-400 mt-0.5">1</div>
                  <div>
                    <span className="font-bold text-zinc-200 block">Month 1: Cold Start Period (40% discount)</span>
                    <span className="text-[10px] text-zinc-500">Scaling applied to guard against early stock buffer bloating.</span>
                  </div>
                </div>
                
                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-amber-500/10 flex items-center justify-center text-[10px] font-bold text-amber-400 mt-0.5">2</div>
                  <div>
                    <span className="font-bold text-zinc-200 block">Month 2-3: Ramping up (70% scaling)</span>
                    <span className="text-[10px] text-zinc-500">Market awareness and footfall begins to match local average.</span>
                  </div>
                </div>

                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 rounded-full bg-emerald-500/10 flex items-center justify-center text-[10px] font-bold text-emerald-400 mt-0.5">3</div>
                  <div>
                    <span className="font-bold text-zinc-200 block">Month 4+: Stabilized forecasting</span>
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
