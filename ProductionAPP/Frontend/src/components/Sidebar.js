import React from 'react';
import { 
  TrendingUp, 
  Package, 
  DollarSign, 
  FileText, 
  Cpu, 
  Store, 
  LogOut, 
  Layers 
} from 'lucide-react';

export default function Sidebar({ 
  activeView, 
  setActiveView, 
  storeInfo, 
  onBackToGateway 
}) {
  const menuItems = [
    { id: 'demand', label: 'Demand Forecasting Hub', icon: TrendingUp },
    { id: 'inventory', label: 'Scientific Replenishment', icon: Package },
    { id: 'financial', label: 'Financial ROI & Capital', icon: DollarSign },
    { id: 'transactions', label: 'Daily Transactions Log', icon: FileText },
    { id: 'learning', label: 'Algorithms & Learning', icon: Cpu }
  ];

  const getStoreBadgeColor = (type) => {
    switch (type) {
      case 'Supermarket': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'Medium': return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      default: return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
    }
  };

  return (
    <aside className="w-64 bg-zinc-950 border-r border-zinc-800 flex flex-col h-screen fixed left-0 top-0 text-zinc-300 font-sans z-20">
      {/* Brand Header */}
      <div className="p-6 border-b border-zinc-800/80 flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center text-white font-bold shadow-md shadow-emerald-900/30">
          <Layers className="w-4 h-4" />
        </div>
        <div>
          <span className="font-bold text-white block text-sm tracking-wide">SMART RETAIL AI</span>
          <span className="text-[10px] text-zinc-500 uppercase tracking-widest font-semibold block">Demand Engine v1.0</span>
        </div>
      </div>

      {/* Store Context Badge */}
      <div className="px-6 py-4 border-b border-zinc-900/60 bg-zinc-900/20 space-y-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-zinc-400">
          <Store className="w-3.5 h-3.5" />
          <span>Active Context</span>
        </div>
        
        {storeInfo ? (
          <div className="space-y-1">
            <span className="block text-xs font-bold text-white truncate">{storeInfo.name}</span>
            <div className="flex flex-wrap gap-1.5 pt-1">
              <span className={`px-2 py-0.5 rounded text-[9px] font-bold border uppercase tracking-wider ${getStoreBadgeColor(storeInfo.type)}`}>
                {storeInfo.type}
              </span>
              <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-zinc-900 text-zinc-400 border border-zinc-800 uppercase tracking-wider">
                {storeInfo.location}
              </span>
            </div>
          </div>
        ) : (
          <div className="space-y-1">
            <span className="block text-xs font-bold text-white">Executive Control Mode</span>
            <div className="flex gap-1.5 pt-1">
              <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-blue-500/10 text-blue-400 border border-blue-500/20 uppercase tracking-wider">
                Multi-Store
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Nav Menu */}
      <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
        {menuItems.map(item => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          return (
            <button
              id={`nav-item-${item.id}`}
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-xs font-semibold transition-all group ${
                isActive 
                  ? 'bg-zinc-900 text-white border border-zinc-800 shadow-md shadow-black/40' 
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/30'
              }`}
            >
              <Icon className={`w-4 h-4 transition-all shrink-0 ${
                isActive ? 'text-emerald-500 scale-105' : 'text-zinc-500 group-hover:text-zinc-400'
              }`} />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Back to Gateway / Workspace exit */}
      <div className="p-4 border-t border-zinc-900 bg-zinc-950">
        <button
          id="btn-back-gateway"
          onClick={onBackToGateway}
          className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg border border-zinc-800/80 hover:border-zinc-700/80 text-zinc-400 hover:text-zinc-200 text-xs font-semibold hover:bg-zinc-900/20 transition-all"
        >
          <LogOut className="w-3.5 h-3.5" />
          <span>Exit Workspace</span>
        </button>
      </div>
    </aside>
  );
}
