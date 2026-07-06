import React from 'react';
import { 
  TrendingUp, 
  Package, 
  DollarSign, 
  FileText, 
  Settings as SettingsIcon, 
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
  // Navigation Menu Items - Replaced 'learning' with 'settings'
  const menuItems = [
    { id: 'demand', label: 'Demand Forecasting Hub', icon: TrendingUp },
    { id: 'inventory', label: 'Scientific Replenishment', icon: Package },
    { id: 'financial', label: 'Financial ROI & Capital', icon: DollarSign },
    { id: 'transactions', label: 'Daily Transactions Log', icon: FileText },
    { id: 'settings', label: 'Platform Settings', icon: SettingsIcon }
  ];

  const getStoreBadgeColor = (type) => {
    switch (type) {
      case 'Supermarket': return 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20';
      case 'Medium': return 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20';
      default: return 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20';
    }
  };

  return (
    <aside className="w-64 bg-white dark:bg-zinc-950 border-r border-zinc-200 dark:border-zinc-800 flex flex-col h-screen fixed left-0 top-0 text-zinc-700 dark:text-zinc-300 font-sans z-20 transition-colors duration-200">
      {/* Brand Header */}
      <div className="p-6 border-b border-zinc-150 dark:border-zinc-800/80 flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center text-white font-bold shadow-md shadow-emerald-900/30">
          <Layers className="w-4.5 h-4.5" />
        </div>
        <div>
          {/* Increased Font Size of Header Title to 15px */}
          <span className="font-bold text-zinc-900 dark:text-white block text-[15px] tracking-wide">SMART RETAIL AI</span>
          {/* Increased Subtitle size to 11px */}
          <span className="text-[11px] text-zinc-400 dark:text-zinc-500 uppercase tracking-widest font-bold block mt-0.5">Demand Engine v1.0</span>
        </div>
      </div>

      {/* Store Context Badge */}
      <div className="px-6 py-4 border-b border-zinc-100 dark:border-zinc-900/60 bg-zinc-50/50 dark:bg-zinc-900/20 space-y-2">
        {/* Increased Label Font size to 13px */}
        <div className="flex items-center gap-2 text-[13px] font-bold text-zinc-400 dark:text-zinc-500">
          <Store className="w-[15px] h-[15px]" />
          <span>Active Context</span>
        </div>
        
        {storeInfo ? (
          <div className="space-y-1">
            {/* Increased Store Name to 13.5px */}
            <span className="block text-[13.5px] font-extrabold text-zinc-900 dark:text-white truncate">{storeInfo.name}</span>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {/* Increased Badge text size to 10px */}
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold border uppercase tracking-wider ${getStoreBadgeColor(storeInfo.type)}`}>
                {storeInfo.type}
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-zinc-100 dark:bg-zinc-900 text-zinc-500 dark:text-zinc-400 border border-zinc-200 dark:border-zinc-800 uppercase tracking-wider">
                {storeInfo.location}
              </span>
            </div>
          </div>
        ) : (
          <div className="space-y-1">
            <span className="block text-[13.5px] font-extrabold text-zinc-900 dark:text-white">Executive Control Mode</span>
            <div className="flex gap-1.5 pt-1">
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 uppercase tracking-wider">
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
              /* Increased Navigation Font Size from text-xs (12px) to text-[13px] */
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-[13px] font-bold transition-all group ${
                isActive 
                  ? 'bg-zinc-100 dark:bg-zinc-900 text-zinc-900 dark:text-white border border-zinc-200 dark:border-zinc-800 shadow-sm' 
                  : 'text-zinc-550 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 hover:bg-zinc-100/50 dark:hover:bg-zinc-900/30'
              }`}
            >
              {/* Increased Nav Icons from w-4 h-4 to w-[17px] h-[17px] */}
              <Icon className={`w-[17px] h-[17px] transition-all shrink-0 ${
                isActive ? 'text-emerald-500 scale-105' : 'text-zinc-400 dark:text-zinc-500 group-hover:text-zinc-900 dark:group-hover:text-zinc-300'
              }`} />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* User's Account Card (Placed above Exit Workspace) */}
      <div className="mx-4 mb-2 p-3 bg-zinc-50 dark:bg-zinc-900/60 border border-zinc-200/80 dark:border-zinc-800/80 rounded-xl flex items-center gap-3 transition-colors duration-200">
        <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-emerald-500 to-teal-500 text-white font-extrabold text-xs flex items-center justify-center shrink-0 shadow-inner">
          RA
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="block text-[13px] font-bold text-zinc-900 dark:text-white truncate">Raja Admin</span>
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shrink-0" />
          </div>
          <span className="block text-[10px] text-zinc-500 dark:text-zinc-400 font-bold truncate mt-0.5">Senior Operations Mgr</span>
        </div>
      </div>

      {/* Exit Workspace */}
      <div className="p-4 border-t border-zinc-150 dark:border-zinc-900 bg-zinc-50/20 dark:bg-zinc-950/60">
        <button
          id="btn-back-gateway"
          onClick={onBackToGateway}
          /* Increased font size to text-[13px] */
          className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg border border-zinc-200 dark:border-zinc-800/80 hover:border-zinc-300 dark:hover:border-zinc-700/80 text-zinc-550 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 text-[13px] font-bold hover:bg-zinc-100/50 dark:hover:bg-zinc-900/20 transition-all"
        >
          <LogOut className="w-[15px] h-[15px]" />
          <span>Exit Workspace</span>
        </button>
      </div>
    </aside>
  );
}
