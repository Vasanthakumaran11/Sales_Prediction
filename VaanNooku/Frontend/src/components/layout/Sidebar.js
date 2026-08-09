"use client";

import React from "react";
import {
  Gauge,
  BarChart3,
  TrendingUp,
  PackagePlus,
  Package,
  LayoutGrid,
  Store,
  Users,
  BellRing,
  FileText,
  Sparkles,
  Settings as SettingsIcon,
  ShoppingCart,
  History as HistoryIcon,
} from "lucide-react";
import { useStoreContext } from "@/context/StoreContext";

export function Sidebar() {
  const { activeView, setActiveView, activeStore } = useStoreContext();

  const menuItems = [
    { id: "data-entry", label: "Data Entry", icon: Gauge },
    { id: "sales", label: "Sales Analytics", icon: BarChart3 },
    { id: "ai-predictions", label: "AI Predictions", icon: Sparkles },
    { id: "inventory", label: "Inventory", icon: PackagePlus },
    { id: "products", label: "Products", icon: Package },
    { id: "suppliers", label: "Suppliers", icon: Users },
    { id: "alerts", label: "Alerts & Risks", icon: BellRing },
    { id: "history", label: "History", icon: HistoryIcon },
    { id: "settings", label: "Settings", icon: SettingsIcon },
  ];

  const getStoreInitials = (name) => {
    if (!name) return "HQ";
    if (name.toLowerCase().includes("store #")) {
      const match = name.match(/#(\d+)/);
      if (match) return match[1];
    }
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .slice(0, 2)
      .toUpperCase();
  };

  const initials = getStoreInitials(activeStore?.name);
  const storeName = activeStore?.name || "Executive HQ";
  const storeDetail = activeStore ? `${activeStore.type} • ${activeStore.location}` : "Multi-Store Control";

  return (
    <aside className="w-(--sidebar-w) bg-white border-r border-slate-200 flex flex-col h-screen fixed left-0 top-0 text-slate-700 font-serif z-20 transition-colors duration-200">
      {/* RetailAI Logo Section */}
      <div className="p-5 border-b border-sky-100/50 flex items-center gap-3 shrink-0">
        <div className="relative">
          <ShoppingCart className="w-7 h-7 text-slate-900" />
          <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-amber-500 rounded-full border border-white" />
        </div>
        <span className="font-extrabold text-slate-900 text-lg tracking-tight font-serif">
          RetailAI

        </span>
      </div>

      {/* Navigation menu list - Set to flex-1 to push the bottom store details card down */}
      <nav className="px-3 py-3 space-y-1 overflow-y-auto font-sans flex-1">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          return (
            <button
              id={`nav-item-${item.id}`}
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`w-full flex items-center gap-3.5 px-4 py-2.5 rounded-xl text-xs font-semibold tracking-wide transition-all group ${
                isActive
                  ? "bg-sky-50 text-blue-600 font-bold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-sky-50/20"
              }`}
            >
              <Icon
                className={`w-4.5 h-4.5 transition-all shrink-0 ${
                  isActive
                    ? "text-blue-600 scale-105"
                    : "text-slate-400 group-hover:text-slate-700"
                }`}
              />
              <span className="font-serif text-[13px]">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Active Store Account Details - Sits anchored at the bottom */}
      <div className="mx-4 mb-4 p-3 bg-slate-50/50 border border-sky-100 rounded-xl flex items-center gap-3 shrink-0">
        <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold text-xs shadow-sm font-sans shrink-0">
          {initials}
        </div>
        <div className="flex-1 min-w-0">
          <span className="block text-xs font-bold text-slate-900 truncate font-sans">
            {storeName}
          </span>
          <span className="block text-[9px] text-slate-400 font-bold uppercase tracking-wide truncate mt-0.5 font-sans">
            {storeDetail}
          </span>
        </div>
      </div>
    </aside>
  );
}
