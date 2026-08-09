"use client";

import React from "react";
import { BrainCircuit, Database, RefreshCw, MessageSquareWarning, Settings as SettingsIcon, ShieldCheck, LogOut } from "lucide-react";
import { useAdminContext } from "@/context/AdminContext";

const NAV_ITEMS = [
  { id: "models", label: "Model Intelligence", icon: BrainCircuit },
  { id: "datasets", label: "Datasets", icon: Database },
  { id: "retraining", label: "Retraining", icon: RefreshCw },
  { id: "complaints", label: "Complaints", icon: MessageSquareWarning },
  { id: "settings", label: "Settings", icon: SettingsIcon },
];

export function Sidebar() {
  const { activeView, setActiveView, admin, logout } = useAdminContext();

  return (
    <aside className="w-[var(--sidebar-w)] bg-white border-r border-slate-200 flex flex-col h-screen fixed left-0 top-0 z-20">
      <div className="p-5 border-b border-slate-100 flex items-center gap-3 shrink-0">
        <div className="w-9 h-9 rounded-xl bg-blue-600 text-white flex items-center justify-center shrink-0">
          <ShieldCheck className="w-5 h-5" />
        </div>
        <div>
          <span className="font-extrabold text-slate-900 text-sm tracking-tight block font-serif">VaanNooku Admin</span>
          <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">AI Operations Console</span>
        </div>
      </div>

      <nav className="px-3 py-3 space-y-1 flex-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-xs font-semibold transition-all ${
                isActive ? "bg-blue-50 text-blue-600" : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              }`}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {item.label}
            </button>
          );
        })}
      </nav>

      <div className="mx-3 mb-3 p-3 bg-slate-50 border border-slate-100 rounded-xl flex items-center justify-between gap-2">
        <div className="min-w-0">
          <span className="block text-[11px] font-bold text-slate-800 truncate">{admin?.email || "Admin"}</span>
          <span className="block text-[9px] text-slate-400 uppercase tracking-wide">{admin?.role || "admin"}</span>
        </div>
        <button onClick={logout} className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors shrink-0">
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </aside>
  );
}
