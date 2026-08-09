import React from "react";

export function Card({ children, className = "" }) {
  return (
    <div className={`bg-white border border-slate-200 rounded-2xl p-5 shadow-sm ${className}`}>
      {children}
    </div>
  );
}

export function PageHeader({ title, subtitle, icon: Icon, action }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-200 pb-4">
      <div>
        <h2 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2 font-serif">
          {Icon && <Icon className="w-5 h-5 text-blue-600" />}
          {title}
        </h2>
        {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function StatTile({ label, value, icon: Icon, tone = "slate" }) {
  const toneClasses = {
    slate: "bg-slate-50 text-slate-600",
    blue: "bg-blue-50 text-blue-600",
    emerald: "bg-emerald-50 text-emerald-600",
    amber: "bg-amber-50 text-amber-600",
    rose: "bg-rose-50 text-rose-600",
  }[tone] || "bg-slate-50 text-slate-600";

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 flex items-center gap-3.5 shadow-sm">
      <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${toneClasses}`}>
        {Icon && <Icon className="w-5 h-5" />}
      </div>
      <div className="min-w-0">
        <span className="block text-[9px] text-slate-400 uppercase font-bold tracking-wider">{label}</span>
        <span className="text-base font-black text-slate-900 leading-none block mt-1 truncate">{value}</span>
      </div>
    </div>
  );
}

export function Badge({ children, tone = "slate" }) {
  const toneClasses = {
    slate: "bg-slate-100 text-slate-600 border-slate-200",
    blue: "bg-blue-50 text-blue-700 border-blue-100",
    emerald: "bg-emerald-50 text-emerald-700 border-emerald-100",
    amber: "bg-amber-50 text-amber-700 border-amber-100",
    rose: "bg-rose-50 text-rose-700 border-rose-100",
  }[tone] || "bg-slate-100 text-slate-600 border-slate-200";

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider border ${toneClasses}`}>
      {children}
    </span>
  );
}
