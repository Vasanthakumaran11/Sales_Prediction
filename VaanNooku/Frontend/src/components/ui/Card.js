import React from "react";

export function Card({ children, className = "" }) {
  return (
    <div
      className={`bg-white border border-sky-100 rounded-2xl p-5 shadow-sm ${className}`}
    >
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, icon: Icon, action }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 mb-4 border-b border-slate-100">
      <div>
        <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 font-serif">
          {Icon && <Icon className="w-4 h-4 text-sky-600" />}
          {title}
        </h3>
        {subtitle && <p className="text-[10px] text-slate-500 mt-0.5 font-sans">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function PageHeader({ title, subtitle, icon: Icon, action }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-sky-200/60 pb-4">
      <div>
        <h2 className="text-xl font-bold text-slate-950 tracking-tight flex items-center gap-2 font-serif">
          {Icon && <Icon className="w-5 h-5 text-sky-600" />}
          {title}
        </h2>
        {subtitle && <p className="text-xs text-slate-500 mt-0.5 font-sans">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}
