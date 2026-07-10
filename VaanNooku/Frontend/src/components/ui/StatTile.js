import React from "react";

const DELTA_COLOR = {
  up: "text-emerald-600",
  down: "text-rose-600",
};

export function StatTile({ label, value, icon: Icon, delta, deltaDirection = "up", hint }) {
  return (
    <div className="bg-white border border-sky-100 rounded-2xl p-4 flex flex-col gap-2 shadow-sm">
      <div className="flex items-center justify-between font-sans">
        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{label}</span>
        {Icon && <Icon className="w-4 h-4 text-sky-500" />}
      </div>
      <div className="flex items-end justify-between gap-2">
        <span className="text-2xl font-black text-slate-900 leading-none font-sans tabular-nums">{value}</span>
        {delta && <span className={`text-xs font-bold ${DELTA_COLOR[deltaDirection]}`}>{delta}</span>}
      </div>
      {hint && <span className="text-[10px] text-slate-400 font-sans">{hint}</span>}
    </div>
  );
}
