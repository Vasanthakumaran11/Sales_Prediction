// Status colors follow the dataviz skill's fixed status palette (good/warning/critical),
// reserved for state and never reused as chart series colors.
const STATUS_STYLES = {
  good: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/20",
  warning: "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/20",
  critical: "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/20",
  neutral: "bg-zinc-100 dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400 border-zinc-200 dark:border-zinc-800",
  info: "bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-500/20",
};

export function Badge({ children, status = "neutral", className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold border uppercase tracking-wider ${STATUS_STYLES[status]} ${className}`}
    >
      {children}
    </span>
  );
}

export function riskToStatus(risk) {
  if (risk === "High") return "critical";
  if (risk === "Medium") return "warning";
  return "good";
}
