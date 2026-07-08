const FILL_STYLES = {
  good: "bg-emerald-500",
  warning: "bg-amber-500",
  critical: "bg-rose-500",
  info: "bg-blue-500",
};

export function ProgressBar({ percent, status = "good", className = "" }) {
  const clamped = Math.max(0, Math.min(100, percent));
  return (
    <div className={`w-full h-1.5 bg-zinc-100 dark:bg-zinc-900 rounded-full overflow-hidden ${className}`}>
      <div
        className={`h-full rounded-full transition-all duration-500 ${FILL_STYLES[status]}`}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}
