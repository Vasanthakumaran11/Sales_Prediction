export function Card({ children, className = "" }) {
  return (
    <div
      className={`bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 shadow-sm ${className}`}
    >
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, icon: Icon, action }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 mb-4 border-b border-zinc-150 dark:border-zinc-800">
      <div>
        <h3 className="text-sm font-bold text-zinc-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
          {Icon && <Icon className="w-4 h-4 text-emerald-500" />}
          {title}
        </h3>
        {subtitle && <p className="text-[10px] text-zinc-500 dark:text-zinc-400 mt-0.5">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function PageHeader({ title, subtitle, icon: Icon, action }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-4">
      <div>
        <h2 className="text-xl font-bold text-zinc-900 dark:text-white tracking-tight flex items-center gap-2">
          {Icon && <Icon className="w-5 h-5 text-emerald-500" />}
          {title}
        </h2>
        {subtitle && <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}
