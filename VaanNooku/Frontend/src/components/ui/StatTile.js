const DELTA_COLOR = {
  up: "text-emerald-600 dark:text-emerald-400",
  down: "text-rose-600 dark:text-rose-400",
};

export function StatTile({ label, value, icon: Icon, delta, deltaDirection = "up", hint }) {
  return (
    <div className="bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">{label}</span>
        {Icon && <Icon className="w-4 h-4 text-zinc-400 dark:text-zinc-600" />}
      </div>
      <div className="flex items-end justify-between gap-2">
        <span className="text-2xl font-black text-zinc-900 dark:text-white leading-none">{value}</span>
        {delta && <span className={`text-xs font-bold ${DELTA_COLOR[deltaDirection]}`}>{delta}</span>}
      </div>
      {hint && <span className="text-[10px] text-zinc-450 dark:text-zinc-500">{hint}</span>}
    </div>
  );
}
