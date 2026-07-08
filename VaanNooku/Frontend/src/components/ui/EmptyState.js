export function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="text-center py-10 text-zinc-500 dark:text-zinc-400">
      {Icon && <Icon className="w-8 h-8 text-zinc-300 dark:text-zinc-700 mx-auto mb-2" />}
      <span className="block text-xs font-semibold">{title}</span>
      {description && <span className="block text-[11px] text-zinc-400 dark:text-zinc-600 mt-1">{description}</span>}
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="text-center py-10 space-y-3">
      <span className="block text-xs font-semibold text-rose-600 dark:text-rose-400">
        {message || "Something went wrong while loading this data."}
      </span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 hover:text-emerald-500"
        >
          Try again
        </button>
      )}
    </div>
  );
}
