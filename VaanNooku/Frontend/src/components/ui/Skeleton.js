export function CardSkeleton({ className = "h-40" }) {
  return (
    <div
      className={`bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-900 rounded-2xl p-5 animate-pulse ${className}`}
    >
      <div className="h-4 w-36 bg-zinc-200 dark:bg-zinc-800 rounded mb-4" />
      <div className="h-full w-full bg-zinc-100 dark:bg-zinc-950/60 rounded-xl" />
    </div>
  );
}

export function PageSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-900 pb-4">
        <div className="space-y-2">
          <div className="h-6 w-56 bg-zinc-200 dark:bg-zinc-900 rounded" />
          <div className="h-3 w-80 bg-zinc-200 dark:bg-zinc-900 rounded" />
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <CardSkeleton className="lg:col-span-2 h-[320px]" />
        <div className="space-y-6">
          <CardSkeleton className="h-[148px]" />
          <CardSkeleton className="h-[148px]" />
        </div>
      </div>
    </div>
  );
}
