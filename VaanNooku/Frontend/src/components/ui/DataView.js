"use client";

import { PageSkeleton } from "./Skeleton";
import { ErrorState } from "./EmptyState";

// Shared loading/error/data switch driven by useAsync(), so every page
// follows the same seam that will carry real backend latency and errors.
export function DataView({ isLoading, error, data, reload, children }) {
  if (isLoading && !data) return <PageSkeleton />;
  if (error) return <ErrorState message={error.message} onRetry={reload} />;
  if (!data) return null;
  return children(data);
}
