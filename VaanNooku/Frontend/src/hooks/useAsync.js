"use client";

import { useEffect, useRef, useState, useCallback } from "react";

/**
 * Runs an async loader and exposes { data, isLoading, error, reload }.
 * `deps` re-triggers the load, mirroring how a real fetch-on-navigation
 * flow would behave once lib/api/* calls a live backend.
 */
export function useAsync(loader, deps = []) {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const requestId = useRef(0);

  const run = useCallback(() => {
    const id = ++requestId.current;
    setIsLoading(true);
    setError(null);
    loader()
      .then((result) => {
        if (id === requestId.current) {
          setData(result);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (id === requestId.current) {
          setError(err);
          setIsLoading(false);
        }
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    run();
  }, [run]);

  return { data, isLoading, error, reload: run };
}
