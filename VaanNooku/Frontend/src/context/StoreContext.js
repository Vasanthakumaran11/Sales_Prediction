"use client";

import { createContext, useContext, useEffect, useMemo, useState, useCallback } from "react";

const StoreContext = createContext(null);

export function StoreProvider({ children }) {
  const [theme, setTheme] = useState("light");
  const [stage, setStage] = useState("gateway"); // 'gateway' | 'active'
  const [activeStore, setActiveStore] = useState(null); // null => executive multi-store mode
  const [isExecutiveMode, setIsExecutiveMode] = useState(false);
  const [activeView, setActiveView] = useState("overview");

  useEffect(() => {
    if (typeof window !== "undefined") {
      if (theme === "dark") {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
    }
  }, [theme]);


  const enterStore = useCallback((store) => {
    setActiveStore(store);
    setIsExecutiveMode(false);
    setActiveView("data-entry");
    setStage("active");
  }, []);

  const enterExecutiveMode = useCallback(() => {
    setActiveStore(null);
    setIsExecutiveMode(true);
    setActiveView("data-entry");
    setStage("active");
  }, []);

  const exitToGateway = useCallback(() => {
    setStage("gateway");
    setActiveStore(null);
    setIsExecutiveMode(false);
  }, []);

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      stage,
      activeStore,
      isExecutiveMode,
      activeView,
      setActiveView,
      enterStore,
      enterExecutiveMode,
      exitToGateway,
    }),
    [theme, stage, activeStore, isExecutiveMode, activeView, enterStore, enterExecutiveMode, exitToGateway]
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStoreContext() {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStoreContext must be used within StoreProvider");
  return ctx;
}
