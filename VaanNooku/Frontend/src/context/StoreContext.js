"use client";

import { createContext, useContext, useEffect, useMemo, useState, useCallback } from "react";

const StoreContext = createContext(null);

export function StoreProvider({ children }) {
  const [theme, setTheme] = useState("light");
  const [stage, setStage] = useState("gateway"); // 'gateway' | 'active'
  const [activeStore, setActiveStore] = useState(null); // null => executive multi-store mode
  const [isExecutiveMode, setIsExecutiveMode] = useState(false);
  const [activeView, setActiveView] = useState("overview");
  const [historyLogs, setHistoryLogs] = useState([
    { date: "May 17, 2026", transactions: 1320, gross: 24567.70, discount: 1122.20, net: 23445.50, checked: true },
    { date: "May 16, 2026", transactions: 1285, gross: 22890.00, discount: 890.00, net: 22000.00, checked: true },
    { date: "May 15, 2026", transactions: 1410, gross: 27120.00, discount: 1540.00, net: 25580.00, checked: false },
    { date: "May 14, 2026", transactions: 1150, gross: 19850.50, discount: 450.50, net: 19400.00, checked: false },
    { date: "May 13, 2026", transactions: 1290, gross: 23410.00, discount: 1010.00, net: 22400.00, checked: false },
    { date: "May 12, 2026", transactions: 1380, gross: 26180.00, discount: 1200.00, net: 24980.00, checked: false },
    { date: "May 11, 2026", transactions: 1210, gross: 21050.00, discount: 850.00, net: 20200.00, checked: false },
  ]);

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
      historyLogs,
      setHistoryLogs,
    }),
    [theme, stage, activeStore, isExecutiveMode, activeView, enterStore, enterExecutiveMode, exitToGateway, historyLogs]
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStoreContext() {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStoreContext must be used within StoreProvider");
  return ctx;
}
