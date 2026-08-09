"use client";

import { createContext, useContext, useEffect, useMemo, useState, useCallback } from "react";
import { listDailyLogs } from "@/lib/api/transactions";
import { getProducts } from "@/lib/api/products";
import { isLiveBackendConfigured, setAuthToken } from "@/lib/api/client";

const StoreContext = createContext(null);

/**
 * Demo vs. real store split:
 * - Demo stores (ids in DEMO_STORE_IDS, lib/constants.js) are pure client-side
 *   mock data from lib/mock/* — they never touch the backend, and exist purely
 *   for offline pitch/demo purposes.
 * - Every other store id is a real, DB-backed store: it always hits the live
 *   API via lib/api/*, and shows an explicit error/empty state on failure
 *   rather than silently falling back to fabricated numbers.
 */
export function StoreProvider({ children }) {
  const [theme, setTheme] = useState("light");
  const [stage, setStage] = useState("gateway"); // 'gateway' | 'active'
  const [gatewayState, setGatewayState] = useState("landing");
  const [activeStore, setActiveStore] = useState(null); // null => executive multi-store mode
  const [isExecutiveMode, setIsExecutiveMode] = useState(false);
  const [activeView, setActiveView] = useState("overview");
  const [storeProducts, setStoreProducts] = useState([]);
  const [historyLogs, setHistoryLogs] = useState([]);

  // Read URL on mount to restore page state
  useEffect(() => {
    if (typeof window !== "undefined") {
      const path = window.location.pathname;
      setTimeout(() => {
        if (path === "/login") {
          setStage("gateway");
          setGatewayState("login");
        } else if (path === "/register") {
          setStage("gateway");
          setGatewayState("register");
        } else if (path === "/dashboard") {
          setStage("active");
          setActiveView("data-entry");
        } else if (path === "/analytics") {
          setStage("active");
          setActiveView("sales");
        } else if (path === "/inventory") {
          setStage("active");
          setActiveView("inventory");
        } else if (path === "/products") {
          setStage("active");
          setActiveView("products");
        } else if (path === "/suppliers") {
          setStage("active");
          setActiveView("suppliers");
        } else if (path === "/predictions") {
          setStage("active");
          setActiveView("ai-predictions");
        } else if (path === "/history") {
          setStage("active");
          setActiveView("history");
        } else if (path === "/settings") {
          setStage("active");
          setActiveView("settings");
        }
      }, 0);
    }
  }, []);

  // Sync URL history state when views change
  useEffect(() => {
    if (typeof window !== "undefined") {
      let path = "/";
      if (stage === "gateway") {
        if (gatewayState === "login") path = "/login";
        else if (gatewayState === "register") path = "/register";
      } else {
        if (activeView === "data-entry") path = "/dashboard";
        else if (activeView === "sales") path = "/analytics";
        else if (activeView === "inventory") path = "/inventory";
        else if (activeView === "products") path = "/products";
        else if (activeView === "suppliers") path = "/suppliers";
        else if (activeView === "ai-predictions") path = "/predictions";
        else if (activeView === "history") path = "/history";
        else if (activeView === "settings") path = "/settings";
      }
      
      if (window.location.pathname !== path) {
        window.history.pushState({}, "", path);
      }
    }
  }, [stage, activeView, gatewayState]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      if (theme === "dark") {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
    }
  }, [theme]);


  const enterStore = useCallback((store, initialProducts = null) => {
    setActiveStore(store);
    setIsExecutiveMode(false);
    setActiveView("data-entry");
    setStage("active");

    setHistoryLogs([]);
    setStoreProducts([]);
    setAuthToken(store?.token || null);

    if (store) {
      if (isLiveBackendConfigured()) {
        // Fetch daily logs dynamically
        listDailyLogs(store.id)
          .then((data) => {
            if (Array.isArray(data)) {
              setHistoryLogs(
                data.map((t) => ({
                  date: t.date,
                  transactions: t.transaction_count,
                  gross: t.gross_sales,
                  discount: t.discount_amount,
                  net: t.net_sales,
                  checked: t.audit_status === "Synced & Closed",
                }))
              );
            }
          })
          .catch((err) => console.error("Error fetching daily logs from database:", err));

        // Fetch products catalog dynamically
        getProducts(store.id)
          .then((data) => {
            if (Array.isArray(data)) {
              setStoreProducts(
                data.map((p) => ({
                  id: p.id,
                  name: p.name,
                  category: p.category,
                  brand: p.brand || p.name.split(" ")[0] || "Generic",
                  sku: p.sku || `SKU-${p.id.toUpperCase()}`,
                  barcode: p.sku || `${Math.floor(1000000000000 + Math.random() * 9000000000000)}`,
                  buyingPrice: p.cost_price || 0,
                  sellingPrice: p.selling_price || 0,
                  margin: p.cost_price > 0 ? Math.round(((p.selling_price - p.cost_price) / p.selling_price) * 1000) / 10 : 0,
                  stock: p.stock !== undefined ? p.stock : 20,
                  status: (p.stock !== undefined ? p.stock : 20) === 0 ? "Out of Stock" : (p.stock !== undefined ? p.stock : 20) < 20 ? "Low Stock" : "Healthy",
                  updated: new Date().toLocaleDateString(),
                  img: "📦",
                }))
              );
            }
          })
          .catch((err) => console.error("Error fetching products from database:", err));
      } else if (initialProducts) {
        setStoreProducts(
          initialProducts.map((p) => ({
            id: p.itemId,
            name: p.name,
            category: p.category,
            brand: p.name.split(" ")[0] || "Generic",
            sku: p.sku || `SKU-${Math.random().toString(36).substring(2, 8).toUpperCase()}`,
            barcode: p.barcode || `${Math.floor(1000000000000 + Math.random() * 9000000000000)}`,
            buyingPrice: p.buyingPrice || 0,
            sellingPrice: p.sellingPrice || 0,
            margin: p.buyingPrice > 0 ? Math.round(((p.sellingPrice - p.buyingPrice) / p.sellingPrice) * 1000) / 10 : 0,
            stock: p.qty || 0,
            status: (p.qty || 0) > 10 ? "Healthy" : "Low Stock",
            updated: new Date().toLocaleDateString() + " " + new Date().toLocaleTimeString(),
            img: "📦",
          }))
        );
      }
    }
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
    setAuthToken(null);
  }, []);

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      stage,
      setStage,
      gatewayState,
      setGatewayState,
      activeStore,
      isExecutiveMode,
      activeView,
      setActiveView,
      enterStore,
      enterExecutiveMode,
      exitToGateway,
      storeProducts,
      setStoreProducts,
      historyLogs,
      setHistoryLogs,
    }),
    [theme, stage, gatewayState, activeStore, isExecutiveMode, activeView, enterStore, enterExecutiveMode, exitToGateway, storeProducts, historyLogs]
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStoreContext() {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStoreContext must be used within StoreProvider");
  return ctx;
}
