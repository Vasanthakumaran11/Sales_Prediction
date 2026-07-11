"use client";

import { createContext, useContext, useEffect, useMemo, useState, useCallback } from "react";

const StoreContext = createContext(null);

const DEMO_PRODUCTS = [
  {
    name: "Amul Taaza Milk 1L",
    category: "Dairy & Bakery",
    brand: "Amul",
    sku: "AMUL-MILK-1L",
    barcode: "8901262150001",
    buyingPrice: 56.0,
    sellingPrice: 68.0,
    margin: 17.6,
    stock: 245,
    status: "Healthy",
    updated: "May 17, 2026 10:30 AM",
    img: "🥛",
  },
  {
    name: "India Gate Basmati Rice 1kg",
    category: "Staples & Grains",
    brand: "India Gate",
    sku: "IG-RICE-1KG",
    barcode: "8901122334455",
    buyingPrice: 82.0,
    sellingPrice: 105.0,
    margin: 21.9,
    stock: 120,
    status: "Low Stock",
    updated: "May 17, 2026 09:45 AM",
    img: "🌾",
  },
  {
    name: "Fortune Sunflower Oil 1L",
    category: "Staples & Grains",
    brand: "Fortune",
    sku: "FORT-OIL-1L",
    barcode: "8901030740012",
    buyingPrice: 135.0,
    sellingPrice: 160.0,
    margin: 15.6,
    stock: 35,
    status: "Low Stock",
    updated: "May 17, 2026 1 Hour AM",
    img: "🌻",
  },
  {
    name: "Tata Tea Premium 250g",
    category: "Beverages",
    brand: "Tata Tea",
    sku: "TATA-TEA-250",
    barcode: "8901030712345",
    buyingPrice: 120.0,
    sellingPrice: 150.0,
    margin: 20.0,
    stock: 0,
    status: "Out of Stock",
    updated: "May 17, 2026 08:50 AM",
    img: "☕",
  },
  {
    name: "Aashirvaad Atta 5kg",
    category: "Staples & Grains",
    brand: "Aashirvaad",
    sku: "AASH-ATTA-5KG",
    barcode: "8901122305678",
    buyingPrice: 245.0,
    sellingPrice: 280.0,
    margin: 12.5,
    stock: 60,
    status: "Low Stock",
    updated: "May 16, 2026 07:20 PM",
    img: "🥡",
  },
];

export function StoreProvider({ children }) {
  const [theme, setTheme] = useState("light");
  const [stage, setStage] = useState("gateway"); // 'gateway' | 'active'
  const [gatewayState, setGatewayState] = useState("landing");
  const [activeStore, setActiveStore] = useState(null); // null => executive multi-store mode
  const [isExecutiveMode, setIsExecutiveMode] = useState(false);
  const [activeView, setActiveView] = useState("overview");
  const [storeProducts, setStoreProducts] = useState(DEMO_PRODUCTS);
  const [historyLogs, setHistoryLogs] = useState([
    { date: "May 17, 2026", transactions: 1320, gross: 24567.70, discount: 1122.20, net: 23445.50, checked: true },
    { date: "May 16, 2026", transactions: 1285, gross: 22890.00, discount: 890.00, net: 22000.00, checked: true },
    { date: "May 15, 2026", transactions: 1410, gross: 27120.00, discount: 1540.00, net: 25580.00, checked: false },
    { date: "May 14, 2026", transactions: 1150, gross: 19850.50, discount: 450.50, net: 19400.00, checked: false },
    { date: "May 13, 2026", transactions: 1290, gross: 23410.00, discount: 1010.00, net: 22400.00, checked: false },
    { date: "May 12, 2026", transactions: 1380, gross: 26180.00, discount: 1200.00, net: 24980.00, checked: false },
    { date: "May 11, 2026", transactions: 1210, gross: 21050.00, discount: 850.00, net: 20200.00, checked: false },
  ]);

  // Read URL on mount to restore page state
  useEffect(() => {
    if (typeof window !== "undefined") {
      const path = window.location.pathname;
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

    const demoIds = ["balaji-store", "shiva-stores", "surya-markets"];
    if (store && !demoIds.includes(store.id)) {
      setHistoryLogs([]);
      setStoreProducts([]);

      const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || "";
      if (apiBase) {
        // Fetch daily logs dynamically
        fetch(`${apiBase}/api/stores/${store.id}/daily-logs`)
          .then((res) => res.json())
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
        fetch(`${apiBase}/api/stores/${store.id}/products`)
          .then((res) => res.json())
          .then((data) => {
            if (Array.isArray(data)) {
              setStoreProducts(
                data.map((p) => ({
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
    } else {
      setStoreProducts(DEMO_PRODUCTS);
      setHistoryLogs([
        { date: "May 17, 2026", transactions: 1320, gross: 24567.70, discount: 1122.20, net: 23445.50, checked: true },
        { date: "May 16, 2026", transactions: 1285, gross: 22890.00, discount: 890.00, net: 22000.00, checked: true },
        { date: "May 15, 2026", transactions: 1410, gross: 27120.00, discount: 1540.00, net: 25580.00, checked: false },
        { date: "May 14, 2026", transactions: 1150, gross: 19850.50, discount: 450.50, net: 19400.00, checked: false },
        { date: "May 13, 2026", transactions: 1290, gross: 23410.00, discount: 1010.00, net: 22400.00, checked: false },
        { date: "May 12, 2026", transactions: 1380, gross: 26180.00, discount: 1200.00, net: 24980.00, checked: false },
        { date: "May 11, 2026", transactions: 1210, gross: 21050.00, discount: 850.00, net: 20200.00, checked: false },
      ]);
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
