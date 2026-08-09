import {
  LayoutDashboard,
  TrendingUp,
  BrainCircuit,
  Package,
  DollarSign,
  Sparkles,
  FileBarChart,
  FileClock,
  Settings as SettingsIcon,
} from "lucide-react";

export const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "demand", label: "Demand Forecasting Hub", icon: TrendingUp },
  { id: "models", label: "Model Intelligence", icon: BrainCircuit },
  { id: "inventory", label: "Scientific Replenishment", icon: Package },
  { id: "financial", label: "Financial ROI & Capital", icon: DollarSign },
  { id: "recommendations", label: "SKU Recommendations", icon: Sparkles },
  { id: "reports", label: "AI Business Reports", icon: FileBarChart },
  { id: "transactions", label: "Daily Transactions Log", icon: FileClock },
  { id: "settings", label: "Platform Settings", icon: SettingsIcon },
];

export const BRAND_NAME = "Smart Retail AI";
export const BRAND_TAGLINE = "Decision Intelligence Platform";

// Store IDs backed by lib/mock/* only — never hit the live backend, used purely
// for offline pitch/demo purposes. Any other store id is a real, DB-backed store.
export const DEMO_STORE_IDS = ["balaji-store", "shiva-stores", "surya-markets"];
