"use client";

import { useStoreContext } from "@/context/StoreContext";
import { Gateway } from "@/components/gateway/Gateway";
import { Sidebar } from "@/components/layout/Sidebar";
import InventoryReplenishment from "@/components/inventory/InventoryReplenishment";
import SalesAnalytics from "@/components/analytics/SalesAnalytics";
import Settings from "@/components/settings/Settings";
import ProductsView from "@/components/products/ProductsView";
import SuppliersView from "@/components/suppliers/SuppliersView";
import AlertsView from "@/components/alerts/AlertsView";
import DataEntryView from "@/components/dashboard/DataEntryView";
import AIPredictions from "@/components/predictions/AIPredictions";
import HistoryView from "@/components/history/HistoryView";

export default function Home() {
  const { stage, activeView } = useStoreContext();

  if (stage === "gateway") {
    return <Gateway />;
  }

  const renderView = () => {
    switch (activeView) {
      case "data-entry":
        return <DataEntryView />;
      case "sales":
        return <SalesAnalytics />;
      case "inventory":
        return <InventoryReplenishment />;
      case "products":
        return <ProductsView />;
      case "suppliers":
        return <SuppliersView />;
      case "alerts":
        return <AlertsView />;
      case "ai-predictions":
        return <AIPredictions />;
      case "history":
        return <HistoryView />;
      case "settings":
        return <Settings />;
      default:
        return <DataEntryView />;
    }
  };




  return (
    <div className="flex h-screen overflow-hidden bg-sky-50 font-sans">
      <Sidebar />
      <main className="flex-1 overflow-y-auto pl-56 p-6 transition-colors duration-205">
        <div className="w-full space-y-4">
          {renderView()}
        </div>
      </main>
    </div>
  );
}

