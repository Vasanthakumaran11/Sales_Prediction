"use client";

import { useAdminContext } from "@/context/AdminContext";
import { AdminLogin } from "@/components/auth/AdminLogin";
import { Sidebar } from "@/components/layout/Sidebar";
import ModelStatusView from "@/components/models/ModelStatusView";
import DatasetsView from "@/components/datasets/DatasetsView";
import RetrainingView from "@/components/retraining/RetrainingView";
import ComplaintsView from "@/components/complaints/ComplaintsView";
import SettingsView from "@/components/settings/SettingsView";

export default function Home() {
  const { admin, activeView, checkedSession } = useAdminContext();

  if (!checkedSession) return null;
  if (!admin) return <AdminLogin />;

  const renderView = () => {
    switch (activeView) {
      case "datasets": return <DatasetsView />;
      case "retraining": return <RetrainingView />;
      case "complaints": return <ComplaintsView />;
      case "settings": return <SettingsView />;
      case "models":
      default:
        return <ModelStatusView />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <Sidebar />
      <main className="flex-1 overflow-y-auto pl-[var(--sidebar-w)] p-6">
        <div className="w-full max-w-6xl mx-auto">{renderView()}</div>
      </main>
    </div>
  );
}
