"use client";

import React from "react";
import { Settings as SettingsIcon } from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";
import { useAdminContext } from "@/context/AdminContext";

export default function SettingsView() {
  const { admin } = useAdminContext();

  return (
    <div className="space-y-6">
      <PageHeader title="Settings" icon={SettingsIcon} />
      <Card className="space-y-3">
        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Account</h3>
        <div className="text-xs text-slate-600 space-y-1">
          <p><span className="text-slate-400">Email:</span> {admin?.email || "—"}</p>
          <p><span className="text-slate-400">Role:</span> {admin?.role || "—"}</p>
        </div>
        <p className="text-[10px] text-slate-400 leading-relaxed pt-2 border-t border-slate-100">
          Additional admin accounts are created via <code className="bg-slate-50 px-1 rounded">backend/seed_admin.py</code>.
          There is no self-serve admin signup by design — this console is for internal ops/ML staff only.
        </p>
      </Card>
    </div>
  );
}
