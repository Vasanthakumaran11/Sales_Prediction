"use client";

import React, { useEffect, useState } from "react";
import { MessageSquareWarning } from "lucide-react";
import { PageHeader, Card, Badge } from "@/components/ui/Card";
import { listComplaints, updateComplaintStatus } from "@/lib/api/admin";

const STATUS_OPTIONS = ["Open", "In Progress", "Resolved"];
const STATUS_TONE = { Open: "rose", "In Progress": "amber", Resolved: "emerald" };

export default function ComplaintsView() {
  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      setTickets(await listComplaints());
    } catch (err) {
      setError(err.message || "Failed to load complaints.");
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const handleStatusChange = async (ticketId, status) => {
    setTickets((prev) => prev.map((t) => (t.id === ticketId ? { ...t, status } : t)));
    try {
      await updateComplaintStatus(ticketId, status);
    } catch {
      load(); // revert to server truth on failure
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Store Complaints"
        subtitle="Issues flagged by store owners — bad predictions, sync problems, or bugs."
        icon={MessageSquareWarning}
      />

      <Card className="space-y-4">
        {loading ? (
          <p className="text-xs text-slate-400">Loading...</p>
        ) : error ? (
          <p className="text-xs text-rose-600 font-bold">{error}</p>
        ) : tickets.length === 0 ? (
          <p className="text-xs text-slate-400">No complaints filed yet.</p>
        ) : (
          <div className="divide-y divide-slate-50">
            {tickets.map((t) => (
              <div key={t.id} className="py-3 space-y-1.5">
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <span className="font-bold text-slate-900 text-sm block truncate">{t.subject}</span>
                    <span className="text-[10px] text-slate-400">Store: {t.store_id} • {new Date(t.created_at).toLocaleString()}</span>
                  </div>
                  <select
                    value={t.status}
                    onChange={(e) => handleStatusChange(t.id, e.target.value)}
                    className="bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-[10px] font-bold uppercase focus:outline-none shrink-0"
                  >
                    {STATUS_OPTIONS.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
                {t.description && <p className="text-xs text-slate-600">{t.description}</p>}
                <Badge tone={STATUS_TONE[t.status] || "slate"}>{t.status}</Badge>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
