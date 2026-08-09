"use client";

import React, { useEffect, useState } from "react";
import { BrainCircuit, RefreshCw, TrendingUp, Target, Activity } from "lucide-react";
import { PageHeader, Card, StatTile, Badge } from "@/components/ui/Card";
import { getModelStatus, reloadModels } from "@/lib/api/admin";

export default function ModelStatusView() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [reloading, setReloading] = useState(false);
  const [reloadMsg, setReloadMsg] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await getModelStatus();
      setStatus(data);
    } catch (err) {
      setError(err.message || "Failed to load model status.");
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const handleReload = async () => {
    setReloading(true);
    setReloadMsg("");
    try {
      await reloadModels();
      setReloadMsg("Models reloaded from disk successfully.");
      await load();
    } catch (err) {
      setReloadMsg(err.message || "Reload failed.");
    }
    setReloading(false);
  };

  const ensemble = status?.ensemble;
  const comparison = status?.modelComparison || [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Intelligence"
        subtitle="Live status of the production demand-forecasting ensemble."
        icon={BrainCircuit}
        action={
          <button
            onClick={handleReload}
            disabled={reloading}
            className="flex items-center gap-1.5 px-3 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold text-xs rounded-lg shadow-sm disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${reloading ? "animate-spin" : ""}`} /> Reload Models
          </button>
        }
      />

      {reloadMsg && (
        <div className="p-3 bg-blue-50 border border-blue-100 rounded-xl text-blue-700 text-xs font-bold">{reloadMsg}</div>
      )}

      {loading ? (
        <Card className="p-12 text-center text-slate-400 text-xs">Loading model status...</Card>
      ) : error ? (
        <Card className="p-8 text-center space-y-2 border-rose-100 bg-rose-50/30">
          <p className="text-sm font-bold text-rose-700">{error}</p>
          <p className="text-xs text-slate-500">Ensure the backend is running and you're logged in.</p>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <StatTile label="Ensemble R²" value={ensemble ? ensemble.r2.toFixed(4) : "—"} icon={Target} tone="blue" />
            <StatTile label="Mean Absolute Error" value={ensemble ? ensemble.mae.toFixed(2) : "—"} icon={Activity} tone="emerald" />
            <StatTile label="RMSE" value={ensemble ? ensemble.rmse.toFixed(2) : "—"} icon={TrendingUp} tone="amber" />
          </div>

          <Card className="space-y-4">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Per-Model Comparison</h3>
            {comparison.length === 0 ? (
              <p className="text-xs text-slate-400">No comparison data found — run a training pass first.</p>
            ) : (
              <div className="overflow-x-auto border border-slate-100 rounded-xl">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-100 text-[9px] font-bold text-slate-400 uppercase tracking-widest">
                      <th className="py-2.5 px-4">Model</th>
                      <th className="py-2.5 px-4 text-right">R²</th>
                      <th className="py-2.5 px-4 text-right">MAE</th>
                      <th className="py-2.5 px-4 text-right">RMSE</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-50">
                    {comparison.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-50/50">
                        <td className="py-2.5 px-4 font-bold text-slate-800">
                          {row.model}
                          {row.model === "Hybrid Ensemble" && <Badge tone="blue">production</Badge>}
                        </td>
                        <td className="py-2.5 px-4 text-right font-semibold text-slate-700">{parseFloat(row.r2).toFixed(4)}</td>
                        <td className="py-2.5 px-4 text-right text-slate-600">{parseFloat(row.mae).toFixed(2)}</td>
                        <td className="py-2.5 px-4 text-right text-slate-600">{parseFloat(row.rmse).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {status?.lastUpdated && (
              <p className="text-[10px] text-slate-400">Last updated: {new Date(status.lastUpdated).toLocaleString()}</p>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
