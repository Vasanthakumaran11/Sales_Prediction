"use client";

import React, { useEffect, useState, useRef } from "react";
import { RefreshCw, PlayCircle, CheckCircle2, XCircle, Clock } from "lucide-react";
import { PageHeader, Card, Badge } from "@/components/ui/Card";
import { listDatasets, triggerRetraining, getRetrainingStatus } from "@/lib/api/admin";

const STATUS_TONE = { queued: "amber", running: "blue", success: "emerald", failed: "rose" };

export default function RetrainingView() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [job, setJob] = useState(null);
  const [triggering, setTriggering] = useState(false);
  const [error, setError] = useState("");
  const pollRef = useRef(null);

  useEffect(() => {
    listDatasets().then(setDatasets).catch(() => {});
    return () => clearInterval(pollRef.current);
  }, []);

  const pollJob = (jobId) => {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const data = await getRetrainingStatus(jobId);
        setJob(data);
        if (data.status === "success" || data.status === "failed") {
          clearInterval(pollRef.current);
        }
      } catch {
        clearInterval(pollRef.current);
      }
    }, 3000);
  };

  const handleTrigger = async () => {
    setTriggering(true);
    setError("");
    try {
      const result = await triggerRetraining(selectedDataset || null);
      setJob(result);
      pollJob(result.jobId);
    } catch (err) {
      setError(err.message || "Failed to trigger retraining.");
    }
    setTriggering(false);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Retraining"
        subtitle="Re-runs the same 4-model ensemble pipeline (RF, XGBoost, LightGBM, CatBoost) on the selected dataset."
        icon={RefreshCw}
      />

      <Card className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-end">
          <div className="sm:col-span-2 space-y-1">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Training Dataset</label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-800 focus:outline-none"
            >
              <option value="">Default (retailai_finalized_dataset.csv)</option>
              {datasets.map((d) => (
                <option key={d.filename} value={d.filename}>{d.filename}</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleTrigger}
            disabled={triggering || job?.status === "running" || job?.status === "queued"}
            className="flex items-center justify-center gap-1.5 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-lg shadow-sm disabled:opacity-50"
          >
            <PlayCircle className="w-4 h-4" /> {triggering ? "Starting..." : "Trigger Retraining"}
          </button>
        </div>
        <p className="text-[10px] text-slate-400 leading-relaxed">
          Runs in the background. New models are only promoted to production if the new ensemble R² doesn't
          regress more than 0.01 below the currently deployed model — otherwise the run is reported as
          not-promoted and production is left untouched.
        </p>
        {error && <p className="text-xs text-rose-600 font-bold">{error}</p>}
      </Card>

      {job && (
        <Card className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Job {job.jobId?.slice(0, 8)}</h3>
            <Badge tone={STATUS_TONE[job.status] || "slate"}>{job.status}</Badge>
          </div>

          {job.status === "running" || job.status === "queued" ? (
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Clock className="w-4 h-4 animate-pulse" /> Training in progress — this can take a minute or two.
            </div>
          ) : job.status === "failed" ? (
            <div className="flex items-start gap-2 text-xs text-rose-600 font-semibold">
              <XCircle className="w-4 h-4 shrink-0 mt-0.5" /> {job.error}
            </div>
          ) : job.status === "success" && job.result ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-xs text-emerald-600 font-bold">
                <CheckCircle2 className="w-4 h-4" />
                {job.result.promoted ? "New models promoted to production." : "Training complete — not promoted."}
              </div>
              {!job.result.promoted && (
                <p className="text-[10px] text-slate-500">{job.result.reason}</p>
              )}
              <div className="grid grid-cols-3 gap-4 text-xs">
                <div>
                  <span className="block text-[9px] text-slate-400 uppercase font-bold">New Ensemble R²</span>
                  <span className="font-black text-slate-900">{job.result.ensembleMetrics.r2.toFixed(4)}</span>
                </div>
                <div>
                  <span className="block text-[9px] text-slate-400 uppercase font-bold">Previous R²</span>
                  <span className="font-black text-slate-900">{job.result.previousEnsembleR2.toFixed(4)}</span>
                </div>
                <div>
                  <span className="block text-[9px] text-slate-400 uppercase font-bold">Dataset Rows</span>
                  <span className="font-black text-slate-900">{job.result.datasetRows.toLocaleString()}</span>
                </div>
              </div>
            </div>
          ) : null}
        </Card>
      )}
    </div>
  );
}
