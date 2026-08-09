"use client";

import React, { useEffect, useState, useRef } from "react";
import { Database, Upload, FileSpreadsheet } from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";
import { listDatasets, uploadDataset } from "@/lib/api/admin";

export default function DatasetsView() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const fileInputRef = useRef(null);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      setDatasets(await listDatasets());
    } catch (err) {
      setError(err.message || "Failed to load datasets.");
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg("");
    try {
      const result = await uploadDataset(file);
      setUploadMsg(`Uploaded ${result.filename} (${result.rows} rows).`);
      await load();
    } catch (err) {
      setUploadMsg(err.message || "Upload failed.");
    }
    setUploading(false);
    e.target.value = "";
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Datasets"
        subtitle="Training data available to the retraining pipeline."
        icon={Database}
        action={
          <>
            <input ref={fileInputRef} type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="flex items-center gap-1.5 px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-lg shadow-sm disabled:opacity-50"
            >
              <Upload className="w-3.5 h-3.5" /> {uploading ? "Uploading..." : "Upload Dataset"}
            </button>
          </>
        }
      />

      {uploadMsg && (
        <div className="p-3 bg-blue-50 border border-blue-100 rounded-xl text-blue-700 text-xs font-bold">{uploadMsg}</div>
      )}

      <Card className="space-y-4">
        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Available Files</h3>
        {loading ? (
          <p className="text-xs text-slate-400">Loading...</p>
        ) : error ? (
          <p className="text-xs text-rose-600 font-bold">{error}</p>
        ) : datasets.length === 0 ? (
          <p className="text-xs text-slate-400">No datasets found under ml_workspace/datasets.</p>
        ) : (
          <div className="divide-y divide-slate-50">
            {datasets.map((d) => (
              <div key={d.filename} className="flex items-center justify-between py-2.5 text-xs">
                <div className="flex items-center gap-2.5 min-w-0">
                  <FileSpreadsheet className="w-4 h-4 text-slate-400 shrink-0" />
                  <span className="font-semibold text-slate-800 truncate">{d.filename}</span>
                </div>
                <div className="flex items-center gap-4 text-slate-400 shrink-0">
                  <span>{(d.sizeBytes / 1024).toFixed(1)} KB</span>
                  <span>{new Date(d.modifiedAt).toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
