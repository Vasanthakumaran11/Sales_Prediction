"use client";

import React, { useState } from "react";
import { ShieldCheck, ArrowRight } from "lucide-react";
import { useAdminContext } from "@/context/AdminContext";
import { loginAdmin } from "@/lib/api/admin";
import { isLiveBackendConfigured } from "@/lib/api/client";

export function AdminLogin() {
  const { login } = useAdminContext();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (!isLiveBackendConfigured()) {
      setError("Backend endpoint not configured (NEXT_PUBLIC_API_BASE_URL).");
      return;
    }
    setSubmitting(true);
    try {
      const response = await loginAdmin(email, password);
      login(response.admin, response.token);
    } catch (err) {
      setError(err.message || "Login failed.");
    }
    setSubmitting(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 px-4">
      <form onSubmit={handleSubmit} className="w-full max-w-sm bg-white border border-slate-200 rounded-2xl shadow-xl p-8 space-y-6">
        <div className="flex flex-col items-center text-center gap-3">
          <div className="w-12 h-12 rounded-2xl bg-blue-600 text-white flex items-center justify-center">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-900 font-serif">VaanNooku Admin</h1>
            <p className="text-xs text-slate-500 mt-0.5">AI Operations Console — staff access only</p>
          </div>
        </div>

        {error && (
          <div className="p-3 bg-rose-50 border border-rose-100 rounded-xl text-rose-600 text-xs font-bold">
            {error}
          </div>
        )}

        <div className="space-y-3">
          <div className="space-y-1">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-800 focus:outline-none focus:border-blue-500"
              placeholder="admin@vaannooku.com"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-800 focus:outline-none focus:border-blue-500"
              placeholder="••••••••"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={submitting}
          className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-bold text-sm rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {submitting ? "Signing in..." : "Sign In"} <ArrowRight className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
