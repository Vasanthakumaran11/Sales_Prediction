"use client";

import React, { useState } from "react";
import {
  User,
  Building,
  Users,
  ShieldAlert,
  Bell,
  Mail,
  Lock,
  Database,
  Link as LinkIcon,
  Sliders,
  CreditCard,
  History,
  Edit2,
  RefreshCw,
  Download,
  AlertTriangle,
  Trash2,
  CheckCircle,
} from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";

export default function Settings() {
  const [activeTab, setActiveTab] = useState("profile");
  
  // Form Preference states
  const [prefs, setPrefs] = useState({
    darkMode: true,
    compactView: true,
    showTips: true,
    autoRefresh: false,
    language: "English (India)",
    dateFormat: "DD MMM YYYY",
    timezone: "(GMT+05:30) Asia/Kolkata",
    currency: "INR (₹)"
  });

  const [toastMsg, setToastMsg] = useState("");

  const triggerToast = (msg) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(""), 3000);
  };

  const togglePref = (key) => {
    setPrefs((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSelectChange = (key, value) => {
    setPrefs((prev) => ({ ...prev, [key]: value }));
  };

  // Sidebar settings tabs
  const tabs = [
    { id: "profile", label: "Profile & Account", icon: User },
    { id: "business", label: "Business Information", icon: Building },
    { id: "users", label: "User Management", icon: Users },
    { id: "roles", label: "Roles & Permissions", icon: ShieldAlert },
    { id: "notifications", label: "Notification Preferences", icon: Bell },
    { id: "email", label: "Email Settings", icon: Mail },
    { id: "security", label: "Security", icon: Lock },
    { id: "backup", label: "Data & Backup", icon: Database },
    { id: "integrations", label: "Integrations", icon: LinkIcon },
    { id: "system", label: "System Preferences", icon: Sliders },
    { id: "billing", label: "Billing & Subscription", icon: CreditCard },
    { id: "logs", label: "Activity Logs", icon: History },
  ];

  return (
    <div className="space-y-6 font-sans">
      <PageHeader
        title="Settings"
        subtitle="Manage your account preferences, business configurations, and application settings."
        icon={Sliders}
      />

      {toastMsg && (
        <div className="fixed bottom-5 right-5 p-4 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center gap-2 text-xs text-emerald-800 shadow-md z-50 animate-fade-in">
          <CheckCircle className="w-4 h-4 text-emerald-500" />
          <span>{toastMsg}</span>
        </div>
      )}

      {/* Main Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
        {/* Left Side Tab Navigation */}
        <div className="col-span-1 bg-white border border-sky-100 rounded-2xl p-2.5 space-y-1 shadow-sm">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isSelected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-semibold tracking-wide transition-all ${
                  isSelected
                    ? "bg-sky-50 text-blue-600 font-bold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-sky-50/20"
                }`}
              >
                <Icon className={`w-4 h-4 ${isSelected ? "text-blue-600" : "text-slate-400"}`} />
                <span className="font-serif">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Right Side Settings Dashboard Container */}
        <div className="col-span-1 lg:col-span-3 space-y-6">
          {/* Active Section: Profile & Account */}
          {activeTab === "profile" ? (
            <div className="space-y-6">
              {/* Profile Information */}
              <Card className="p-5 flex flex-col justify-between gap-4">
                <div className="flex justify-between items-center pb-2 border-b border-slate-100">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 font-serif">Profile Information</h3>
                    <p className="text-[10px] text-slate-500">Update your personal information and profile details.</p>
                  </div>
                  <button
                    onClick={() => triggerToast("Profile edit wizard loaded.")}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold rounded-lg shadow-sm font-sans"
                  >
                    <Edit2 className="w-3.5 h-3.5 text-slate-400" /> Edit Profile
                  </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-xs font-sans mt-1">
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Full Name</span>
                    <span className="font-bold text-slate-800">Arjun Sharma</span>
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Email Address</span>
                    <span className="font-semibold text-slate-800">arjun.sharma@retailai.com</span>
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Phone Number</span>
                    <span className="font-semibold text-slate-800">+91 98765 43210</span>
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Role</span>
                    <span className="px-2.5 py-0.5 rounded bg-blue-50 text-blue-600 text-[10px] font-bold mt-0.5 inline-block uppercase">
                      Administrator
                    </span>
                  </div>
                </div>
              </Card>

              {/* Change Password */}
              <Card className="p-5 flex flex-col justify-between gap-4">
                <div className="flex justify-between items-center pb-2 border-b border-slate-100">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 font-serif">Change Password</h3>
                    <p className="text-[10px] text-slate-500">Ensure your account is using a strong password.</p>
                  </div>
                  <button
                    onClick={() => triggerToast("Password updated successfully.")}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold rounded-lg shadow-sm font-sans"
                  >
                    <Lock className="w-3.5 h-3.5 text-slate-400" /> Change Password
                  </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-sans mt-1">
                  <div className="space-y-1">
                    <label className="text-[9px] text-slate-400 font-bold uppercase">Current Password</label>
                    <input
                      type="password"
                      value="password123"
                      readOnly
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-700 focus:outline-none"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] text-slate-400 font-bold uppercase">New Password</label>
                    <input
                      type="password"
                      placeholder="••••••••"
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-700 focus:outline-none"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] text-slate-400 font-bold uppercase">Confirm New Password</label>
                    <input
                      type="password"
                      placeholder="••••••••"
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-700 focus:outline-none"
                    />
                  </div>
                </div>
              </Card>

              {/* Preferences */}
              <Card className="p-5 flex flex-col justify-between gap-4">
                <div className="pb-2 border-b border-slate-100">
                  <h3 className="text-sm font-bold text-slate-900 font-serif">Preferences</h3>
                  <p className="text-[10px] text-slate-500">Customize your application experience.</p>
                </div>

                {/* Grid inputs */}
                <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-xs font-sans mt-1">
                  <div className="space-y-1">
                    <label className="text-[9px] text-slate-400 font-bold uppercase">Language</label>
                    <select
                      value={prefs.language}
                      onChange={(e) => handleSelectChange("language", e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none"
                    >
                      <option>English (India)</option>
                      <option>Tamil</option>
                      <option>Hindi</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] text-slate-400 font-bold uppercase">Date Format</label>
                    <select
                      value={prefs.dateFormat}
                      onChange={(e) => handleSelectChange("dateFormat", e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none"
                    >
                      <option>DD MMM YYYY</option>
                      <option>YYYY-MM-DD</option>
                      <option>MM/DD/YYYY</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] text-slate-400 font-bold uppercase">Time Zone</label>
                    <select
                      value={prefs.timezone}
                      onChange={(e) => handleSelectChange("timezone", e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none"
                    >
                      <option>(GMT+05:30) Asia/Kolkata</option>
                      <option>(GMT+00:00) UTC</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] text-slate-400 font-bold uppercase">Currency</label>
                    <select
                      value={prefs.currency}
                      onChange={(e) => handleSelectChange("currency", e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none"
                    >
                      <option>INR (₹)</option>
                      <option>USD ($)</option>
                    </select>
                  </div>
                </div>

                {/* Toggles */}
                <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-xs font-sans mt-3 border-t border-slate-100/50 pt-4">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-800">Enable Dark Mode</span>
                      <input
                        type="checkbox"
                        checked={prefs.darkMode}
                        onChange={() => togglePref("darkMode")}
                        className="w-8 h-4 rounded-full accent-blue-600 cursor-pointer"
                      />
                    </div>
                    <span className="block text-[9px] text-slate-400 leading-tight">Switch between light and dark theme</span>
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-800">Compact View</span>
                      <input
                        type="checkbox"
                        checked={prefs.compactView}
                        onChange={() => togglePref("compactView")}
                        className="w-8 h-4 rounded-full accent-blue-600 cursor-pointer"
                      />
                    </div>
                    <span className="block text-[9px] text-slate-400 leading-tight">Show more content in less space</span>
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-800">Show Tips & Suggestions</span>
                      <input
                        type="checkbox"
                        checked={prefs.showTips}
                        onChange={() => togglePref("showTips")}
                        className="w-8 h-4 rounded-full accent-blue-600 cursor-pointer"
                      />
                    </div>
                    <span className="block text-[9px] text-slate-400 leading-tight">Receive helpful tips while using the app</span>
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-800">Auto Refresh Data</span>
                      <input
                        type="checkbox"
                        checked={prefs.autoRefresh}
                        onChange={() => togglePref("autoRefresh")}
                        className="w-8 h-4 rounded-full accent-blue-600 cursor-pointer"
                      />
                    </div>
                    <span className="block text-[9px] text-slate-400 leading-tight">Automatically refresh dashboard data</span>
                  </div>
                </div>
              </Card>

              {/* Data & Backup */}
              <Card className="p-5 flex flex-col justify-between gap-4">
                <div className="flex justify-between items-center pb-2 border-b border-slate-100">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 font-serif">Data & Backup</h3>
                    <p className="text-[10px] text-slate-500">Manage your data backups and export options.</p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => triggerToast("Backup initiated.")}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-705 text-xs font-bold rounded-lg shadow-sm font-sans"
                    >
                      <RefreshCw className="w-3.5 h-3.5 text-slate-400" /> Backup Now
                    </button>
                    <button
                      onClick={() => triggerToast("Data export file downloaded.")}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-705 text-xs font-bold rounded-lg shadow-sm font-sans"
                    >
                      <Download className="w-3.5 h-3.5 text-slate-400" /> Export Data
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-sans mt-1">
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Last Backup</span>
                    <span className="font-bold text-slate-800">May 16, 2026 10:30 AM</span>
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Backup Frequency</span>
                    <span className="font-semibold text-slate-800">Daily</span>
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Next Backup</span>
                    <span className="font-semibold text-slate-800">May 17, 2026 10:30 AM</span>
                  </div>
                </div>
              </Card>

              {/* Danger Zone */}
              <Card className="p-5 flex flex-col justify-between gap-4 border-rose-100 bg-rose-50/10">
                <div className="pb-2 border-b border-rose-100">
                  <h3 className="text-sm font-bold text-rose-700 font-serif">Danger Zone</h3>
                  <p className="text-[10px] text-rose-500">Irreversible actions that affect your account and data.</p>
                </div>

                <div className="flex flex-wrap gap-3 mt-1">
                  <button
                    onClick={() => triggerToast("Application state resetted to factory defaults.")}
                    className="flex items-center gap-1.5 px-3.5 py-2 border border-rose-500 text-rose-600 hover:bg-rose-50 text-xs font-bold rounded-lg font-sans transition-all"
                  >
                    <RefreshCw className="w-3.5 h-3.5" /> Reset Application
                  </button>
                  <button
                    onClick={() => triggerToast("Account deletion ticket submitted.")}
                    className="flex items-center gap-1.5 px-3.5 py-2 border border-rose-500 text-rose-600 hover:bg-rose-50 text-xs font-bold rounded-lg font-sans transition-all"
                  >
                    <Trash2 className="w-3.5 h-3.5" /> Delete Account
                  </button>
                </div>
              </Card>
            </div>
          ) : (
            <Card className="flex flex-col items-center justify-center text-center p-12 text-slate-400 min-h-[300px]">
              <Lock className="w-12 h-12 text-slate-300 mb-2" />
              <h3 className="text-sm font-bold text-slate-800 font-serif">Module Lock</h3>
              <p className="text-xs text-slate-500 mt-1 max-w-sm">
                This configuration group is locked for sub-administrators. Contact Arjun Sharma for deployment clearance.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
