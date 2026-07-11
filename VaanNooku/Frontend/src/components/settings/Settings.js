"use client";

import React, { useState, useEffect } from "react";
import { useStoreContext } from "@/context/StoreContext";
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
  Eye,
  EyeOff,
} from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";

export default function Settings() {
  const { theme, setTheme, exitToGateway, activeStore } = useStoreContext();
  const [activeTab, setActiveTab] = useState("profile");

  // Determine if this is a demo store (not a real registered store)
  const isDemo = activeStore
    ? ["balaji-store", "shiva-stores", "surya-markets"].includes(activeStore.id)
    : true;
  
  // Form Preference states
  const [prefs, setPrefs] = useState({
    darkMode: theme === "dark",
    compactView: true,
    showTips: true,
    autoRefresh: false,
    language: "English (India)",
    dateFormat: "DD MMM YYYY",
    timezone: "(GMT+05:30) Asia/Kolkata",
    currency: "INR (₹)"
  });

  // Profile Information states
  const [profile, setProfile] = useState({
    name: "Arjun Sharma",
    email: "arjun.sharma@retailai.com",
    phone: "+91 98765 43210",
    role: "Administrator"
  });
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [profileForm, setProfileForm] = useState({ ...profile });

  // Store Admin details (shown only if dynamically registered)
  const [adminDetails, setAdminDetails] = useState({
    fullName: "",
    email: "",
    phone: "",
    role: "Store Admin"
  });

  // Password Update Form states
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: "password123",
    newPassword: "",
    confirmNewPassword: ""
  });

  // Password visibility toggles
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Account deletion modal state
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  // Backup logs
  const [lastBackup, setLastBackup] = useState("May 16, 2026 10:30 AM");

  // Sync profile values when activeStore changes
  useEffect(() => {
    if (activeStore) {
      const defaultProfile = isDemo ? {
        name: activeStore.name === "Balaji Store" ? "Arjun Sharma" : `${activeStore.name} Admin`,
        email: activeStore.name === "Balaji Store" ? "arjun.sharma@retailai.com" : `admin@${activeStore.id}.com`,
        phone: activeStore.name === "Balaji Store" ? "+91 98765 43210" : "+91 99999 88888",
        role: "Administrator"
      } : {
        // Use real admin data from signup if available
        name: activeStore.adminName || `${activeStore.name} Owner`,
        email: activeStore.adminEmail || `owner@${activeStore.name.toLowerCase().replace(/\s+/g, "")}.com`,
        phone: activeStore.adminPhone || "+91 —",
        role: activeStore.adminRole || "Store Owner"
      };

      // Defer state updates to avoid synchronous cascading renders warning
      setTimeout(() => {
        setProfile(defaultProfile);
        setProfileForm(defaultProfile);

        if (!isDemo) {
          setAdminDetails({
            fullName: `${activeStore.name} Manager`,
            email: `manager@${activeStore.name.toLowerCase().replace(/\s+/g, "")}.com`,
            phone: "+91 99999 88888",
            role: "Store Admin"
          });
        }
      }, 0);
    }
  }, [activeStore]);

  useEffect(() => {
    setTimeout(() => {
      setPrefs((prev) => {
        if (prev.darkMode !== (theme === "dark")) {
          return { ...prev, darkMode: theme === "dark" };
        }
        return prev;
      });
    }, 0);
  }, [theme]);

  const [toastMsg, setToastMsg] = useState("");

  const triggerToast = (msg) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(""), 3000);
  };

  const togglePref = (key) => {
    if (key === "darkMode") {
      const nextTheme = theme === "dark" ? "light" : "dark";
      setTheme(nextTheme);
      triggerToast(`Theme changed to ${nextTheme === "dark" ? "Dark" : "Light"} Mode.`);
    } else {
      setPrefs((prev) => ({ ...prev, [key]: !prev[key] }));
    }
  };

  const handleSelectChange = (key, value) => {
    setPrefs((prev) => ({ ...prev, [key]: value }));
  };

  // Profile Edit Toggle & Save
  const handleEditProfileToggle = () => {
    if (isEditingProfile) {
      setProfile({ ...profileForm });
      setIsEditingProfile(false);
      triggerToast("Profile details updated successfully.");
    } else {
      setProfileForm({ ...profile });
      setIsEditingProfile(true);
    }
  };

  // Change Password Action
  const handleChangePassword = () => {
    if (!passwordForm.newPassword || !passwordForm.confirmNewPassword) {
      triggerToast("Please fill in both new password fields.");
      return;
    }
    if (passwordForm.newPassword !== passwordForm.confirmNewPassword) {
      triggerToast("Error: Confirm password does not match new password.");
      return;
    }
    setPasswordForm((prev) => ({
      currentPassword: prev.newPassword,
      newPassword: "",
      confirmNewPassword: ""
    }));
    triggerToast("Password changed successfully.");
  };

  // Backup Now Action
  const handleBackupNow = () => {
    const timeString = new Date().toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hour12: true
    });
    setLastBackup(timeString);
    triggerToast("Database backup completed successfully.");
  };

  // Export Data settings Action
  const handleExportData = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({ profile, prefs }, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "retail_ai_settings_backup.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    document.body.removeChild(downloadAnchor);
    triggerToast("Settings backup exported successfully.");
  };

  // Danger Zone handlers
  const handleResetApplication = () => {
    if (window.confirm("Are you sure you want to restore application parameters to factory settings? All temporary logs will be cleared.")) {
      setPrefs({
        darkMode: false,
        compactView: true,
        showTips: true,
        autoRefresh: false,
        language: "English (India)",
        dateFormat: "DD MMM YYYY",
        timezone: "(GMT+05:30) Asia/Kolkata",
        currency: "INR (₹)"
      });
      setTheme("light");
      setProfile({
        name: "Arjun Sharma",
        email: "arjun.sharma@retailai.com",
        phone: "+91 98765 43210",
        role: "Administrator"
      });
      triggerToast("Application state reset to defaults.");
    }
  };

  const handleDeleteAccount = () => {
    setShowDeleteModal(true);
  };

  const executeDeleteAccount = () => {
    setShowDeleteModal(false);
    triggerToast("Deactivating account...");
    setTimeout(() => {
      exitToGateway();
    }, 1000);
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
    <div className="space-y-6 font-sans px-6">
      <PageHeader
        title="Settings"
        icon={Sliders}
      />

      {toastMsg && (
        <div className="fixed bottom-5 right-5 p-4 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center gap-2 text-xs text-emerald-800 shadow-md z-50 animate-fade-in">
          <CheckCircle className="w-4 h-4 text-emerald-500" />
          <span>{toastMsg}</span>
        </div>
      )}

      {/* Main Layout Grid - Adjusted to take full width */}
      <div className="w-full space-y-6">
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
                    onClick={handleEditProfileToggle}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold rounded-lg shadow-sm font-sans"
                  >
                    <Edit2 className="w-3.5 h-3.5" /> {isEditingProfile ? "Save Changes" : "Edit Profile"}
                  </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-xs font-sans mt-1">
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Full Name</span>
                    {isEditingProfile ? (
                      <input
                        type="text"
                        value={profileForm.name}
                        onChange={(e) => setProfileForm((prev) => ({ ...prev, name: e.target.value }))}
                        className="mt-1 w-full bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-slate-800 focus:outline-none"
                      />
                    ) : (
                      <span className="font-bold text-slate-800">{profile.name}</span>
                    )}
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Email Address</span>
                    {isEditingProfile ? (
                      <input
                        type="email"
                        value={profileForm.email}
                        onChange={(e) => setProfileForm((prev) => ({ ...prev, email: e.target.value }))}
                        className="mt-1 w-full bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-slate-800 focus:outline-none"
                      />
                    ) : (
                      <span className="font-semibold text-slate-800">{profile.email}</span>
                    )}
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Phone Number</span>
                    {isEditingProfile ? (
                      <input
                        type="text"
                        value={profileForm.phone}
                        onChange={(e) => setProfileForm((prev) => ({ ...prev, phone: e.target.value }))}
                        className="mt-1 w-full bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-slate-800 focus:outline-none"
                      />
                    ) : (
                      <span className="font-semibold text-slate-800">{profile.phone}</span>
                    )}
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Role</span>
                    <span className="px-2.5 py-0.5 rounded bg-blue-50 text-blue-600 text-[10px] font-bold mt-0.5 inline-block uppercase">
                      {profile.role}
                    </span>
                  </div>
                </div>
              </Card>

              {/* Store Admin Details (Shown only for dynamically registered stores) */}
              {!isDemo && (
                <Card className="p-5 flex flex-col justify-between gap-4">
                  <div className="flex justify-between items-center pb-2 border-b border-slate-100">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900 font-serif">Store Admin Details</h3>
                      <p className="text-[10px] text-slate-500 font-sans">Configure the primary contact person for this outlet.</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-sans mt-1">
                    <div className="space-y-1">
                      <label className="text-[9px] text-slate-400 font-bold uppercase">Admin Full Name</label>
                      <input
                        type="text"
                        value={adminDetails.fullName}
                        onChange={(e) => setAdminDetails(prev => ({ ...prev, fullName: e.target.value }))}
                        placeholder="e.g. Vasanthakumaran"
                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none focus:border-sky-500 font-sans font-semibold"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-[9px] text-slate-400 font-bold uppercase">Admin Email</label>
                      <input
                        type="email"
                        value={adminDetails.email}
                        onChange={(e) => setAdminDetails(prev => ({ ...prev, email: e.target.value }))}
                        placeholder="e.g. admin@store.com"
                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none focus:border-sky-500 font-sans font-semibold"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-[9px] text-slate-400 font-bold uppercase">Admin Phone Number</label>
                      <input
                        type="text"
                        value={adminDetails.phone}
                        onChange={(e) => setAdminDetails(prev => ({ ...prev, phone: e.target.value }))}
                        placeholder="e.g. +91 99999 88888"
                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none focus:border-sky-500 font-sans font-semibold"
                      />
                    </div>
                  </div>
                  <div className="flex justify-end pt-2">
                    <button
                      onClick={() => {
                        triggerToast("Store admin contact configuration saved successfully.");
                      }}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-lg shadow-sm font-sans"
                    >
                      Save Admin Details
                    </button>
                  </div>
                </Card>
              )}

              {/* Change Password */}
              <Card className="p-5 flex flex-col justify-between gap-4">
                <div className="flex justify-between items-center pb-2 border-b border-slate-100">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 font-serif">Change Password</h3>
                    <p className="text-[10px] text-slate-500">Ensure your account is using a strong password.</p>
                  </div>
                  <button
                    onClick={handleChangePassword}
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
                      value={passwordForm.currentPassword}
                      readOnly
                      className="w-full bg-slate-100 border border-slate-200 rounded-lg px-3 py-2 text-slate-500 focus:outline-none"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] text-slate-400 font-bold uppercase">New Password</label>
                    <div className="relative">
                      <input
                        type={showNewPassword ? "text" : "password"}
                        placeholder="••••••••"
                        value={passwordForm.newPassword}
                        onChange={(e) => setPasswordForm((prev) => ({ ...prev, newPassword: e.target.value }))}
                        className="w-full bg-slate-50 border border-slate-200 rounded-lg pl-3 pr-10 py-2 text-slate-700 focus:outline-none"
                      />
                      <button
                        type="button"
                        onClick={() => setShowNewPassword(!showNewPassword)}
                        className="absolute right-2.5 top-2 text-slate-400 hover:text-sky-650 transition-colors"
                      >
                        {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] text-slate-400 font-bold uppercase">Confirm New Password</label>
                    <div className="relative">
                      <input
                        type={showConfirmPassword ? "text" : "password"}
                        placeholder="••••••••"
                        value={passwordForm.confirmNewPassword}
                        onChange={(e) => setPasswordForm((prev) => ({ ...prev, confirmNewPassword: e.target.value }))}
                        className="w-full bg-slate-50 border border-slate-200 rounded-lg pl-3 pr-10 py-2 text-slate-700 focus:outline-none"
                      />
                      <button
                        type="button"
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                        className="absolute right-2.5 top-2 text-slate-400 hover:text-sky-655 transition-colors"
                      >
                        {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
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
                      onClick={handleBackupNow}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold rounded-lg shadow-sm font-sans"
                    >
                      <RefreshCw className="w-3.5 h-3.5 text-slate-400" /> Backup Now
                    </button>
                    <button
                      onClick={handleExportData}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold rounded-lg shadow-sm font-sans"
                    >
                      <Download className="w-3.5 h-3.5 text-slate-400" /> Export Data
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-sans mt-1">
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Last Backup</span>
                    <span className="font-bold text-slate-800">{lastBackup}</span>
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Backup Frequency</span>
                    <span className="font-semibold text-slate-800">Daily</span>
                  </div>
                  <div>
                    <span className="block text-[9px] text-slate-400 font-bold uppercase">Next Backup</span>
                    <span className="font-semibold text-slate-800">Scheduled Daily</span>
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
                    onClick={handleResetApplication}
                    className="flex items-center gap-1.5 px-3.5 py-2 border border-rose-500 text-rose-600 hover:bg-rose-50 text-xs font-bold rounded-lg font-sans transition-all"
                  >
                    <RefreshCw className="w-3.5 h-3.5" /> Reset Application
                  </button>
                  <button
                    onClick={handleDeleteAccount}
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
      {/* Custom Account Deletion Overlay Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center animate-fade-in font-sans">
          <div className="bg-white border border-rose-100 rounded-3xl p-8 max-w-md w-full shadow-2xl space-y-6 text-center">
            <div className="w-14 h-14 rounded-full bg-rose-50 border border-rose-100 flex items-center justify-center mx-auto text-rose-500">
              <AlertTriangle className="w-8 h-8 animate-bounce" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 font-serif">Confirm Account Deactivation</h3>
            <p className="text-xs text-slate-555 leading-relaxed">
              WARNING: Account deletion is permanent and cannot be undone. Are you sure you want to deactivate your license and purge all daily sales history from the server?
            </p>
            <div className="grid grid-cols-2 gap-4 pt-2">
              <button
                type="button"
                onClick={() => setShowDeleteModal(false)}
                className="py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold rounded-lg transition-all"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={executeDeleteAccount}
                className="py-2.5 bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold rounded-lg transition-all shadow-md"
              >
                Yes, Delete Account
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
