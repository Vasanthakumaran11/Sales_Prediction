"use client";

import { createContext, useContext, useMemo, useState, useCallback, useEffect } from "react";
import { setAdminToken, getAdminToken } from "@/lib/api/client";

const AdminContext = createContext(null);

export function AdminProvider({ children }) {
  const [admin, setAdmin] = useState(null);
  const [activeView, setActiveView] = useState("models");
  const [checkedSession, setCheckedSession] = useState(false);

  // A stored token without a hydrated `admin` object (e.g. page refresh) still
  // counts as logged in for routing purposes — the API calls will 401 and
  // bounce back to login if the token turned out to be invalid/expired.
  useEffect(() => {
    if (getAdminToken()) {
      setAdmin((prev) => prev || { email: null, restoredFromSession: true });
    }
    setCheckedSession(true);
  }, []);

  const login = useCallback((adminData, token) => {
    setAdminToken(token);
    setAdmin(adminData);
  }, []);

  const logout = useCallback(() => {
    setAdminToken(null);
    setAdmin(null);
  }, []);

  const value = useMemo(
    () => ({ admin, login, logout, activeView, setActiveView, checkedSession }),
    [admin, login, logout, activeView, checkedSession]
  );

  return <AdminContext.Provider value={value}>{children}</AdminContext.Provider>;
}

export function useAdminContext() {
  const ctx = useContext(AdminContext);
  if (!ctx) throw new Error("useAdminContext must be used within AdminProvider");
  return ctx;
}
