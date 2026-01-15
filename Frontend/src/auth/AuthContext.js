import React, { createContext, useContext, useMemo, useState } from "react";

const AuthContext = createContext(null);
const STORAGE_KEY = "taxi_stgnn_admin_authed";

export function AuthProvider({ children }) {
  const [isAuthed, setIsAuthed] = useState(() => window.localStorage.getItem(STORAGE_KEY) === "true");

  const login = (email, password) => {
    const adminEmail = process.env.REACT_APP_ADMIN_EMAIL || "";
    const adminPassword = process.env.REACT_APP_ADMIN_PASSWORD || "";
    if (email === adminEmail && password === adminPassword) {
      setIsAuthed(true);
      window.localStorage.setItem(STORAGE_KEY, "true");
      return { ok: true };
    }
    return { ok: false, error: "Invalid email or password." };
  };

  const logout = () => {
    setIsAuthed(false);
    window.localStorage.removeItem(STORAGE_KEY);
  };

  const value = useMemo(() => ({ isAuthed, login, logout }), [isAuthed]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
