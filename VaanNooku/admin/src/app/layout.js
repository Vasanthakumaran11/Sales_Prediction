import "./globals.css";
import { AdminProvider } from "@/context/AdminContext";

export const metadata = {
  title: "VaanNooku Admin — AI Operations Console",
  description: "Model monitoring, dataset management, retraining, and complaint triage for the RetailAI ensemble.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900">
        <AdminProvider>{children}</AdminProvider>
      </body>
    </html>
  );
}
