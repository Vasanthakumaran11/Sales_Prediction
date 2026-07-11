"use client";

import React, { useState, useEffect } from "react";
import { Users, ShieldCheck, Mail, Phone } from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";
import { useStoreContext } from "@/context/StoreContext";

const DEFAULT_SUPPLIERS = [];

export default function SuppliersView() {
  const { activeStore } = useStoreContext();
  const [apiSuppliers, setApiSuppliers] = useState(null);

  useEffect(() => {
    const demoIds = ["balaji-store", "shiva-stores", "surya-markets"];
    const isDemo = activeStore ? demoIds.includes(activeStore.id) : true;
    
    if (!isDemo) {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || "";
      if (apiBase) {
        fetch(`${apiBase}/api/suppliers/suggestions`)
          .then((res) => res.json())
          .then((data) => {
            if (Array.isArray(data)) {
              setApiSuppliers(data.map(s => ({
                name: s.name,
                category: s.category || "General Wholesaler",
                email: s.email || "contact@wholesaler.com",
                phone: s.phone || "+91 99999 88888",
                leadTime: `${s.lead_time_days || 2} Days`,
                discount: `${s.min_order_qty || 10} Units Min Order`,
                status: "Verified Supplier"
              })));
            }
          })
          .catch((err) => {
            console.error("Error fetching suppliers:", err);
          });
      }
    }
  }, [activeStore]);

  // Derive suppliers list dynamically to prevent cascading render warnings
  const suppliers = apiSuppliers || DEFAULT_SUPPLIERS;

  return (
    <div className="space-y-6 font-sans px-6 ">
      <PageHeader
        title="Wholesale Suppliers Matrix"
        subtitle="Manage vendor delivery contracts, wholesale lead times, and bulk volume rates."
        icon={Users}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {suppliers.map((sup, idx) => (
          <Card key={idx} className="flex flex-col justify-between gap-5 border-sky-100 hover:border-sky-300 transition-all">
            <div className="space-y-3">
              <div className="flex justify-between items-start">
                <span className="px-2 py-0.5 rounded bg-sky-50 border border-sky-200 text-sky-700 text-[9px] font-bold uppercase tracking-wider">
                  {sup.category}
                </span>
                <span className="inline-flex items-center gap-1 text-[9px] font-bold text-emerald-600">
                  <ShieldCheck className="w-3.5 h-3.5" /> {sup.status}
                </span>
              </div>

              <h3 className="text-sm font-bold text-slate-900 font-serif">{sup.name}</h3>

              <div className="space-y-1.5 text-xs text-slate-600">
                <div className="flex items-center gap-2">
                  <Mail className="w-3.5 h-3.5 text-slate-400" />
                  <span>{sup.email}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Phone className="w-3.5 h-3.5 text-slate-400" />
                  <span>{sup.phone}</span>
                </div>
              </div>
            </div>

            <div className="border-t border-slate-100 pt-3 flex justify-between items-center text-[10px] text-slate-500">
              <div>
                <span className="block font-semibold">Wholesale Discount:</span>
                <span className="font-bold text-sky-600">{sup.discount}</span>
              </div>
              <div className="text-right">
                <span className="block font-semibold">Lead Time:</span>
                <span className="font-bold text-slate-800">{sup.leadTime}</span>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
