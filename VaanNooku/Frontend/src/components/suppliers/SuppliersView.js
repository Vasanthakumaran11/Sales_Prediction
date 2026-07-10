"use client";

import React from "react";
import { Users, ShieldCheck, Mail, Phone } from "lucide-react";
import { PageHeader, Card } from "@/components/ui/Card";

export default function SuppliersView() {
  const suppliers = [
    {
      name: "Balaji Agro Distributors",
      category: "Staples & Grains",
      contact: "Mukesh Patel",
      phone: "+91 98450 12345",
      email: "procurement@balajiagro.com",
      leadTime: "2 Days",
      discount: "15% Bulk Contract",
      status: "Verified Partner",
    },
    {
      name: "Shiva Dairy & Farms",
      category: "Perishables & Cold Storage",
      contact: "Aravind Swamy",
      phone: "+91 99120 54321",
      email: "supply@shivafarms.in",
      leadTime: "1 Day (Same-Day Express)",
      discount: "10% Standard Rate",
      status: "Active Vendor",
    },
    {
      name: "Surya Packaged Goods Ltd",
      category: "Snacks, Beverages & Care",
      contact: "K. R. Nair",
      phone: "+91 88770 98765",
      email: "nair@suryapackaged.com",
      leadTime: "3 Days",
      discount: "12% Volume Scaling",
      status: "Verified Partner",
    },
    {
      name: "Krishna Beverages Co.",
      category: "Beverages & Drinks",
      contact: "Amit Sharma",
      phone: "+91 97654 32109",
      email: "sales@krishnabev.in",
      leadTime: "2 Days",
      discount: "8% Standard Contract",
      status: "Active Vendor",
    },
    {
      name: "Durga Household Essentials",
      category: "Household & Personal Care",
      contact: "Priya Das",
      phone: "+91 81234 56789",
      email: "priya@durgaessentials.com",
      leadTime: "4 Days",
      discount: "14% Bulk Volume Rate",
      status: "Verified Partner",
    },
  ];

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
