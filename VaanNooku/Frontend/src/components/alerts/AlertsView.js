"use client";

import React, { useCallback } from "react";
import { BellRing, AlertTriangle, ArrowRight } from "lucide-react";
import { useStoreContext } from "@/context/StoreContext";
import { useAsync } from "@/hooks/useAsync";
import { getInventory } from "@/lib/api/inventory";
import { PageHeader, Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";

export default function AlertsView() {
  const { activeStore } = useStoreContext();

  const loader = useCallback(async () => {
    const rawItems = await getInventory(activeStore?.id);
    return rawItems.filter((i) => i.stock <= i.rop);
  }, [activeStore]);

  const { data: shortages, isLoading } = useAsync(loader, [activeStore]);

  if (isLoading || !shortages) {
    return (
      <div className="space-y-6">
        <div className="h-10 w-1/3 bg-slate-200 rounded animate-pulse" />
        <Skeleton className="h-62.5" />
      </div>
    );
  }

  return (
    <div className="space-y-6 font-sans px-6">
      <PageHeader
        title="Alerts & Stock Risks"
        subtitle="Critical stock warnings and items currently below reorder points."
        icon={BellRing}
      />

      {shortages.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {shortages.map((item, idx) => (
            <Card key={idx} className="border-rose-100 bg-rose-50/20 p-5 flex flex-col justify-between gap-4">
              <div className="space-y-2">
                <div className="flex justify-between items-start">
                  <span className="px-2 py-0.5 rounded bg-rose-100 text-rose-700 text-[9px] font-bold uppercase tracking-wider">
                    {item.category}
                  </span>
                  <span className="inline-flex items-center gap-1 text-[9px] font-bold text-rose-600">
                    <AlertTriangle className="w-3.5 h-3.5" /> High Risk
                  </span>
                </div>

                <h3 className="text-sm font-bold text-slate-900 font-serif">{item.name}</h3>

                <p className="text-xs text-slate-600 leading-relaxed font-sans">
                  Active stock level is <strong className="text-rose-700">{item.stock} units</strong>, falling below the Reorder Point limit of <strong className="text-slate-800">{item.rop} units</strong>. Safety stock threshold is {item.minStock} units.
                </p>
              </div>

              <div className="border-t border-rose-100/50 pt-3 flex justify-between items-center text-[10px] text-slate-500 font-sans">
                <span>Calculated Order EOQ: <strong>{item.eoq} units</strong></span>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="flex flex-col items-center justify-center text-center p-12 text-slate-400">
          <BellRing className="w-12 h-12 text-slate-300 mb-2" />
          <h3 className="text-sm font-bold text-slate-800 font-serif">No Active Deficits</h3>
          <p className="text-xs text-slate-500 mt-1 max-w-sm">
            All inventory levels are currently above reorder limits and safety buffers.
          </p>
        </Card>
      )}
    </div>
  );
}
