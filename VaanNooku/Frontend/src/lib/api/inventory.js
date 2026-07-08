import { resolveData } from "./client";
import { INVENTORY_ITEMS, BULK_CONSOLIDATION_ITEMS, BULK_CONSOLIDATION_DISCOUNT } from "@/lib/mock/inventory";
import { STORE_PROFILES } from "@/lib/mock/stores";

// GET /api/stores/:storeId/inventory
export async function getInventory(storeId) {
  return resolveData(`/api/stores/${storeId ?? "executive"}/inventory`, () => INVENTORY_ITEMS);
}

// GET /api/inventory/bulk-consolidation
export async function getBulkConsolidation() {
  return resolveData("/api/inventory/bulk-consolidation", () => {
    const stores = STORE_PROFILES;
    const items = BULK_CONSOLIDATION_ITEMS.map((item) => {
      const totalQty = stores.reduce((sum, s) => sum + (item[s.id] ?? 0), 0);
      const standardCost = totalQty * item.price;
      return { ...item, totalQty, standardCost };
    });
    const totalStandardCost = items.reduce((sum, i) => sum + i.standardCost, 0);
    const totalDiscountedCost = Math.round(totalStandardCost * (1 - BULK_CONSOLIDATION_DISCOUNT));
    return {
      stores,
      items,
      discountRate: BULK_CONSOLIDATION_DISCOUNT,
      totalStandardCost,
      totalDiscountedCost,
      totalSavings: totalStandardCost - totalDiscountedCost,
    };
  });
}

// POST /api/inventory/bulk-consolidation/orders
export async function placeBulkOrder(orderPayload) {
  return resolveData(
    "/api/inventory/bulk-consolidation/orders",
    () => ({ orderId: `PO-${Date.now()}`, status: "dispatched", ...orderPayload }),
    { method: "POST", body: JSON.stringify(orderPayload) }
  );
}
