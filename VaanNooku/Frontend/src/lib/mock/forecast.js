// 7-day rolling raw ML signal + actual sales baseline, before Decision
// Intelligence / Market Realism adjustments are layered on in lib/api/forecasting.js
export const DEMAND_BASELINE = {
  days: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
  rawSignal: [450, 480, 520, 590, 840, 910, 680],
  actualSales: [430, 470, 510, 570, 780, 800, 650],
};
