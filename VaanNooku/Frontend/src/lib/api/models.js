import { resolveData } from "./client";
import { BASE_MODELS, STACKED_ENSEMBLE, TRAINING_SUMMARY } from "@/lib/mock/models";

// GET /api/models/performance
export async function getModelPerformance() {
  return resolveData("/api/models/performance", () => ({
    baseModels: BASE_MODELS,
    stackedEnsemble: STACKED_ENSEMBLE,
    training: TRAINING_SUMMARY,
  }));
}
