// Base learners of the production Stacking Ensemble (see summary.md).
export const BASE_MODELS = [
  { name: "LightGBM", type: "Gradient Boosting", r2: 0.926, mae: 11.74, rmse: 16.86, trainTimeSec: 4.2, stackWeight: 0.27 },
  { name: "Random Forest", type: "Bagging / Tree Ensemble", r2: 0.9306, mae: 11.46, rmse: 16.32, trainTimeSec: 6.8, stackWeight: 0.26 },
  { name: "XGBoost", type: "Gradient Boosting", r2: 0.9281, mae: 11.67, rmse: 16.62, trainTimeSec: 5.1, stackWeight: 0.25 },
  { name: "CatBoost", type: "Gradient Boosting (ordered)", r2: 0.9245, mae: 11.98, rmse: 17.05, trainTimeSec: 7.4, stackWeight: 0.22 },
];

// The meta-learner combines all four base learners' out-of-fold predictions.
export const STACKED_ENSEMBLE = {
  name: "Stacking Ensemble (Meta-Learner)",
  metaLearner: "Ridge Regression",
  r2: 0.9412,
  mae: 10.21,
  rmse: 15.04,
  description:
    "A ridge meta-learner blends out-of-fold predictions from LightGBM, Random Forest, XGBoost, and CatBoost, correcting individual model bias and producing the most stable daily demand signal.",
};

export const TRAINING_SUMMARY = {
  datasetRows: 3780,
  featureCount: 28,
  trainRows: 3024,
  testRows: 756,
  splitRatio: "80 / 20",
  retrainCadenceDays: 14,
};
