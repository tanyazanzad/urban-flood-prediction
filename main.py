"""
main.py
-------
End-to-end urban flood prediction pipeline:

  1. Load/generate catchment data (rainfall, drainage, elevation, land use, soil)
  2. Train ML models -> flood probability (classifier) + runoff (regressor)
  3. Evaluate models
  4. Score a design-storm scenario across the catchment
  5. Render risk map, runoff map, and evaluation charts
  6. Save models + prediction tables

Run:  python3 main.py
"""

import os
import pandas as pd

from data_generator import generate_catchment
from model import (
    train_flood_classifier, train_runoff_regressor,
    get_feature_importance, save_models,
)
from risk_map import (
    score_design_storm, plot_risk_map, plot_runoff_map,
    plot_elevation_context, plot_feature_importance, plot_model_evaluation,
)

OUT_DIR = "outputs"
DESIGN_STORM_MM = 90  # e.g. a ~10-20yr return-period 24h rainfall depth


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/6] Generating catchment dataset (rainfall, drainage, elevation, land use, soil)...")
    df = generate_catchment(grid_size=40, storms_per_cell=6, seed=42)
    df.to_csv(f"{OUT_DIR}/catchment_dataset.csv", index=False)
    print(f"    -> {df.shape[0]} rows across {df['x'].max()+1}x{df['y'].max()+1} grid cells")
    print(f"    -> observed flood rate in training data: {df['flood_occurred'].mean():.1%}")

    print("\n[2/6] Training flood probability classifier (RandomForestClassifier)...")
    clf, clf_metrics, (Xc_test, yc_test, yc_proba) = train_flood_classifier(df)
    print(clf_metrics["classification_report"])
    print(f"    ROC-AUC: {clf_metrics['roc_auc']:.3f}")

    print("\n[3/6] Training runoff regressor (RandomForestRegressor, SCS-CN-based target)...")
    reg, reg_metrics, (Xr_test, yr_test, yr_pred) = train_runoff_regressor(df)
    print(f"    MAE:  {reg_metrics['mae_mm']:.2f} mm")
    print(f"    RMSE: {reg_metrics['rmse_mm']:.2f} mm")
    print(f"    R^2:  {reg_metrics['r2']:.3f}")

    print("\n[4/6] Scoring design storm scenario "
          f"({DESIGN_STORM_MM} mm / 24h) across the catchment...")
    cells = score_design_storm(df, clf, reg, rainfall_mm=DESIGN_STORM_MM)
    cells_out = cells[[
        "x", "y", "elevation_m", "land_use", "soil_type",
        "drainage_capacity_mmhr", "flood_probability", "predicted_runoff_mm",
    ]].copy()
    cells_out["risk_category"] = pd.cut(
        cells_out["flood_probability"], bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["Low", "Moderate", "High", "Very High"], include_lowest=True,
    )
    cells_out.to_csv(f"{OUT_DIR}/design_storm_predictions.csv", index=False)
    print(cells_out["risk_category"].value_counts().sort_index())

    print("\n[5/6] Rendering maps and evaluation charts...")
    plot_elevation_context(df, f"{OUT_DIR}/map_elevation_context.png")
    plot_risk_map(cells, DESIGN_STORM_MM, f"{OUT_DIR}/map_flood_risk.png")
    plot_runoff_map(cells, DESIGN_STORM_MM, f"{OUT_DIR}/map_predicted_runoff.png")
    plot_model_evaluation(yc_test, yc_proba, yr_test, yr_pred, f"{OUT_DIR}/model_evaluation.png")

    clf_importance = get_feature_importance(clf)
    reg_importance = get_feature_importance(reg)
    plot_feature_importance(clf_importance, "Flood Probability — Feature Importance",
                             f"{OUT_DIR}/feature_importance_flood.png")
    plot_feature_importance(reg_importance, "Runoff — Feature Importance",
                             f"{OUT_DIR}/feature_importance_runoff.png")

    print("\n[6/6] Saving trained models...")
    save_models(clf, reg, OUT_DIR)

    print(f"\nDone. All outputs written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
