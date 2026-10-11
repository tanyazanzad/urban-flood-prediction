"""
risk_map.py
-----------
Renders the trained models' outputs as spatial maps for a chosen "design
storm" scenario, plus evaluation charts. Uses matplotlib so it runs with no
extra dependencies. If geopandas/rasterio are available in your environment,
swap `plot_risk_map`'s imshow call for a proper CRS-aware raster/vector plot
of the same underlying grid — the model and prediction logic don't change.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

from features import ALL_FEATURES

RISK_COLORS = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c"]  # low -> very high
RISK_LABELS = ["Low", "Moderate", "High", "Very High"]
RISK_BOUNDS = [0.0, 0.25, 0.5, 0.75, 1.0]


def score_design_storm(df: pd.DataFrame, clf, reg, rainfall_mm: float) -> pd.DataFrame:
    """
    Apply a uniform design-storm rainfall depth (e.g. the local 100-year,
    24-hour return-period value from IDF curves) across every grid cell,
    holding elevation/drainage/land use/soil fixed, and score the models.
    """
    cells = df.drop_duplicates(subset=["x", "y"]).copy()
    cells["rainfall_mm_24h"] = rainfall_mm
    X = cells[ALL_FEATURES]
    cells["flood_probability"] = clf.predict_proba(X)[:, 1]
    cells["predicted_runoff_mm"] = reg.predict(X)
    return cells


def _grid_from_cells(cells: pd.DataFrame, value_col: str) -> np.ndarray:
    n_x = cells["x"].max() + 1
    n_y = cells["y"].max() + 1
    grid = np.full((n_y, n_x), np.nan)
    for _, r in cells.iterrows():
        grid[int(r["y"]), int(r["x"])] = r[value_col]
    return grid


def plot_risk_map(cells: pd.DataFrame, rainfall_mm: float, out_path: str):
    grid = _grid_from_cells(cells, "flood_probability")

    cmap = ListedColormap(RISK_COLORS)
    norm = BoundaryNorm(RISK_BOUNDS, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap=cmap, norm=norm, origin="upper")
    ax.set_title(f"Urban Flood Risk Map\nDesign storm: {rainfall_mm:.0f} mm / 24h", fontsize=13)
    ax.set_xlabel("Grid X (east \u2192)")
    ax.set_ylabel("Grid Y (north \u2191)")

    cbar = fig.colorbar(im, ax=ax, ticks=[0.125, 0.375, 0.625, 0.875])
    cbar.ax.set_yticklabels(RISK_LABELS)
    cbar.set_label("Flood Risk Category")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_runoff_map(cells: pd.DataFrame, rainfall_mm: float, out_path: str):
    grid = _grid_from_cells(cells, "predicted_runoff_mm")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="YlGnBu", origin="upper")
    ax.set_title(f"Predicted Runoff (mm)\nDesign storm: {rainfall_mm:.0f} mm / 24h", fontsize=13)
    ax.set_xlabel("Grid X (east \u2192)")
    ax.set_ylabel("Grid Y (north \u2191)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Runoff (mm)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_elevation_context(df: pd.DataFrame, out_path: str):
    cells = df.drop_duplicates(subset=["x", "y"])
    elev_grid = _grid_from_cells(cells, "elevation_m")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(elev_grid, cmap="terrain", origin="upper")
    ax.set_title("Catchment Elevation (context layer)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Elevation (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance: pd.Series, title: str, out_path: str, top_n: int = 12):
    top = importance.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.index, top.values, color="#2b83ba")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Relative Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_evaluation(y_test_clf, y_proba, y_test_reg, y_pred_reg, out_path: str):
    from sklearn.metrics import roc_curve, RocCurveDisplay

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_test_clf, y_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("Flood Classifier: ROC Curve")

    axes[1].scatter(y_test_reg, y_pred_reg, alpha=0.3, s=10, color="#2b83ba")
    lims = [0, max(y_test_reg.max(), y_pred_reg.max())]
    axes[1].plot(lims, lims, "k--", lw=1)
    axes[1].set_xlabel("Actual Runoff (mm)")
    axes[1].set_ylabel("Predicted Runoff (mm)")
    axes[1].set_title("Runoff Regressor: Predicted vs Actual")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
