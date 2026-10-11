# Urban Flood Prediction System

A Python pipeline that predicts **flood probability**, generates a **spatial risk map**,
and estimates **runoff** from rainfall, drainage, elevation, land use, and soil inputs.

## Stack
Python · Pandas · Scikit-learn · Matplotlib (GIS-ready — see "Using real GIS data" below)

## Pipeline
```
data_generator.py  -> synthetic catchment (stand-in for real rainfall/GIS inputs)
features.py         -> preprocessing pipeline (scaling + one-hot encoding)
model.py             -> RandomForestClassifier (flood probability)
                        RandomForestRegressor  (runoff, mm)
risk_map.py          -> design-storm scoring + map/chart rendering
main.py              -> orchestrates the full run
```

Run everything with:
```bash
python3 main.py
```

## Method

**Runoff target** is computed with the SCS Curve Number method, a standard
hydrology formula relating rainfall, land use, and soil hydrologic group to
runoff depth:
```
S  = 25400/CN - 254        (potential retention, mm)
Ia = 0.2 * S                (initial abstraction)
Q  = (P - Ia)^2 / (P - Ia + S)   for P > Ia, else 0
```
This gives the regressor a physically grounded, noisy target to learn — rather
than an arbitrary number — so its R² and error metrics mean something.

**Flood label** is built from a composite risk score (runoff vs. drainage
capacity deficit, low elevation, flat slope, imperviousness), turned into a
stochastic binary "did this cell flood in this storm" outcome. The classifier
then learns to reproduce that pattern from the raw features, and its
`predict_proba` output is the reported "flood probability."

**Risk map** applies a chosen design-storm rainfall depth (e.g. a 10–20yr
return-period 24h total) uniformly across the catchment grid, holding terrain/
drainage/land-use/soil fixed, and plots the resulting per-cell flood
probability in four bands (Low / Moderate / High / Very High).

## Outputs (written to `outputs/`)
| File | Description |
|---|---|
| `catchment_dataset.csv` | Full synthetic training dataset |
| `design_storm_predictions.csv` | Per-cell flood probability + runoff for the design storm |
| `map_flood_risk.png` | Risk map (4-tier classification) |
| `map_predicted_runoff.png` | Runoff depth map |
| `map_elevation_context.png` | Elevation layer for reference |
| `model_evaluation.png` | ROC curve + predicted-vs-actual runoff scatter |
| `feature_importance_flood.png` / `feature_importance_runoff.png` | Driver rankings |
| `flood_classifier.joblib` / `runoff_regressor.joblib` | Trained model pipelines |

## Using real GIS data

Swap `data_generator.py` for real loaders — the rest of the pipeline is
unchanged as long as you produce a DataFrame with one row per grid cell (or
per cell × storm event) containing these columns:

| Column | Source |
|---|---|
| `elevation_m`, `slope` | DEM raster (rasterio), resampled to your grid |
| `rainfall_mm_24h` | Rain gauge / radar / IDF design-storm value |
| `drainage_capacity_mmhr` | Stormwater network shapefile (pipe density/capacity per cell) |
| `land_use`, `imperviousness_pct` | Land-cover raster or zoning shapefile |
| `soil_type` (or `curve_number` directly) | Soil survey shapefile (hydrologic soil group) |
| `x`, `y` | Grid indices, or swap for real lon/lat and plot with `geopandas`/`contextily` instead of `imshow` |

If `geopandas` and `rasterio` are available in your environment, replace the
`imshow`-based plots in `risk_map.py` with a proper `GeoDataFrame.plot(...)`
call using the same `flood_probability` / `predicted_runoff_mm` columns —
no changes needed to the model training code.

## Notes / limitations
- The bundled dataset is **synthetic** (physically-informed but not real
  observations) since no rainfall/GIS files were supplied — swap in real data
  as above before using this for actual planning decisions.
- This is a data-driven risk screening tool, not a substitute for a calibrated
  hydraulic/hydrologic model (e.g. HEC-RAS, SWMM) for engineering design.
