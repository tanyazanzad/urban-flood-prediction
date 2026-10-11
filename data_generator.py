"""
data_generator.py
------------------
Generates a physically-informed SYNTHETIC dataset representing an urban
catchment, since no real rainfall/GIS files were supplied.

Replace this module with real data loaders when you have actual inputs:
  - rainfall        -> gauge/radar CSV or NetCDF
  - elevation       -> DEM raster (read with rasterio) -> flatten to grid
  - drainage        -> stormwater network shapefile (density/capacity per cell)
  - land_use        -> zoning / land-cover raster or shapefile
  - soil            -> soil survey shapefile (hydrologic soil group)

The grid abstraction (rows of x, y cells) is the same shape you'd get after
rasterizing real GIS layers to a common resolution with geopandas/rasterio,
so the rest of the pipeline (features.py, model.py, risk_map.py) does not
need to change when you swap in real data.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

LAND_USE_TYPES = ["water_body", "green_space", "residential", "commercial", "industrial", "road"]
SOIL_TYPES = ["sand", "loam", "silt", "clay"]  # roughly increasing runoff potential (HSG A->D)

# Curve Numbers (SCS-CN, AMC-II) by [land_use][soil_type], approx standard values
CN_TABLE = {
    "water_body":   {"sand": 100, "loam": 100, "silt": 100, "clay": 100},
    "green_space":  {"sand": 39,  "loam": 61,  "silt": 74,  "clay": 80},
    "residential":  {"sand": 61,  "loam": 75,  "silt": 83,  "clay": 87},
    "commercial":   {"sand": 89,  "loam": 92,  "silt": 94,  "clay": 95},
    "industrial":   {"sand": 81,  "loam": 88,  "silt": 91,  "clay": 93},
    "road":         {"sand": 98,  "loam": 98,  "silt": 98,  "clay": 98},
}

IMPERVIOUSNESS_RANGE = {
    "water_body": (0, 0), "green_space": (5, 20), "residential": (35, 60),
    "commercial": (65, 85), "industrial": (70, 90), "road": (90, 98),
}


def _make_elevation_field(n):
    """Elevation surface: a river valley on one edge + rolling terrain + noise."""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y)
    valley = 40 * xx  # low elevation near x=0 (the "river" edge), rising inland
    hills = 15 * np.sin(3 * yy * np.pi) * np.cos(2 * xx * np.pi)
    noise = RNG.normal(0, 3, size=(n, n))
    elev = 25 + valley + hills + noise
    elev = np.clip(elev, 5, None)
    return xx, yy, elev


def _assign_land_use(xx, yy, elev):
    """Zone land use loosely by distance from the river edge and elevation."""
    n = xx.shape[0]
    out = np.empty((n, n), dtype=object)
    dist_from_river = xx  # 0 = at river, 1 = far inland
    for i in range(n):
        for j in range(n):
            d = dist_from_river[i, j]
            e = elev[i, j]
            r = RNG.random()
            if d < 0.04 and e < 20:
                out[i, j] = "water_body"
            elif d < 0.15:
                probs = [0.02, 0.45, 0.35, 0.08, 0.05, 0.05]
            elif d < 0.45:
                probs = [0.0, 0.10, 0.45, 0.20, 0.15, 0.10]
            elif d < 0.75:
                probs = [0.0, 0.15, 0.30, 0.30, 0.15, 0.10]
            else:
                probs = [0.0, 0.20, 0.20, 0.15, 0.35, 0.10]
            if out[i, j] is None:
                out[i, j] = RNG.choice(LAND_USE_TYPES, p=np.array(probs) / np.sum(probs))
    return out


def _assign_soil(n):
    """Soil type field, spatially smoothed (soils are spatially autocorrelated)."""
    raw = RNG.integers(0, len(SOIL_TYPES), size=(n, n)).astype(float)
    # crude smoothing pass
    smooth = raw.copy()
    for _ in range(2):
        smooth = (
            smooth
            + np.roll(smooth, 1, axis=0) + np.roll(smooth, -1, axis=0)
            + np.roll(smooth, 1, axis=1) + np.roll(smooth, -1, axis=1)
        ) / 5
    idx = np.clip(np.round(smooth).astype(int), 0, len(SOIL_TYPES) - 1)
    return np.vectorize(lambda k: SOIL_TYPES[k])(idx)


def _drainage_capacity(xx, yy, land_use):
    """
    Engineered drainage capacity (mm/hr the stormwater network can remove),
    higher in newer/commercial districts, lower in dense old residential core.
    """
    n = xx.shape[0]
    base = 20 + 15 * np.sin(4 * xx * np.pi + 1) * np.cos(3 * yy * np.pi)
    base += RNG.normal(0, 4, size=(n, n))
    lu_bonus = np.vectorize({
        "water_body": 0, "green_space": 5, "residential": 0,
        "commercial": 15, "industrial": 10, "road": 20,
    }.get)(land_use).astype(float)
    cap = np.clip(base + lu_bonus, 2, 80)
    return cap


def generate_catchment(grid_size: int = 40, storms_per_cell: int = 6, seed: int = 42) -> pd.DataFrame:
    """
    Build a (grid_size x grid_size) synthetic catchment and simulate several
    storm events per cell, returning a flat DataFrame ready for ML.
    """
    global RNG
    RNG = np.random.default_rng(seed)
    n = grid_size
    xx, yy, elev = _make_elevation_field(n)
    land_use = _assign_land_use(xx, yy, elev)
    soil = _assign_soil(n)
    drainage_cap = _drainage_capacity(xx, yy, land_use)

    # slope proxy from local elevation gradient (m per 100 m cell)
    dzdx = np.gradient(elev, axis=1)
    dzdy = np.gradient(elev, axis=0)
    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)

    rows = []
    storm_intensities = [15, 30, 55, 80, 110, 150]  # mm/24h: light -> extreme
    for i in range(n):
        for j in range(n):
            lu = land_use[i, j]
            so = soil[i, j]
            cn = CN_TABLE[lu][so]
            imp_lo, imp_hi = IMPERVIOUSNESS_RANGE[lu]
            imperviousness = RNG.uniform(imp_lo, imp_hi) if imp_hi > imp_lo else imp_lo

            for _ in range(storms_per_cell):
                rainfall = RNG.choice(storm_intensities) * RNG.uniform(0.85, 1.15)
                rainfall = max(rainfall, 0)

                # SCS-CN runoff (physically grounded target, mm)
                S = (25400.0 / cn) - 254.0
                Ia = 0.2 * S
                if rainfall > Ia:
                    runoff = (rainfall - Ia) ** 2 / (rainfall - Ia + S)
                else:
                    runoff = 0.0
                runoff = max(0.0, runoff + RNG.normal(0, runoff * 0.05 + 0.5))

                rows.append({
                    "x": j, "y": i,
                    "elevation_m": elev[i, j],
                    "slope": slope[i, j],
                    "land_use": lu,
                    "soil_type": so,
                    "imperviousness_pct": imperviousness,
                    "drainage_capacity_mmhr": drainage_cap[i, j],
                    "rainfall_mm_24h": rainfall,
                    "runoff_mm": runoff,
                    "curve_number": cn,
                })
    df = pd.DataFrame(rows)

    # --- Physically-informed flood risk score -> stochastic binary label ---
    elev_norm = (df["elevation_m"].max() - df["elevation_m"]) / (df["elevation_m"].max() - df["elevation_m"].min())
    slope_norm = 1 - (df["slope"] - df["slope"].min()) / (df["slope"].max() - df["slope"].min() + 1e-9)  # flatter -> higher risk
    excess = np.clip(df["runoff_mm"] - df["drainage_capacity_mmhr"], 0, None)
    excess_norm = excess / (excess.max() + 1e-9)

    risk_score = (
        0.45 * excess_norm
        + 0.25 * elev_norm
        + 0.15 * slope_norm
        + 0.15 * (df["imperviousness_pct"] / 100.0)
    )
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    df["flood_risk_score"] = risk_score

    flood_prob_latent = 1 / (1 + np.exp(-10 * (risk_score - 0.5)))  # logistic sharpening
    df["flood_occurred"] = (RNG.random(len(df)) < flood_prob_latent).astype(int)

    return df


if __name__ == "__main__":
    data = generate_catchment()
    print(data.head())
    print("\nShape:", data.shape)
    print("Flood rate:", data["flood_occurred"].mean().round(3))
