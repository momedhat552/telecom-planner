import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

NAME = "Berlin, Germany"             # must match Phase 1 and 2
CENTER = (52.52, 13.41)     # (lat, lon), the site goes here

# --- Radio assumptions (all still to verify, copy them into ASSUMPTIONS.md) ---
FC_GHZ = 3.5                # carrier frequency, band n78
H_BS = 25.0                 # base station antenna height (m)
H_UT = 1.5                  # user device height (m)
H_E = 1.0                   # effective environment height used by the 3GPP breakpoint formula (m)
TX_POWER_DBM = 46           # transmit power of the site
N_SUBCARRIERS = 3276        # 100 MHz at 30 kHz spacing = 273 resource blocks x 12
ANT_GAIN_DBI = 15           # antenna gain, my assumption, no sector pattern yet
THRESHOLD_DBM = -100        # covered = RSRP at or above this


def los_probability(d2d):
    """Chance that the user has a clear line of sight (3GPP UMa, user height <= 13 m)."""
    d = np.maximum(d2d, 1.0)
    p = 18 / d + np.exp(-d / 63) * (1 - 18 / d)
    return np.where(d2d <= 18, 1.0, p)


def pl_los(d2d, d3d):
    """Path loss in dB with line of sight (3GPP TR 38.901, UMa-LOS)."""
    d_bp = 4 * (H_BS - H_E) * (H_UT - H_E) * FC_GHZ * 1e9 / 3e8   # breakpoint distance
    pl1 = 28.0 + 22 * np.log10(d3d) + 20 * np.log10(FC_GHZ)
    pl2 = (28.0 + 40 * np.log10(d3d) + 20 * np.log10(FC_GHZ)
           - 9 * np.log10(d_bp ** 2 + (H_BS - H_UT) ** 2))
    return np.where(d2d <= d_bp, pl1, pl2)


def pl_nlos(d2d, d3d):
    """Path loss in dB without line of sight (3GPP TR 38.901, UMa-NLOS)."""
    pl_n = (13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(FC_GHZ)
            - 0.6 * (H_UT - 1.5))
    return np.maximum(pl_los(d2d, d3d), pl_n)


# 1. Load the demand grid from Phase 2
grid = gpd.read_file(f"data/{NAME}_demand_grid.gpkg")
crs = grid.crs

# 2. Put the site at CENTER, converted to the same metre system
site = gpd.GeoSeries([Point(CENTER[1], CENTER[0])], crs="EPSG:4326").to_crs(crs).iloc[0]

# 3. Distance from the site to the middle of every grid cell
cent = grid.geometry.centroid
d2d = np.hypot(cent.x.values - site.x, cent.y.values - site.y)
d2d = np.maximum(d2d, 10.0)                       # the model is valid from 10 m
d3d = np.sqrt(d2d ** 2 + (H_BS - H_UT) ** 2)      # includes the height difference

# 4. Path loss: LOS, NLOS, and a mix weighted by the LOS probability
p_los = los_probability(d2d)
pl_l = pl_los(d2d, d3d)
pl_n = pl_nlos(d2d, d3d)
pl_exp = p_los * pl_l + (1 - p_los) * pl_n

# 5. Signal strength (RSRP) per cell
offset = TX_POWER_DBM - 10 * np.log10(N_SUBCARRIERS) + ANT_GAIN_DBI
grid["rsrp_dbm"] = offset - pl_exp
grid["rsrp_nlos_dbm"] = offset - pl_n
grid["covered"] = grid["rsrp_dbm"] >= THRESHOLD_DBM
grid["covered_nlos"] = grid["rsrp_nlos_dbm"] >= THRESHOLD_DBM

# 6. Numbers
total_demand = grid["built_m2"].sum()
for label, col in [("Expected (LOS/NLOS mix)", "covered"),
                   ("Worst case (all NLOS)", "covered_nlos")]:
    cov = grid[col]
    print(label)
    print("  Cells covered:", int(cov.sum()), "of", len(grid), f"({cov.mean():.1%})")
    print("  Demand covered:", f"{grid.loc[cov, 'built_m2'].sum() / total_demand:.1%}")
    print("  Furthest covered cell (m):", round(float(d2d[cov.values].max())) if cov.any() else 0)

# 7. Save for later phases
grid.to_file(f"data/{NAME}_coverage_one_site.gpkg", driver="GPKG")

# 8. Maps: signal strength and covered / not covered
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
grid.plot(column="rsrp_dbm", cmap="viridis", legend=True, ax=axes[0],
          legend_kwds={"label": "RSRP (dBm)", "shrink": 0.7})
axes[0].set_title("RSRP from one site (expected path loss)")
grid.plot(column="covered", categorical=True, cmap="RdYlGn", legend=True, ax=axes[1])
axes[1].set_title(f"Covered (RSRP >= {THRESHOLD_DBM} dBm)")
for ax in axes:
    ax.plot(site.x, site.y, marker="*", color="white", markeredgecolor="black", markersize=18)
    ax.set_axis_off()
plt.savefig(f"data/{NAME}_phase3_coverage.png", dpi=150)
plt.show()

# 9. Sanity plot: RSRP against distance
d = np.linspace(20, 1500, 300)
d3 = np.sqrt(d ** 2 + (H_BS - H_UT) ** 2)
p = los_probability(d)
plt.figure(figsize=(8, 5))
plt.plot(d, offset - pl_los(d, d3), label="LOS")
plt.plot(d, offset - pl_nlos(d, d3), label="NLOS")
plt.plot(d, offset - (p * pl_los(d, d3) + (1 - p) * pl_nlos(d, d3)), label="Expected mix")
plt.axhline(THRESHOLD_DBM, color="red", linestyle="--", label="Coverage threshold")
plt.xlabel("Distance from site (m)")
plt.ylabel("RSRP (dBm)")
plt.title("Signal strength vs distance, 3.5 GHz UMa")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"data/{NAME}_phase3_rsrp_vs_distance.png", dpi=150)
plt.show()