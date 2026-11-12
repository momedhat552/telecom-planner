import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

NAME = "Berlin, Germany"          # label used in all output file names
CENTER = (52.52, 13.41)     # (lat, lon) starting guess for New Cairo, check on openstreetmap.org
HALF_SIDE_M = 1500          # 1500 m each side of centre = 3 x 3 km square
CELL_M = 100                # grid square size in metres

# 1. Load the buildings saved by Phase 1 (already in metres)
buildings = gpd.read_file(f"data/{NAME}_buildings.gpkg")
crs = buildings.crs

# 2. Convert the centre point from lat/lon to the same metre system
centre = gpd.GeoSeries([Point(CENTER[1], CENTER[0])], crs="EPSG:4326").to_crs(crs).iloc[0]
cx, cy = centre.x, centre.y

# 3. Build the chessboard: one square (box) per grid cell
xs = np.arange(cx - HALF_SIDE_M, cx + HALF_SIDE_M, CELL_M)
ys = np.arange(cy - HALF_SIDE_M, cy + HALF_SIDE_M, CELL_M)
cells = [box(x, y, x + CELL_M, y + CELL_M) for x in xs for y in ys]
grid = gpd.GeoDataFrame({"geometry": cells}, crs=crs)
grid["cell_id"] = grid.index

# 4. Cut every building at the square borders
pieces = gpd.overlay(grid, buildings[["geometry"]], how="intersection")
pieces["piece_m2"] = pieces.geometry.area

# 5. Add up the pieces inside each square
built = pieces.groupby("cell_id")["piece_m2"].sum()
grid["built_m2"] = grid["cell_id"].map(built).fillna(0)

# 6. Sanity checks
print("Grid cells:", len(grid))
print("Empty cells (no building):", int((grid["built_m2"] == 0).sum()))
print("Max built area in one cell (m2):", round(grid["built_m2"].max(), 1), "(limit is 10000)")
print("Total in grid (km2):", round(grid["built_m2"].sum() / 1e6, 3))
print("Total from Phase 1 (km2):", round(buildings["area_m2"].sum() / 1e6, 3))

# 7. Save for later phases
grid.to_file(f"data/{NAME}_demand_grid.gpkg", driver="GPKG")

# 8. Heatmap
fig, ax = plt.subplots(figsize=(9, 9))
grid.plot(column="built_m2", cmap="inferno", legend=True, ax=ax,
          legend_kwds={"label": "Building footprint per 100 x 100 m cell (m2)", "shrink": 0.7})
ax.set_title(f"{NAME}: demand proxy (building footprint per cell)")
ax.set_axis_off()
plt.savefig(f"data/{NAME}_phase2_heatmap.png", dpi=150)
plt.show()