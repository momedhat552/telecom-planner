import os
import osmnx as ox
import matplotlib.pyplot as plt

NAME = "Berlin, Germany"          # label used in all output file names
CENTER = (52.52, 13.41)     # (lat, lon) starting guess for New Cairo, check on openstreetmap.org
HALF_SIDE_M = 1500          # 1500 m each side of centre = 3 x 3 km square
UTM_CRS = "EPSG:32633"      # metric coordinates (UTM zone 36N, covers both Giza and New Cairo)

os.makedirs("data", exist_ok=True)

# 1. Buildings
buildings = ox.features_from_point(CENTER, tags={"building": True}, dist=HALF_SIDE_M)
buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
buildings = buildings[["geometry"]].reset_index(drop=True)

# 2. Project to metres and compute footprint area
buildings = buildings.to_crs(UTM_CRS)
buildings["area_m2"] = buildings.geometry.area

# 3. Road network
roads = ox.graph_from_point(CENTER, dist=HALF_SIDE_M, network_type="drive")
roads = ox.project_graph(roads, to_crs=UTM_CRS)

# 4. Save for later phases
buildings.to_file(f"data/{NAME}_buildings.gpkg", driver="GPKG")
ox.save_graphml(roads, f"data/{NAME}_roads.graphml")

# 5. Quick stats
total_m2 = buildings["area_m2"].sum()
area_m2 = (2 * HALF_SIDE_M) ** 2
print("Buildings:", len(buildings))
print("Total footprint (km2):", round(total_m2 / 1e6, 3))
print("Median building area (m2):", round(buildings["area_m2"].median(), 1))
print("Built-up fraction of area:", round(total_m2 / area_m2, 3))

# 6. Plot
fig, ax = plt.subplots(figsize=(9, 9))
buildings.plot(ax=ax, color="#888888", linewidth=0)
ox.plot_graph(roads, ax=ax, node_size=0, edge_color="#d9534f",
              edge_linewidth=0.6, show=False, close=False)
ax.set_title(f"{NAME}: OSM buildings and roads")
plt.savefig(f"data/{NAME}_phase1_map.png", dpi=150)
plt.show()