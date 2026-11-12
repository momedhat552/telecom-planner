import numpy as np
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point

NAME = "Berlin, Germany"             # must match the file names in data/

# --- Radio assumptions (same as before, all still to verify) ---
FC_GHZ = 3.5
H_BS = 25.0
H_UT = 1.5
H_E = 1.0
TX_POWER_DBM = 46           # per sector
N_SUBCARRIERS = 3276
ANT_GAIN_DBI = 15           # antenna gain at the centre of the beam
THRESHOLD_DBM = -100

# --- Sector antenna (3GPP horizontal pattern) ---
BEAMWIDTH_DEG = 65
FRONT_BACK_DB = 30

# --- Planning and frequency settings ---
REPLAN = True               # True: choose towers again with real sectors; False: keep Phase 4 towers
TARGET_DEMAND = 0.95
MAX_SITES = 60
OFFSETS = range(0, 120, 15)  # rotations of the 3-sector pattern to try (degrees)
INTERF_MARGIN_DB = 10       # a sector disturbs a square if it is within this many dB of the serving one
INTERF_FLOOR_DBM = -110     # ...and stronger than this
MIN_CELLS = 2               # squares needed before two sectors count as a conflict


def los_probability(d2d):
    d = np.maximum(d2d, 1.0)
    p = 18 / d + np.exp(-d / 63) * (1 - 18 / d)
    return np.where(d2d <= 18, 1.0, p)


def pl_los(d2d, d3d):
    d_bp = 4 * (H_BS - H_E) * (H_UT - H_E) * FC_GHZ * 1e9 / 3e8
    pl1 = 28.0 + 22 * np.log10(d3d) + 20 * np.log10(FC_GHZ)
    pl2 = (28.0 + 40 * np.log10(d3d) + 20 * np.log10(FC_GHZ)
           - 9 * np.log10(d_bp ** 2 + (H_BS - H_UT) ** 2))
    return np.where(d2d <= d_bp, pl1, pl2)


def pl_nlos(d2d, d3d):
    pl_n = (13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(FC_GHZ)
            - 0.6 * (H_UT - 1.5))
    return np.maximum(pl_los(d2d, d3d), pl_n)


def sector_gain_db(direction_deg, boresight_deg):
    """Antenna gain towards a direction: full gain in front, less and less to the sides and back."""
    diff = (direction_deg - boresight_deg + 180) % 360 - 180      # angle off the beam centre
    return ANT_GAIN_DBI - np.minimum(12 * (diff / BEAMWIDTH_DEG) ** 2, FRONT_BACK_DB)


# 1. Load the grid and the Phase 4 towers
grid = gpd.read_file(f"data/{NAME}_demand_grid.gpkg")
crs = grid.crs
cent = grid.geometry.centroid
cx, cy = cent.x.values, cent.y.values
demand = grid["built_m2"].values
total_demand = demand.sum()
n_cells = len(grid)

sites4 = gpd.read_file(f"data/{NAME}_sites_greedy.gpkg")
sx4, sy4 = sites4.geometry.x.values, sites4.geometry.y.values


def share(mask):
    return demand[mask].sum() / total_demand


def prepare(sx, sy):
    """For each tower and each square: signal before the antenna (dBm) and direction to the square."""
    dx = cx[None, :] - np.asarray(sx)[:, None]      # east distance
    dy = cy[None, :] - np.asarray(sy)[:, None]      # north distance
    d2d = np.maximum(np.hypot(dx, dy), 10.0)
    d3d = np.sqrt(d2d ** 2 + (H_BS - H_UT) ** 2)
    p = los_probability(d2d)
    pl = p * pl_los(d2d, d3d) + (1 - p) * pl_nlos(d2d, d3d)
    base = TX_POWER_DBM - 10 * np.log10(N_SUBCARRIERS) - pl
    direction = np.degrees(np.arctan2(dx, dy))      # compass direction: 0 = north, 90 = east
    return base, direction


def sector_rsrp(base, direction, az0):
    """RSRP of the 3 sectors of every tower at every square: shape (towers, 3, squares)."""
    return np.stack([base + sector_gain_db(direction, az0 + 120 * k) for k in range(3)], axis=1)


def greedy(cover_sites, target, max_sites):
    """Add the tower that reaches the most not-yet-covered demand, until the target is met."""
    chosen, covered = [], np.zeros(n_cells, dtype=bool)
    while len(chosen) < max_sites:
        left = ~covered
        gain = cover_sites[:, left].astype(float) @ demand[left]
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        chosen.append(best)
        covered |= cover_sites[best]
        if share(covered) >= target:
            break
    return chosen, covered


# 2. What happens to the Phase 4 towers when the antennas are realistic?
base4, dir4 = prepare(sx4, sy4)
omni = (base4 + ANT_GAIN_DBI >= THRESHOLD_DBM).any(axis=0)
print(f"Phase 4 towers, antenna as before (full gain in every direction): {share(omni):.1%} of demand")
best4 = (-1.0, 0)
for off in OFFSETS:
    rs = sector_rsrp(base4, dir4, off)
    f = share((rs >= THRESHOLD_DBM).any(axis=(0, 1)))
    if f > best4[0]:
        best4 = (f, off)
print(f"Same {len(sx4)} towers with 3 real sectors: {best4[0]:.1%} of demand (best rotation {best4[1]} deg)")

# 3. Plan the towers again with real sectors (or keep the Phase 4 towers)
if REPLAN:
    base_c, dir_c = prepare(cx, cy)                  # every square middle is a possible tower spot
    options = []
    for off in OFFSETS:
        rs = sector_rsrp(base_c, dir_c, off)
        cov_c = (rs >= THRESHOLD_DBM).any(axis=1)
        chosen, cov = greedy(cov_c, TARGET_DEMAND, MAX_SITES)
        options.append((len(chosen), -share(cov), off, chosen))
        print(f"  rotation {off:>3} deg: {len(chosen)} towers, {share(cov):.1%} of demand")
    n_best, neg_frac, az0, chosen = min(options, key=lambda r: (r[0], r[1]))
    fx, fy = cx[chosen], cy[chosen]
    print(f"Replanned: {n_best} towers, rotation {az0} deg")
else:
    fx, fy, az0 = sx4, sy4, best4[1]

# 4. Final network: signal of every sector at every square
n_sites = len(fx)
n_sec = 3 * n_sites
base_f, dir_f = prepare(fx, fy)
rsrp = sector_rsrp(base_f, dir_f, az0)
flat = rsrp.reshape(n_sec, n_cells)         # sector number = tower * 3 + sector within tower
serving = flat.argmax(axis=0)               # the strongest sector at each square
best = flat.max(axis=0)
covered = best >= THRESHOLD_DBM
print()
print(f"Final network: {n_sites} towers, {n_sec} sectors")
print(f"  Demand with signal: {share(covered):.1%} | squares with signal: {covered.mean():.1%}")

# 5. Conflict graph: which sectors must NOT share a frequency
G = nx.Graph()
G.add_nodes_from(range(n_sec))
for site in range(n_sites):                 # the 3 sectors of one tower always differ
    for a in range(3):
        for b in range(a + 1, 3):
            G.add_edge(3 * site + a, 3 * site + b)

hits = np.zeros((n_sec, n_sec), dtype=int)
for cell in np.where(covered)[0]:
    s = serving[cell]
    floor = max(best[cell] - INTERF_MARGIN_DB, INTERF_FLOOR_DBM)
    for other in np.where(flat[:, cell] >= floor)[0]:
        if other != s:
            hits[s, other] += 1             # "other" would disturb this square
for a in range(n_sec):
    for b in range(a + 1, n_sec):
        if hits[a, b] + hits[b, a] >= MIN_CELLS:
            G.add_edge(a, b)

print(f"  Conflict graph: {G.number_of_edges()} conflicts, "
      f"most conflicts for one sector: {max(dict(G.degree).values())}")

# 6. Colour the graph: one colour = one frequency group
strategies = ["largest_first", "smallest_last", "saturation_largest_first", "connected_sequential_bfs"]
runs = {s: nx.greedy_color(G, strategy=s) for s in strategies}
for s, c in runs.items():
    print(f"  {s}: {len(set(c.values()))} frequency groups")
best_strategy = min(runs, key=lambda s: len(set(runs[s].values())))
colors = runs[best_strategy]
n_colors = len(set(colors.values()))
clique = max(len(c) for c in nx.find_cliques(G))
valid = all(colors[a] != colors[b] for a, b in G.edges)
print(f"Frequency groups used: {n_colors} (strategy {best_strategy})")
print(f"Minimum possible (largest group of mutually conflicting sectors): at least {clique}")
print("Check, no conflicting sectors share a group:", valid)

# 7. Put the results on the grid and save
freq_of_sector = np.array([colors[s] for s in range(n_sec)])
grid["serving_sector"] = serving
grid["best_rsrp_dbm"] = best
grid["covered"] = covered
grid["freq_group"] = np.where(covered, freq_of_sector[serving], -1)
grid.to_file(f"data/{NAME}_coverage_sectors.gpkg", driver="GPKG")

rows = []
for s in range(n_sec):
    site, k = divmod(s, 3)
    rows.append({"sector_id": s, "site_id": site,
                 "azimuth_deg": (az0 + 120 * k) % 360, "freq_group": int(colors[s])})
sectors = gpd.GeoDataFrame(rows, geometry=[Point(fx[s // 3], fy[s // 3]) for s in range(n_sec)], crs=crs)
sectors.to_file(f"data/{NAME}_sectors.gpkg", driver="GPKG")

# 8. Map: each square takes the colour of the frequency group of its strongest sector
fig, ax = plt.subplots(figsize=(9, 9))
if (~covered).any():
    grid[~grid["covered"]].plot(ax=ax, color="lightgrey")
grid[grid["covered"]].plot(column="freq_group", categorical=True, cmap="tab20", legend=True, ax=ax,
                           legend_kwds={"title": "Frequency group"})
for s in range(n_sec):
    site, k = divmod(s, 3)
    a = np.radians(az0 + 120 * k)
    ax.plot([fx[site], fx[site] + 200 * np.sin(a)], [fy[site], fy[site] + 200 * np.cos(a)],
            color="black", linewidth=1)
ax.plot(fx, fy, "k^", markersize=8, markeredgecolor="white")
ax.set_title(f"{NAME}: {n_sites} towers, {n_sec} sectors, {n_colors} frequency groups")
ax.set_axis_off()
plt.savefig(f"data/{NAME}_phase5_frequency_map.png", dpi=150)
plt.show()