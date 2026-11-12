import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

NAME = "Berlin, Germany"             # must match the file names in data/

# --- Radio assumptions (same as Phase 3 and 4) ---
FC_GHZ = 3.5
H_BS = 25.0
H_UT = 1.5
H_E = 1.0
TX_POWER_DBM = 46
N_SUBCARRIERS = 3276
ANT_GAIN_DBI = 15
THRESHOLD_DBM = -100

# --- Comparison settings ---
MAX_SITES = 30              # compare up to this many towers
N_REPORT = 19               # tower count for the maps and the saved sites
SPACINGS = range(300, 1501, 25)   # honeycomb spacings to try (m)
ANGLES = [0, 15, 30, 45]          # honeycomb rotations to try (degrees)
N_SHIFTS = 6                      # slide positions to try in x and in y

offset = TX_POWER_DBM - 10 * np.log10(N_SUBCARRIERS) + ANT_GAIN_DBI


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


# 1. Load the grid
grid = gpd.read_file(f"data/{NAME}_demand_grid.gpkg")
crs = grid.crs
cent = grid.geometry.centroid
cx = cent.x.values
cy = cent.y.values
demand = grid["built_m2"].values
total_demand = demand.sum()
n_cells = len(grid)

xmin, ymin, xmax, ymax = grid.total_bounds
cx0, cy0 = (xmin + xmax) / 2, (ymin + ymax) / 2
half = (xmax - xmin) / 2


def covers(sx, sy):
    """Row = one tower position, column = one map square, True = square has signal."""
    d2d = np.hypot(np.asarray(sx)[:, None] - cx[None, :], np.asarray(sy)[:, None] - cy[None, :])
    d2d = np.maximum(d2d, 10.0)
    d3d = np.sqrt(d2d ** 2 + (H_BS - H_UT) ** 2)
    p = los_probability(d2d)
    pl = p * pl_los(d2d, d3d) + (1 - p) * pl_nlos(d2d, d3d)
    return (offset - pl) >= THRESHOLD_DBM


def hex_lattice(spacing, ox, oy, angle_deg):
    """Honeycomb of towers: slid by (ox, oy), turned by angle, kept only inside the area."""
    dy = spacing * np.sqrt(3) / 2
    n = int(2 * half / dy) + 4
    r, k = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    x = k * spacing + (spacing / 2) * (r % 2)       # every second row sticks out half a step
    y = r * dy
    pts = np.column_stack([x.ravel(), y.ravel()])
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = pts @ rot.T + np.array([cx0 + ox, cy0 + oy])
    inside = (np.abs(pts[:, 0] - cx0) <= half) & (np.abs(pts[:, 1] - cy0) <= half)
    return pts[inside]


# 2. Greedy: same as Phase 4, but keep going up to MAX_SITES and record every step
cover = covers(cx, cy)
chosen, covered = [], np.zeros(n_cells, dtype=bool)
greedy_demand, greedy_cells = [], []
for _ in range(MAX_SITES):
    left = ~covered
    gain = cover[:, left].astype(float) @ demand[left]
    best = int(np.argmax(gain))
    if gain[best] <= 0:
        break
    chosen.append(best)
    covered |= cover[best]
    greedy_demand.append(demand[covered].sum() / total_demand)
    greedy_cells.append(covered.mean())

# 3. Tuned honeycomb: try many layouts, keep the best one for each tower count
best_hex = {}               # tower count -> (demand share, cell share, points, spacing, angle)
tried = 0
for s in SPACINGS:
    for ang in ANGLES:
        for ox in np.linspace(0, s, N_SHIFTS, endpoint=False):
            for oy in np.linspace(0, s, N_SHIFTS, endpoint=False):
                pts = hex_lattice(s, ox, oy, ang)
                n = len(pts)
                if n < 5 or n > MAX_SITES:
                    continue
                tried += 1
                cov = covers(pts[:, 0], pts[:, 1]).any(axis=0)
                frac = demand[cov].sum() / total_demand
                if n not in best_hex or frac > best_hex[n][0]:
                    best_hex[n] = (frac, cov.mean(), pts, s, ang)
print("Honeycomb layouts tried:", tried)

# 4. Table
print()
print("Towers | Greedy demand | Hex demand | Greedy squares | Hex squares")
for n in (10, 15, 19, 25):
    if n <= len(greedy_demand) and n in best_hex:
        h = best_hex[n]
        print(f"{n:>6} | {greedy_demand[n-1]:>13.1%} | {h[0]:>10.1%} | "
              f"{greedy_cells[n-1]:>14.1%} | {h[1]:>11.1%}")
    else:
        print(f"{n:>6} | no result for this tower count")

# 5. Curves: demand with signal against number of towers
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(greedy_demand) + 1), [g * 100 for g in greedy_demand],
         marker="o", label="Greedy")
ns = sorted(best_hex)
plt.plot(ns, [best_hex[n][0] * 100 for n in ns], marker="s", label="Best honeycomb (tuned)")
plt.xlabel("Number of towers")
plt.ylabel("Demand with signal (%)")
plt.title("Greedy vs tuned honeycomb")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"data/{NAME}_phase4b_curves.png", dpi=150)
plt.show()

# 6. Maps and saved sites for N_REPORT towers
if N_REPORT in best_hex and N_REPORT <= len(chosen):
    h = best_hex[N_REPORT]
    hpts = h[2]
    grid["cov_greedy"] = cover[chosen[:N_REPORT]].any(axis=0)
    grid["cov_hex"] = covers(hpts[:, 0], hpts[:, 1]).any(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    panels = [(axes[0], "cov_greedy", f"Greedy ({N_REPORT} towers)",
               cx[chosen[:N_REPORT]], cy[chosen[:N_REPORT]]),
              (axes[1], "cov_hex",
               f"Tuned honeycomb ({N_REPORT} towers, spacing {h[3]} m, turned {h[4]} deg)",
               hpts[:, 0], hpts[:, 1])]
    for ax, col, title, xs, ys in panels:
        grid.plot(column=col, categorical=True, cmap="RdYlGn", ax=ax, legend=True)
        ax.plot(xs, ys, "k^", markersize=8, markeredgecolor="white")
        ax.set_title(title)
        ax.set_axis_off()
    plt.savefig(f"data/{NAME}_phase4b_maps.png", dpi=150)
    plt.show()

    sites = gpd.GeoDataFrame({"site_id": range(len(hpts))},
                             geometry=[Point(x, y) for x, y in hpts], crs=crs)
    sites.to_file(f"data/{NAME}_sites_hex_tuned.gpkg", driver="GPKG")

    # The greedy towers at the same count, for Phase 5 (it replans them with real sectors).
    greedy_sites = gpd.GeoDataFrame(
        {"site_id": range(N_REPORT)},
        geometry=[Point(x, y) for x, y in zip(cx[chosen[:N_REPORT]], cy[chosen[:N_REPORT]])],
        crs=crs)
    greedy_sites.to_file(f"data/{NAME}_sites_greedy.gpkg", driver="GPKG")
else:
    print("No layout found with exactly", N_REPORT, "towers, change N_REPORT.")