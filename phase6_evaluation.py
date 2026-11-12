import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

NAME = "Berlin, Germany"    # must match the file names in data/

# --- Radio assumptions (baseline, all still to verify) ---
FC_GHZ = 3.5
H_BS = 25.0
H_UT = 1.5
H_E = 1.0
TX_POWER_DBM = 46           # per sector
N_SUBCARRIERS = 3276        # 100 MHz at 30 kHz spacing
BASE_GAIN_DBI = 15.0        # baseline antenna gain (centre of the beam)
BASE_THRESHOLD_DBM = -100.0 # baseline: covered = RSRP at or above this
BASE_MARGIN_DB = 10.0       # baseline: conflict margin for frequency planning

# --- Sector antenna (3GPP horizontal pattern) ---
BEAMWIDTH_DEG = 65
FRONT_BACK_DB = 30

# --- Planning settings (same as Phase 5) ---
TARGET_DEMAND = 0.95
MAX_SITES = 80
OFFSETS = list(range(0, 120, 15))   # rotations of the 3-sector pattern to try (degrees)
MIN_CELLS = 2

# --- Experiment 1 settings: greedy against the tuned honeycomb ---
COMPARE_N = [15, 20, 26, 30]        # tower counts shown in the printed table
MAX_N = 40                          # compare every tower count from 5 up to this
SPACINGS = range(450, 1501, 25)     # honeycomb spacings to try (m)
ANGLES = [0, 15, 30, 45]            # honeycomb rotations to try (degrees)
N_SHIFTS = 5                        # slide positions to try in x and in y


# ---------------------------------------------------------------- radio model
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


def sector_gain_db(direction_deg, boresight_deg, gain_dbi):
    diff = (direction_deg - boresight_deg + 180) % 360 - 180
    return gain_dbi - np.minimum(12 * (diff / BEAMWIDTH_DEG) ** 2, FRONT_BACK_DB)


# ------------------------------------------------------------------ load data
grid = gpd.read_file(f"data/{NAME}_demand_grid.gpkg")
cent = grid.geometry.centroid
cx, cy = cent.x.values, cent.y.values
demand = grid["built_m2"].values
total_demand = demand.sum()
n_cells = len(grid)
xmin, ymin, xmax, ymax = grid.total_bounds
cx0, cy0 = (xmin + xmax) / 2, (ymin + ymax) / 2
half = (xmax - xmin) / 2


def share(mask):
    return demand[mask].sum() / total_demand


def prepare(sx, sy):
    """Signal before the antenna (dBm) and compass direction, for each tower and each square."""
    dx = cx[None, :] - np.asarray(sx)[:, None]
    dy = cy[None, :] - np.asarray(sy)[:, None]
    d2d = np.maximum(np.hypot(dx, dy), 10.0)
    d3d = np.sqrt(d2d ** 2 + (H_BS - H_UT) ** 2)
    p = los_probability(d2d)
    pl = p * pl_los(d2d, d3d) + (1 - p) * pl_nlos(d2d, d3d)
    base = TX_POWER_DBM - 10 * np.log10(N_SUBCARRIERS) - pl
    direction = np.degrees(np.arctan2(dx, dy))
    return base, direction


def sector_rsrp(base, direction, az0, gain_dbi):
    """RSRP of the 3 sectors of every tower at every square: shape (towers, 3, squares)."""
    return np.stack([base + sector_gain_db(direction, az0 + 120 * k, gain_dbi)
                     for k in range(3)], axis=1)


def greedy_curve(cover, max_sites, target=None):
    """Add the tower that reaches the most not-yet-covered demand. Returns the order chosen
    and, after every tower, the share of demand and of squares covered."""
    covered = np.zeros(n_cells, dtype=bool)
    chosen, dem, cel = [], [], []
    while len(chosen) < max_sites:
        left = ~covered
        gain = cover[:, left].astype(float) @ demand[left]
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        chosen.append(best)
        covered |= cover[best]
        dem.append(share(covered))
        cel.append(covered.mean())
        if target is not None and dem[-1] >= target:
            break
    return chosen, dem, cel


def hex_lattice(spacing, ox, oy, angle_deg):
    """Honeycomb of tower spots: slid by (ox, oy), turned by the angle, kept inside the area."""
    dy = spacing * np.sqrt(3) / 2
    n = int(2 * half / dy) + 4
    r, k = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    x = k * spacing + (spacing / 2) * (r % 2)
    y = r * dy
    pts = np.column_stack([x.ravel(), y.ravel()])
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = pts @ rot.T + np.array([cx0 + ox, cy0 + oy])
    inside = (np.abs(pts[:, 0] - cx0) <= half) & (np.abs(pts[:, 1] - cy0) <= half)
    return pts[inside]


# every square middle is a possible tower spot (same as Phase 5)
base_c, dir_c = prepare(cx, cy)

# ============================================================================
# EXPERIMENT 1: greedy against the tuned honeycomb, both with real sectors
# ============================================================================
print("EXPERIMENT 1: greedy vs tuned honeycomb, real sector antennas")

greedy_best = {}            # tower count -> (demand share, square share, rotation)
for off in OFFSETS:
    rs = sector_rsrp(base_c, dir_c, off, BASE_GAIN_DBI)
    cov_c = (rs >= BASE_THRESHOLD_DBM).any(axis=1)
    chosen, dem, cel = greedy_curve(cov_c, MAX_N)
    for i in range(len(chosen)):
        n = i + 1
        if n not in greedy_best or dem[i] > greedy_best[n][0]:
            greedy_best[n] = (dem[i], cel[i], off)

hex_best = {}               # tower count -> (demand share, square share, spacing, angle, rotation)
tried = 0
for s in SPACINGS:
    for ang in ANGLES:
        for ox in np.linspace(0, s, N_SHIFTS, endpoint=False):
            for oy in np.linspace(0, s, N_SHIFTS, endpoint=False):
                pts = hex_lattice(s, ox, oy, ang)
                n = len(pts)
                if n < 5 or n > MAX_N:
                    continue
                base_h, dir_h = prepare(pts[:, 0], pts[:, 1])
                for off in OFFSETS:
                    rs = sector_rsrp(base_h, dir_h, off, BASE_GAIN_DBI)
                    cov = (rs >= BASE_THRESHOLD_DBM).any(axis=(0, 1))
                    frac = share(cov)
                    tried += 1
                    if n not in hex_best or frac > hex_best[n][0]:
                        hex_best[n] = (frac, cov.mean(), s, ang, off)
print("Honeycomb layouts tried (layout x antenna rotation):", tried)
print("The honeycomb gets thousands of tries, greedy only", len(OFFSETS),
      "(one per antenna rotation), so this comparison favours the honeycomb.")

rows = []
for n in sorted(set(greedy_best) & set(hex_best)):
    g, h = greedy_best[n], hex_best[n]
    rows.append({"towers": n,
                 "greedy_demand_pct": round(g[0] * 100, 1), "hex_demand_pct": round(h[0] * 100, 1),
                 "greedy_squares_pct": round(g[1] * 100, 1), "hex_squares_pct": round(h[1] * 100, 1)})
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(f"data/{NAME}_phase6_greedy_vs_hex.csv", index=False)

print()
print("Towers | Greedy demand | Hex demand | Greedy squares | Hex squares")
for n in COMPARE_N:
    r = cmp_df[cmp_df["towers"] == n]
    if r.empty:
        print(f"{n:>6} | no result for this tower count")
    else:
        r = r.iloc[0]
        print(f"{n:>6} | {r.greedy_demand_pct:>12.1f}% | {r.hex_demand_pct:>9.1f}% | "
              f"{r.greedy_squares_pct:>13.1f}% | {r.hex_squares_pct:>10.1f}%")

wins = int((cmp_df["greedy_demand_pct"] > cmp_df["hex_demand_pct"]).sum())
print(f"\nGreedy covers more demand than the honeycomb at {wins} of {len(cmp_df)} tower counts")
print("Largest greedy advantage (demand points):",
      round((cmp_df["greedy_demand_pct"] - cmp_df["hex_demand_pct"]).max(), 1))
print("Largest honeycomb advantage (demand points):",
      round((cmp_df["hex_demand_pct"] - cmp_df["greedy_demand_pct"]).max(), 1))

plt.figure(figsize=(8, 5))
plt.plot(cmp_df["towers"], cmp_df["greedy_demand_pct"], marker="o", label="Greedy")
plt.plot(cmp_df["towers"], cmp_df["hex_demand_pct"], marker="s", label="Best honeycomb (tuned)")
plt.xlabel("Number of towers (3 sectors each)")
plt.ylabel("Demand with signal (%)")
plt.title("Greedy vs tuned honeycomb with real sector antennas")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"data/{NAME}_phase6_curves.png", dpi=150)
plt.show()


# ============================================================================
# EXPERIMENT 2: how much do the assumptions change the answer?
# ============================================================================
def frequency_plan(rsrp, thr, margin):
    """Conflict graph between sectors, then graph colouring (one colour = one frequency group)."""
    n_sites = rsrp.shape[0]
    n_sec = 3 * n_sites
    flat = rsrp.reshape(n_sec, n_cells)         # sector number = tower * 3 + sector within tower
    serving = flat.argmax(axis=0)
    best = flat.max(axis=0)
    covered = best >= thr

    G = nx.Graph()
    G.add_nodes_from(range(n_sec))
    for site in range(n_sites):                 # the 3 sectors of one tower always differ
        for a in range(3):
            for b in range(a + 1, 3):
                G.add_edge(3 * site + a, 3 * site + b)

    floor_dbm = thr - 10
    hits = np.zeros((n_sec, n_sec), dtype=int)
    for cell in np.where(covered)[0]:
        s = serving[cell]
        floor = max(best[cell] - margin, floor_dbm)
        for other in np.where(flat[:, cell] >= floor)[0]:
            if other != s:
                hits[s, other] += 1
    conflict = (hits + hits.T) >= MIN_CELLS
    for a, b in np.argwhere(np.triu(conflict, 1)):
        G.add_edge(int(a), int(b))

    strategies = ["largest_first", "smallest_last", "saturation_largest_first",
                  "connected_sequential_bfs"]
    runs = {s: nx.greedy_color(G, strategy=s) for s in strategies}
    best_strategy = min(runs, key=lambda s: len(set(runs[s].values())))
    colors = runs[best_strategy]
    return {"sectors": n_sec, "conflicts": G.number_of_edges(),
            "groups": len(set(colors.values())),
            "lower_bound": max(len(c) for c in nx.find_cliques(G)),
            "valid": all(colors[a] != colors[b] for a, b in G.edges),
            "strategy": best_strategy}


_cache = {}


def plan_network(gain, thr):
    """Replan towers with real sectors; keep the antenna rotation that needs the fewest towers."""
    key = (gain, thr)
    if key in _cache:
        return _cache[key]
    best = None
    for off in OFFSETS:
        rs = sector_rsrp(base_c, dir_c, off, gain)
        cov_c = (rs >= thr).any(axis=1)
        chosen, dem, cel = greedy_curve(cov_c, MAX_SITES, TARGET_DEMAND)
        if not chosen:
            continue
        rank = (dem[-1] < TARGET_DEMAND, len(chosen), -dem[-1])
        if best is None or rank < best[0]:
            best = (rank, off, chosen, dem[-1], cel[-1])
    _cache[key] = best
    return best


def evaluate(label, gain, thr, margin):
    best = plan_network(gain, thr)
    if best is None:
        return {"scenario": label, "towers": 0}
    _, off, chosen, dem, cel = best
    rs = sector_rsrp(base_c[chosen], dir_c[chosen], off, gain)
    fp = frequency_plan(rs, thr, margin)
    return {"scenario": label, "towers": len(chosen), "sectors": fp["sectors"],
            "demand_pct": round(dem * 100, 1), "squares_pct": round(cel * 100, 1),
            "conflicts": fp["conflicts"], "freq_groups": fp["groups"],
            "lower_bound": fp["lower_bound"], "valid": fp["valid"]}


print("\nEXPERIMENT 2: sensitivity to the assumptions (one change at a time)")
scenarios = [
    ("Baseline (15 dBi, -100 dBm, 10 dB)", BASE_GAIN_DBI, BASE_THRESHOLD_DBM, BASE_MARGIN_DB),
    ("Antenna gain 12 dBi", 12.0, BASE_THRESHOLD_DBM, BASE_MARGIN_DB),
    ("Antenna gain 18 dBi", 18.0, BASE_THRESHOLD_DBM, BASE_MARGIN_DB),
    ("Threshold -105 dBm", BASE_GAIN_DBI, -105.0, BASE_MARGIN_DB),
    ("Threshold -95 dBm", BASE_GAIN_DBI, -95.0, BASE_MARGIN_DB),
    ("Conflict margin 6 dB", BASE_GAIN_DBI, BASE_THRESHOLD_DBM, 6.0),
    ("Conflict margin 15 dB", BASE_GAIN_DBI, BASE_THRESHOLD_DBM, 15.0),
]
results = []
for label, gain, thr, margin in scenarios:
    print("  running:", label)
    results.append(evaluate(label, gain, thr, margin))
sens_df = pd.DataFrame(results)
sens_df.to_csv(f"data/{NAME}_phase6_sensitivity.csv", index=False)
print()
print(sens_df.to_string(index=False))
print("\nCheck: the baseline row should match Phase 5 (26 towers, 4 frequency groups).")

# ============================================================================
# Copy the pictures to images/ with short names, because data/ is not on GitHub
# ============================================================================
os.makedirs("images", exist_ok=True)
pictures = ["phase1_map.png", "phase2_heatmap.png", "phase3_coverage.png",
            "phase3_rsrp_vs_distance.png", "phase4_placement.png", "phase4_curve.png",
            "phase4b_curves.png", "phase4b_maps.png", "phase5_frequency_map.png",
            "phase6_curves.png"]
copied = []
for pic in pictures:
    src = f"data/{NAME}_{pic}"
    if os.path.exists(src):
        shutil.copy(src, f"images/{pic}")
        copied.append(pic)
print("\nCopied to images/:", ", ".join(copied))
print("Missing (re-run that phase if you want it in the README):",
      ", ".join(p for p in pictures if p not in copied) or "none")