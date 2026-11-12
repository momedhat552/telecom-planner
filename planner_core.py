"""Planning engine for any area.

An AreaPlanner is built from a demand grid (one number per 100 m square: building footprint
in m2) in metres. Everything else (propagation, tower placement, honeycomb, sectors, frequency
groups, plan picture) works on that grid, so the same code serves Berlin, any area found by
name, and later an area read from a picture.
"""
import os
import numpy as np
import networkx as nx
try:
    from dotenv import load_dotenv
    load_dotenv()      # lets a .env file hold CARTO_API_KEY
except ImportError:
    pass
import matplotlib
matplotlib.use("Agg")       # draw straight to files, no pop-up windows
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection, PolyCollection
from matplotlib.patches import Rectangle, Patch
import matplotlib.patheffects as pe

# --- Radio assumptions (baseline, all still to verify) ---
FC_GHZ = 3.5
H_BS = 25.0
H_UT = 1.5
H_E = 1.0
TX_POWER_DBM = 46           # per sector
N_SUBCARRIERS = 3276        # 100 MHz at 30 kHz spacing
BASE_GAIN_DBI = 15.0
BASE_THRESHOLD_DBM = -100.0
BASE_MARGIN_DB = 10.0
BEAMWIDTH_DEG = 65
FRONT_BACK_DB = 30

# --- Planning settings ---
TARGET_DEMAND = 0.95
MAX_SITES = 80
OFFSETS = list(range(0, 120, 15))       # rotations of the 3-sector pattern (degrees)
MIN_CELLS = 2
SPACINGS = range(450, 1501, 25)         # honeycomb spacings for a 3 km wide area (m), scaled for other sizes
ANGLES = [0, 15, 30, 45]                # honeycomb rotations searched (degrees)
N_SHIFTS = 5                            # honeycomb slide positions in x and in y
HEX_MIN_N, HEX_MAX_N = 5, 40            # honeycomb search works for these tower counts


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


class AreaPlanner:
    def __init__(self, name, slug, crs, cx, cy, demand, cell_m=100.0, layers=None, info=None):
        """name: shown to people. slug: safe short name for files. crs: coordinate system of the
        metre coordinates. cx, cy: middle of every square (m). demand: footprint per square (m2).
        layers: (buildings, roads) as lists of coordinate arrays, only used for the picture."""
        self.name, self.slug, self.crs = name, slug, crs
        self.cx, self.cy = np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)
        self.demand = np.asarray(demand, dtype=float)
        self.n_cells = len(self.cx)
        self.cell_m = float(cell_m)
        self.total_demand = float(self.demand.sum())
        if self.total_demand <= 0:
            raise ValueError("this area has no buildings, so there is no demand to plan for")
        h = self.cell_m / 2
        self.xmin, self.xmax = self.cx.min() - h, self.cx.max() + h
        self.ymin, self.ymax = self.cy.min() - h, self.cy.max() + h
        self.cx0, self.cy0 = (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2
        self.half = max(self.xmax - self.xmin, self.ymax - self.ymin) / 2
        self.layers = layers if layers is not None else ([], [])
        self.info = dict(info or {})
        self.base_c, self.dir_c = self.prepare(self.cx, self.cy)   # every square is a possible tower spot
        self._greedy_cache = {}
        self._lattices = None
        self._hex_cache = {}

    # ------------------------------------------------------------ basic pieces
    def share(self, mask):
        return self.demand[mask].sum() / self.total_demand

    def prepare(self, sx, sy):
        """Signal before the antenna (dBm) and compass direction, for each tower and each square."""
        dx = self.cx[None, :] - np.asarray(sx)[:, None]
        dy = self.cy[None, :] - np.asarray(sy)[:, None]
        d2d = np.maximum(np.hypot(dx, dy), 10.0)
        d3d = np.sqrt(d2d ** 2 + (H_BS - H_UT) ** 2)
        p = los_probability(d2d)
        pl = p * pl_los(d2d, d3d) + (1 - p) * pl_nlos(d2d, d3d)
        base = TX_POWER_DBM - 10 * np.log10(N_SUBCARRIERS) - pl
        direction = np.degrees(np.arctan2(dx, dy))
        return base, direction

    @staticmethod
    def sector_rsrp(base, direction, az0, gain_dbi):
        """RSRP of the 3 sectors of every tower at every square: shape (towers, 3, squares)."""
        return np.stack([base + sector_gain_db(direction, az0 + 120 * k, gain_dbi)
                         for k in range(3)], axis=1)

    # ------------------------------------------------------------------ greedy
    def greedy_curve(self, cover, max_sites):
        covered = np.zeros(self.n_cells, dtype=bool)
        chosen, dem, cel = [], [], []
        while len(chosen) < max_sites:
            left = ~covered
            gain = cover[:, left].astype(float) @ self.demand[left]
            best = int(np.argmax(gain))
            if gain[best] <= 0:
                break
            chosen.append(best)
            covered |= cover[best]
            dem.append(self.share(covered))
            cel.append(covered.mean())
        return chosen, dem, cel

    def greedy_all(self, gain, thr):
        """Greedy tower order for every antenna rotation (computed once, then remembered)."""
        key = (float(gain), float(thr))
        if key not in self._greedy_cache:
            out = {}
            for off in OFFSETS:
                rs = self.sector_rsrp(self.base_c, self.dir_c, off, gain)
                out[off] = self.greedy_curve((rs >= thr).any(axis=1), MAX_SITES)
            self._greedy_cache[key] = out
        return self._greedy_cache[key]

    # --------------------------------------------------------------- honeycomb
    def hex_lattice(self, spacing, ox, oy, angle_deg):
        dy = spacing * np.sqrt(3) / 2
        n = int(2 * self.half / dy) + 4
        r, k = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
        x = k * spacing + (spacing / 2) * (r % 2)
        y = r * dy
        pts = np.column_stack([x.ravel(), y.ravel()])
        a = np.radians(angle_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = pts @ rot.T + np.array([self.cx0 + ox, self.cy0 + oy])
        inside = (np.abs(pts[:, 0] - self.cx0) <= self.half) & (np.abs(pts[:, 1] - self.cy0) <= self.half)
        return pts[inside]

    def lattices(self):
        """All honeycomb layouts, grouped by how many towers they have (built once)."""
        if self._lattices is None:
            scale = 2 * self.half / 3000.0
            groups = {}
            for s0 in SPACINGS:
                s = s0 * scale
                for ang in ANGLES:
                    for ox in np.linspace(0, s, N_SHIFTS, endpoint=False):
                        for oy in np.linspace(0, s, N_SHIFTS, endpoint=False):
                            pts = self.hex_lattice(s, ox, oy, ang)
                            if HEX_MIN_N <= len(pts) <= HEX_MAX_N:
                                groups.setdefault(len(pts), []).append((pts, s, ang))
            self._lattices = groups
        return self._lattices

    def best_honeycomb(self, n_towers, gain, thr):
        """Layout with exactly n_towers towers that covers the most demand (same search as Phase 6)."""
        key = (n_towers, float(gain), float(thr))
        if key not in self._hex_cache:
            best = None
            for pts, s, ang in self.lattices().get(n_towers, []):
                base_h, dir_h = self.prepare(pts[:, 0], pts[:, 1])
                for off in OFFSETS:
                    rs = self.sector_rsrp(base_h, dir_h, off, gain)
                    frac = self.share((rs >= thr).any(axis=(0, 1)))
                    if best is None or frac > best[0]:
                        best = (frac, pts, off, s, ang)
            self._hex_cache[key] = best
        return self._hex_cache[key]

    # ----------------------------------------------------------------- network
    def build_network(self, method, n_towers, gain=BASE_GAIN_DBI, thr=BASE_THRESHOLD_DBM):
        """Tower positions, antenna rotation and the signal of every sector at every square."""
        info = {}
        if method == "greedy":
            best = None
            for off, (chosen, dem, cel) in self.greedy_all(gain, thr).items():
                if len(chosen) >= n_towers and (best is None or dem[n_towers - 1] > best[0]):
                    best = (dem[n_towers - 1], off, chosen[:n_towers])
            if best is None:
                return None
            _, off, chosen = best
            fx, fy = self.cx[chosen], self.cy[chosen]
            base, direction = self.base_c[chosen], self.dir_c[chosen]
        else:
            best = self.best_honeycomb(n_towers, gain, thr)
            if best is None:
                return None
            _, pts, off, spacing, angle = best
            fx, fy = pts[:, 0], pts[:, 1]
            base, direction = self.prepare(fx, fy)
            info = {"honeycomb_spacing_m": int(round(spacing)), "honeycomb_rotation_deg": int(angle)}
        rsrp = self.sector_rsrp(base, direction, off, gain)
        flat = rsrp.reshape(3 * len(fx), self.n_cells)
        return {"method": method, "n": len(fx), "fx": fx, "fy": fy, "off": off, "rsrp": rsrp,
                "serving": flat.argmax(axis=0), "covered": flat.max(axis=0) >= thr,
                "thr": thr, "gain": gain, "info": info}

    def frequency_plan(self, net, margin):
        """Conflict graph between sectors, then graph colouring (one colour = one frequency group)."""
        flat = net["rsrp"].reshape(3 * net["n"], self.n_cells)
        n_sec = flat.shape[0]
        serving, covered, thr = net["serving"], net["covered"], net["thr"]
        best = flat.max(axis=0)

        G = nx.Graph()
        G.add_nodes_from(range(n_sec))
        for site in range(net["n"]):                # the 3 sectors of one tower always differ
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
        return {"conflicts": G.number_of_edges(), "groups": len(set(colors.values())),
                "lower_bound": max(len(c) for c in nx.find_cliques(G)),
                "valid": all(colors[a] != colors[b] for a, b in G.edges),
                "sector_group": np.array([colors[s] for s in range(n_sec)])}

    def summary(self, net, margin=BASE_MARGIN_DB):
        fp = self.frequency_plan(net, margin)
        out = {"area": self.name, "method": net["method"], "towers": int(net["n"]),
               "sectors": int(3 * net["n"]),
               "demand_covered_percent": round(float(self.share(net["covered"])) * 100, 1),
               "squares_covered_percent": round(float(net["covered"].mean()) * 100, 1),
               "frequency_groups": int(fp["groups"]),
               "frequency_groups_lower_bound": int(fp["lower_bound"]),
               "sector_conflicts": int(fp["conflicts"]),
               "frequency_plan_valid": bool(fp["valid"]),
               "antenna_gain_dbi": float(net["gain"]),
               "coverage_threshold_dbm": float(net["thr"]),
               "conflict_margin_db": float(margin)}
        out.update(net["info"])
        return out, fp

    # -------------------------------------------------- results used by the tools
    @staticmethod
    def check(method, n_towers):
        method = str(method).strip().lower()
        if method not in ("greedy", "honeycomb"):
            raise ValueError("method must be 'greedy' or 'honeycomb'")
        n = int(round(float(n_towers)))
        lo, hi = (1, MAX_SITES) if method == "greedy" else (HEX_MIN_N, HEX_MAX_N)
        if not lo <= n <= hi:
            raise ValueError(f"for {method}, n_towers must be between {lo} and {hi}")
        return method, n

    def describe(self):
        side = float(max(self.xmax - self.xmin, self.ymax - self.ymin))
        out = {
            "area": self.name,
            "side_km": round(side / 1000, 2),
            "grid_cells": int(self.n_cells),
            "cell_size_m": int(self.cell_m),
            "total_building_footprint_km2": round(self.total_demand / 1e6, 3),
            "built_up_fraction_in_grid": round(self.total_demand / (side ** 2), 3),
            "cells_without_buildings": int((self.demand == 0).sum()),
            "assumptions": {
                "band_ghz": FC_GHZ, "bandwidth_mhz": 100, "tx_power_dbm_per_sector": TX_POWER_DBM,
                "antenna_gain_dbi": BASE_GAIN_DBI, "coverage_threshold_dbm": BASE_THRESHOLD_DBM,
                "base_station_height_m": H_BS, "user_height_m": H_UT, "sectors_per_tower": 3,
                "propagation_model": "3GPP TR 38.901 Urban Macro, LOS/NLOS mix",
                "demand_proxy": "building footprint area per grid cell"},
            "limitations": [
                "demand is a proxy (building footprint), not real traffic",
                "buildings missing in OpenStreetMap count as zero demand",
                "no shadow fading, terrain or indoor loss",
                "radio values are assumptions, not operator data",
                "frequency conflicts use a heuristic rule, not an SINR calculation"]}
        out.update(self.info)
        return out

    def plan(self, n_towers, method="greedy"):
        method, n = self.check(method, n_towers)
        net = self.build_network(method, n)
        if net is None:
            raise ValueError("no layout found for that number of towers")
        return self.summary(net)[0]

    def towers_needed(self, target_percent):
        target = float(target_percent)
        if not 1 <= target <= 100:
            raise ValueError("target_demand_percent must be between 1 and 100")
        greedy_n = None
        for off, (chosen, dem, cel) in self.greedy_all(BASE_GAIN_DBI, BASE_THRESHOLD_DBM).items():
            for i, d in enumerate(dem):
                if d * 100 >= target:
                    greedy_n = i + 1 if greedy_n is None else min(greedy_n, i + 1)
                    break
        centre = greedy_n if greedy_n else HEX_MAX_N - 8
        lo, hi = max(HEX_MIN_N, centre - 8), min(HEX_MAX_N, centre + 8)
        hex_n = None
        for n in range(lo, hi + 1):
            best = self.best_honeycomb(n, BASE_GAIN_DBI, BASE_THRESHOLD_DBM)
            if best is not None and best[0] * 100 >= target:
                hex_n = n
                break
        return {"area": self.name, "target_demand_percent": target, "greedy_towers": greedy_n,
                "honeycomb_towers": hex_n,
                "assumptions_used": {"antenna_gain_dbi": BASE_GAIN_DBI,
                                     "coverage_threshold_dbm": BASE_THRESHOLD_DBM},
                "note": f"honeycomb searched from {lo} to {hi} towers (not reached = none in that range)"}

    def compare(self, n_towers):
        n = int(round(float(n_towers)))
        if not HEX_MIN_N <= n <= HEX_MAX_N:
            raise ValueError(f"n_towers must be between {HEX_MIN_N} and {HEX_MAX_N}")
        g = self.build_network("greedy", n)
        h = self.build_network("honeycomb", n)
        if g is None or h is None:
            raise ValueError("no layout found for that number of towers")
        gd, hd = float(self.share(g["covered"])) * 100, float(self.share(h["covered"])) * 100
        gs, hs = float(g["covered"].mean()) * 100, float(h["covered"].mean()) * 100
        return {"area": self.name, "towers": n,
                "greedy_demand_percent": round(gd, 1), "honeycomb_demand_percent": round(hd, 1),
                "greedy_squares_percent": round(gs, 1), "honeycomb_squares_percent": round(hs, 1),
                "greedy_minus_honeycomb_demand_points": round(gd - hd, 1),
                "greedy_minus_honeycomb_squares_points": round(gs - hs, 1),
                "note": "honeycomb is the best of thousands of tried layouts, greedy of 8 antenna rotations"}

    def what_if(self, gain, thr, margin):
        gain, thr, margin = float(gain), float(thr), float(margin)
        if not (0 <= gain <= 30 and -130 <= thr <= -70 and 0 <= margin <= 30):
            raise ValueError("use antenna gain 0 to 30 dBi, threshold -130 to -70 dBm, margin 0 to 30 dB")
        n_needed = None
        for off, (chosen, dem, cel) in self.greedy_all(gain, thr).items():
            for i, d in enumerate(dem):
                if d >= TARGET_DEMAND:
                    n_needed = i + 1 if n_needed is None else min(n_needed, i + 1)
                    break
        reached = n_needed is not None
        net = self.build_network("greedy", n_needed if reached else MAX_SITES, gain, thr)
        out, _ = self.summary(net, margin)
        out["reached_95_percent_demand"] = reached
        if not reached:
            out["note"] = f"95% of demand was not reached with {MAX_SITES} towers"
        return out

    def draw(self, n_towers, method="greedy", basemap="tiles", folder="plans"):
        method, n = self.check(method, n_towers)
        net = self.build_network(method, n)
        if net is None:
            raise ValueError("no layout found for that number of towers")
        out, fp = self.summary(net)
        os.makedirs(folder, exist_ok=True)
        path = f"{folder}/{self.slug}_plan_{method}_{n}_towers.png"
        title = (f"{self.name}: {n} towers ({method}), {out['frequency_groups']} frequency groups, "
                 f"{out['demand_covered_percent']}% of demand covered")
        out["background"] = self.draw_picture(net, fp, title, path, basemap)
        out["picture_file"] = path
        return out

    # ----------------------------------------------------------------- picture
    def draw_picture(self, net, fp, title, path, basemap="tiles"):
        """Plan on top of a real map. basemap: 'tiles' (CARTO street map tiles, only if the
        CARTO_API_KEY environment variable is set, otherwise the saved OpenStreetMap layers),
        'osm' (saved layers only) or 'none'."""
        fig, ax = plt.subplots(figsize=(10, 9))
        cmap = plt.get_cmap("tab10")
        h = self.cell_m / 2
        halo = [pe.withStroke(linewidth=2.5, foreground="white")]
        ax.set_xlim(self.cx.min() - h, self.cx.max() + h)
        ax.set_ylim(self.cy.min() - h, self.cy.max() + h)
        ax.set_aspect("equal")

        # 1. the real map underneath
        used = "none"
        key = os.environ.get("CARTO_API_KEY")       # free key from https://carto.com/basemaps/apikey/
        if basemap == "tiles" and key:
            try:
                import contextily as ctx
                source = f"https://basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?key={key}"
                ctx.add_basemap(ax, crs=self.crs, source=source, zorder=0,
                                attribution="(C) OpenStreetMap contributors (C) CARTO")
                ax.set_xlim(self.cx.min() - h, self.cx.max() + h)
                ax.set_ylim(self.cy.min() - h, self.cy.max() + h)
                used = "CARTO street map tiles"
            except Exception as e:
                print("Note: could not load street map tiles, using the saved map layers:", e)
        if used == "none" and basemap in ("tiles", "osm"):
            buildings, roads = self.layers
            if roads:
                ax.add_collection(LineCollection(roads, colors="#b0b0b0", linewidths=0.6, zorder=1))
            if buildings:
                # buildings sit above the colour tint so every outline stays sharp and dark
                ax.add_collection(PolyCollection(buildings, facecolors="#3a3a3a",
                                                 edgecolors="none", alpha=0.9, zorder=4))
            if roads or buildings:
                used = "saved OpenStreetMap buildings and roads"

        # 2. the plan on top, see-through so the map stays visible
        idx = np.where(net["covered"])[0]
        patches = [Rectangle((self.cx[i] - h, self.cy[i] - h), self.cell_m, self.cell_m) for i in idx]
        colors = [cmap(fp["sector_group"][net["serving"][i]] % 10) for i in idx]
        ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="none",
                                          alpha=0.35, zorder=3))

        # 3. towers, sector directions and frequency labels
        fx, fy = net["fx"], net["fy"]
        reach = 0.06 * 2 * self.half            # arrow length scales with the area size
        for s in range(3 * net["n"]):
            site, k = divmod(s, 3)
            g = int(fp["sector_group"][s])
            az = np.radians(net["off"] + 120 * k)
            ax.plot([fx[site], fx[site] + reach * np.sin(az)], [fy[site], fy[site] + reach * np.cos(az)],
                    color=cmap(g % 10), linewidth=2.2, solid_capstyle="round",
                    path_effects=[pe.withStroke(linewidth=3.6, foreground="black")], zorder=5)
            ax.text(fx[site] + 1.4 * reach * np.sin(az), fy[site] + 1.4 * reach * np.cos(az), f"F{g + 1}",
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    path_effects=halo, clip_on=True, zorder=6)
        ax.plot(fx, fy, "^", color="black", markersize=10, markeredgecolor="white", zorder=7)

        ax.set_axis_off()
        handles = [Patch(facecolor=cmap(g % 10), alpha=0.6, label=f"F{g + 1}") for g in range(fp["groups"])]
        handles.append(Patch(facecolor="white", edgecolor="#999999", label="no colour = no signal"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Frequency group")
        ax.set_title(title)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return used