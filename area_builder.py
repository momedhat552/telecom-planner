"""Build an AreaPlanner for any place: look up the name, download OpenStreetMap buildings and
roads, cut the area into 100 m squares (same method as Phases 1 and 2) and cache the result."""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import Point, box
from planner_core import AreaPlanner

AREAS_DIR = "areas"         # cached areas are saved here, one folder per area
BERLIN = "Berlin, Germany"  # the case study: data/ holds the Phase 1 and 2 files for this square
MIN_KM, MAX_KM = 1.0, 5.0   # bigger areas are slow and the planner is tuned for 3 km
LOW_BUILT_UP = 0.10         # below this the area is probably mostly empty or poorly mapped
MIN_BUILT_UP = 0.02         # below this there is nothing worth planning, so we refuse it
BUSY_WORDS = ("429", "500", "502", "503", "504", "timeout", "capacity")   # Overpass is overloaded
CONNECTION_WORDS = ("connection", "connecttimeout", "max retries", "getaddrinfo",
                    "name or service not known", "network is unreachable",
                    "temporary failure in name resolution")
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
MAX_PLACE_RANK = 25      # Nominatim: 30 = a shop or building, 26 = a street; 25 and below a place
USER_AGENT = "telecom-planner/1.0 (student portfolio project)"

# Public Overpass servers that have worldwide data. The main overpass-api.de host often never
# connects from here, so it is last. maps.mail.ru is first: it is the one that has actually
# returned buildings (including a Cairo query) while the others were busy or unreachable.
# overpass.osm.ch is left out on purpose: it only has Swiss data.
GLOBAL_OVERPASS_URLS = [
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://lz4.overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://z.overpass-api.de/api",
    "https://overpass-api.de/api",
]
# This one answers quickly, but it only has the Geofabrik Europe extract. A query in Egypt comes
# back with an empty building list, which would look like "this city has no buildings". An empty
# answer from this server is therefore never treated as the real result.
EUROPE_OVERPASS_URL = "https://ovp.dresel.systems/api"
OVERPASS_URLS = [EUROPE_OVERPASS_URL] + GLOBAL_OVERPASS_URLS
PROBE_QUERY = '[out:json][timeout:20];way["building"](52.5200,13.4080,52.5210,13.4090);out count;'
_PROBE = {"time": 0.0, "key": None, "urls": []}     # which servers answered, and when they were checked


def _is_connection_error(e):
    """True when the server could not be reached at all (not just busy or slow)."""
    text = f"{type(e).__name__} {e}".lower()
    return any(word in text for word in CONNECTION_WORDS)


def _probe(url, timeout_s=20):
    """Does this Overpass server answer with usable JSON right now? The query is tiny."""
    try:
        r = requests.post(f"{url}/interpreter", data={"data": PROBE_QUERY},
                          headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
        return url if r.status_code == 200 and "elements" in r.text else None
    except Exception:
        return None


def working_overpass_urls(urls=None, timeout_s=12, cache_s=120):
    """Which of these servers answer a tiny query right now. Checked in parallel, because a dead
    server would otherwise cost the full connection timeout each time. This only decides the
    order of the real download: an empty result must not cancel it. Servers often miss a 12 s
    check and then answer the download."""
    urls = list(urls or OVERPASS_URLS)
    key = tuple(urls)
    now = time.time()
    if _PROBE["key"] == key and _PROBE["urls"] and now - _PROBE["time"] < cache_s:
        return list(_PROBE["urls"])
    found = []
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = [pool.submit(_probe, url, timeout_s) for url in urls]
        for future in as_completed(futures):
            url = future.result()
            if url:
                found.append(url)
    _PROBE.update(time=now, key=key, urls=found)
    return found


def _empty_map(result):
    """True when a download came back with nothing in it (no buildings, or a road graph with no
    nodes). A server that does not cover the country does this instead of returning an error."""
    try:
        return len(result) == 0
    except Exception:
        return False


def _worth_retry(e):
    """Busy or slow servers are worth one more try. A server that cannot be connected to is not:
    the second try fails the same way."""
    text = f"{type(e).__name__} {e}".lower()
    if any(word in text for word in ("connecttimeout", "connectionerror", "name resolution",
                                     "name or service not known")):
        return False
    return any(word in text for word in ("429", "500", "502", "503", "504", "timeout", "capacity"))


def _download(label, fn, centre=None):
    """Run an osmnx download. A failed health check used to stop here and tell you to open a
    saved area instead. The real download is always attempted.

    osmnx normally asks each server's status page first, with the same long timeout as the
    download, and waits 60 s more when that page is down. That wait is turned off for this call.
    One area at a time does not need it, and a broken status page was delaying the download
    even when the map server itself was ready."""
    urls = list(OVERPASS_URLS)
    healthy = working_overpass_urls(urls)
    ordered = [u for u in urls if u in healthy] + [u for u in urls if u not in healthy]
    errors = []
    empty_from_world = None
    old_limit = ox.settings.overpass_rate_limit
    old_timeout = ox.settings.requests_timeout
    ox.settings.overpass_rate_limit = False
    ox.settings.requests_timeout = 90
    try:
        for url in ordered:
            host = url.split("/")[2]
            ox.settings.overpass_url = url
            for attempt in (1, 2):
                try:
                    print(f"Downloading {label} from {host} (try {attempt})...")
                    result = fn()
                except Exception as e:
                    errors.append(f"{host} try {attempt} ({type(e).__name__})")
                    if attempt == 1 and _worth_retry(e):
                        print(f"{host} is busy ({type(e).__name__}). Trying it once more in 8 s...")
                        time.sleep(8)
                        continue
                    break
                if _empty_map(result):
                    errors.append(f"{host} (answered, but no map features)")
                    if url != EUROPE_OVERPASS_URL:
                        empty_from_world = result
                    break
                return result
    finally:
        ox.settings.overpass_rate_limit = old_limit
        ox.settings.requests_timeout = old_timeout
    if empty_from_world is not None:
        return empty_from_world
    raise ValueError(_download_failed_message(label, errors))


def _words(text):
    """Lower-case words of a place name, for comparing 'Mohandseen, Giza, Egypt' with a saved
    'Mohandseen, Dokki, Giza, Egypt'."""
    return [w for w in re.split(r"[^a-z0-9]+", str(text).lower()) if w]



def _with_retry(label, fn, attempts=2, wait_s=10):
    """Run a download; if the OpenStreetMap server is busy, wait and try once more."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            busy = any(word in str(e).lower() for word in BUSY_WORDS)
            if _is_connection_error(e) or not busy or i == attempts:
                raise
            print(f"OpenStreetMap is busy while downloading {label} "
                  f"({type(e).__name__}). Trying again in {wait_s} s...")
            time.sleep(wait_s)
            wait_s *= 2


def slugify(text):
    """'Berlin, Germany' -> 'berlin_germany' (safe for file names)."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "area"


def parse_coordinates(text):
    """'52.52, 13.41' -> (52.52, 13.41), otherwise None."""
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$", str(text))
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def candidate_queries(place):
    """The name as typed, then simpler versions. People often write '+' between districts or
    misspell one of them, so we also try each district with the city and country at the end."""
    place = str(place).strip()
    parts = [x.strip() for x in re.split(r"\s*(?:\+|&|,|;|\band\b)\s*", place, flags=re.I) if x.strip()]
    queries = [place]
    if len(parts) >= 3:
        tail = ", ".join(parts[-2:])
        queries += [f"{x}, {tail}" for x in parts[:-2]]
        queries.append(tail)
    seen, out = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def _search_place(query):
    """One Nominatim name search. Returns (lat, lon, name, display_name, place_rank), or None
    if nothing was found."""
    r = requests.get(NOMINATIM_URL,
                     params={"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1},
                     headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    hits = r.json()
    if not hits:
        return None
    h = hits[0]
    display = str(h.get("display_name", ""))
    return (float(h["lat"]), float(h["lon"]),
            str(h.get("name") or display or query), display, int(h.get("place_rank", 30)))


def _find_place(place):
    """Look the place up by name. Returns ((lat, lon), the search text that worked).
    A result that is a shop, a building or a street is not accepted: Nominatim otherwise turns
    'netherlands capital' into a bar in Maastricht and 'where is amsterdam' into a speech
    therapist in New York."""
    tried, not_a_place = [], []
    for i, q in enumerate(candidate_queries(place)):
        if i:
            time.sleep(1)                # Nominatim asks for at most one request per second
        try:
            hit = _with_retry(f"the place name '{q}'", lambda: _search_place(q))
        except Exception as e:
            if _is_connection_error(e):
                raise ValueError(_place_lookup_message(q)) from e
            tried.append(q)
            continue
        if hit is None:
            tried.append(q)
            continue
        lat, lon, name, display, rank = hit
        if rank <= MAX_PLACE_RANK:
            return (lat, lon), q
        not_a_place.append(f"'{name}' ({display})")
    hint = ("The name search only found " + "; ".join(not_a_place) +
            ", which is not a city or district. ") if not_a_place else ""
    searched = ("Searches tried: " + " | ".join(tried) + ". ") if tried else ""
    raise ValueError("could not find the place. " + searched + hint +
                     "Use a city or district name such as 'Amsterdam, Netherlands', or "
                     "coordinates such as '52.37, 4.89'.")


def _layer_coords(buildings, roads_graph, crs):
    """Buildings and roads as plain coordinate lists, used to draw the map under the plan."""
    bl, rl = [], []
    try:
        for g in buildings.geometry:
            parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
            bl.extend(np.array(p.exterior.coords) for p in parts if p.geom_type == "Polygon")
    except Exception as e:
        print("Note: could not read the building outlines for the picture:", e)
    try:
        if roads_graph is not None:
            edges = ox.graph_to_gdfs(roads_graph, nodes=False).to_crs(crs)
            for g in edges.geometry:
                parts = g.geoms if g.geom_type == "MultiLineString" else [g]
                rl.extend(np.array(p.coords) for p in parts)
    except Exception as e:
        print("Note: could not read the roads for the picture:", e)
    return bl, rl


def _planner(name, slug, grid, buildings, roads_graph, info):
    cent = grid.geometry.centroid
    cx, cy = cent.x.values, cent.y.values
    cell_m = float(np.min(np.diff(np.unique(np.round(cx)))))
    layers = _layer_coords(buildings, roads_graph, grid.crs)
    return AreaPlanner(name, slug, grid.crs, cx, cy, grid["built_m2"].values,
                       cell_m=cell_m, layers=layers, info=info)


def _make_grid(buildings, centre_xy, half_m, cell_m):
    """Chessboard of cell_m squares; sum the building footprint inside each square (Phase 2)."""
    xs = np.arange(centre_xy[0] - half_m, centre_xy[0] + half_m, cell_m)
    ys = np.arange(centre_xy[1] - half_m, centre_xy[1] + half_m, cell_m)
    cells = [box(x, y, x + cell_m, y + cell_m) for x in xs for y in ys]
    grid = gpd.GeoDataFrame({"geometry": cells}, crs=buildings.crs)
    grid["cell_id"] = grid.index
    pieces = gpd.overlay(grid, buildings[["geometry"]], how="intersection")
    pieces["piece_m2"] = pieces.geometry.area
    built = pieces.groupby("cell_id")["piece_m2"].sum()
    grid["built_m2"] = grid["cell_id"].map(built).fillna(0)
    return grid


def load_saved_area(name="Berlin, Germany", data_dir="data"):
    """Load the area made in Phases 1 and 2 (files named data/<name>_...)."""
    grid = gpd.read_file(f"{data_dir}/{name}_demand_grid.gpkg")
    buildings = gpd.read_file(f"{data_dir}/{name}_buildings.gpkg").to_crs(grid.crs)
    roads = None
    roads_path = f"{data_dir}/{name}_roads.graphml"
    if os.path.exists(roads_path):
        roads = ox.load_graphml(roads_path)
    info = {"buildings_in_area": int(len(buildings)), "data_source": "OpenStreetMap (Phase 1 files)"}
    return _planner(name, slugify(name), grid, buildings, roads, info)


def saved_areas():
    """The areas already saved in areas/, with their name, size and centre. They can be opened
    without internet (see load_cached)."""
    out = []
    if not os.path.isdir(AREAS_DIR):
        return out
    for name in sorted(os.listdir(AREAS_DIR)):
        path = f"{AREAS_DIR}/{name}/meta.json"
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                meta = json.load(f)
        except Exception:
            continue
        info = meta.get("info", {})
        out.append({"folder": name, "place": meta.get("place", name),
                    "size_km": float(info.get("requested_size_km", 3.0)),
                    "centre": (info.get("centre_lat"), info.get("centre_lon")),
                    "buildings": info.get("buildings_in_area")})
    return out


def _load_folder(folder, requested):
    """Read one saved area from areas/<folder> and return a planner for it."""
    with open(f"{folder}/meta.json") as f:
        meta = json.load(f)
    grid = gpd.read_file(f"{folder}/demand_grid.gpkg")
    if _built_up_fraction(grid) < MIN_BUILT_UP:      # a bad copy saved by an older version
        raise ValueError(
            f"the saved copy of '{meta.get('place', requested)}' has almost no buildings "
            f"({_built_up_fraction(grid):.1%} of the square). Delete the folder "
            f"'{folder}' if you want to download it again.")
    buildings = gpd.read_file(f"{folder}/buildings.gpkg")
    roads = ox.load_graphml(f"{folder}/roads.graphml") if os.path.exists(f"{folder}/roads.graphml") else None
    info = dict(meta.get("info", {}))
    info["data_source"] = f"saved copy in {folder}"      # nothing was downloaded now
    return _planner(meta.get("place", requested), os.path.basename(folder), grid, buildings, roads, info)


def _saved_match(place, size_km):
    """Folder of a saved copy of the same request: the same words, or the same first word plus
    the rest of the words. So 'Mohandseen, Giza, Egypt' finds the saved 'Mohandseen, Dokki, Giza,
    Egypt', but 'Giza, Egypt' does not (that is probably another square). None if no safe match."""
    want = _words(place)
    if not want:
        return None
    for area in saved_areas():
        if abs(area["size_km"] - float(size_km)) > 1e-9:
            continue
        folder = f"{AREAS_DIR}/{area['folder']}"
        try:
            with open(f"{folder}/meta.json") as f:
                meta = json.load(f)
        except Exception:
            continue
        names = [area["place"]] + [str(x) for x in meta.get("aliases", [])]
        for name in names:
            have = _words(name)
            if have == want or (have and have[0] == want[0] and set(want) <= set(have)):
                return folder
    return None


def load_cached(place, size_km=3.0):
    """Open an area that is already saved in areas/, without using the internet."""
    size_km = float(size_km)
    folder = f"{AREAS_DIR}/{slugify(place)}_{size_km:g}km"
    if os.path.exists(f"{folder}/meta.json"):
        return _load_folder(folder, place)
    folder = _saved_match(place, size_km)
    if folder:
        return _load_folder(folder, place)
    names = "; ".join(a["place"] for a in saved_areas()) or "none yet"
    raise ValueError(f"no saved area matches '{place}'. Saved areas: {names}. "
                     f"Downloading a new one needs OpenStreetMap.")


def _place_lookup_message(query):
    """Nominatim (the place-name search) could not be reached. This is not the map download."""
    return ("could not reach the place-name server (Nominatim) to look up "
            f"'{query}'. Check the internet connection and try again. "
            "Coordinates such as '30.047, 31.205' do not need that lookup.")


def _download_failed_message(label, errors):
    """The download was attempted. An empty health check must not be described as a refusal."""
    names = "; ".join(a["place"] for a in saved_areas()) or "none yet"
    detail = ", ".join(errors) if errors else "no server could be contacted"
    return (f"tried to download {label}, and no worldwide OpenStreetMap server returned it. "
            f"A new area is allowed; the public servers are overloaded or unreachable from here. "
            f"What happened: {detail}. "
            f"One fast server only has Europe, so an empty answer from it is ignored. "
            f"Saved areas you can open without internet: {names}. Try the same place again "
            f"in a few minutes.")


def _built_up_fraction(grid):
    """Share of the study square that is covered by buildings (same number as describe() shows)."""
    x0, y0, x1, y1 = grid.total_bounds
    side = max(x1 - x0, y1 - y0)
    return float(grid["built_m2"].sum() / side ** 2)


def build_area(place, size_km=3.0, cell_m=100.0, use_cache=True):
    """Plan-ready area for a place name ('Paris, France') or coordinates ('48.86, 2.35').
    Takes about a minute the first time, then loads from the areas/ folder."""
    place = str(place).strip()
    size_km = float(size_km)
    if not MIN_KM <= size_km <= MAX_KM:
        raise ValueError(f"size_km must be between {MIN_KM:g} and {MAX_KM:g}")
    slug = f"{slugify(place)}_{size_km:g}km"
    folder = f"{AREAS_DIR}/{slug}"

    # Berlin is the case study of this repo: the saved Phase 1 and 2 files are the square all the
    # README numbers come from. Use them when they are here, so every entry point plans exactly
    # the same square (the geocoded centre of "Berlin, Germany" is about a kilometre away).
    berlin_case_study = slugify(place) == "berlin_germany" and abs(size_km - 3.0) < 1e-9
    if berlin_case_study and os.path.exists(f"data/{BERLIN}_demand_grid.gpkg"):
        return load_saved_area(BERLIN)

    # cached copy? (exact name first, then the same place written a little differently)
    if use_cache and os.path.exists(f"{folder}/meta.json"):
        return _load_folder(folder, place)
    if use_cache:
        hit = _saved_match(place, size_km)
        if hit:
            return _load_folder(hit, place)

    # 1. where is it? (the name search is a different service, so it still works when the map
    # server is down and the answer can say whether the name itself is the problem)
    centre = parse_coordinates(place)
    search_used = "coordinates"
    if berlin_case_study:                       # no saved files: same square as Phase 1
        centre = (52.52, 13.41)
        search_used = "the Phase 1 centre of the case study (52.52, 13.41)"
    elif centre is None:
        centre, search_used = _find_place(place)
    lat, lon = centre
    half_m = size_km * 1000 / 2

    # 2. buildings (Phase 1), from whichever public server answers
    b = _download("the buildings",
                  lambda: ox.features_from_point((lat, lon), tags={"building": True},
                                                 dist=half_m), centre)
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])]
    b = b[["geometry"]].reset_index(drop=True)
    if len(b) == 0:
        raise ValueError(f"OpenStreetMap has no buildings within {size_km:g} km of '{place}'")
    utm = b.estimate_utm_crs()                          # metre coordinates that fit this place
    b = b.to_crs(utm)
    try:
        b["geometry"] = b.geometry.make_valid()
    except AttributeError:
        pass
    b["area_m2"] = b.geometry.area

    # 3. roads (only used to draw the map, so a failure is not fatal)
    roads = None
    try:
        roads = _download("the roads",
                          lambda: ox.project_graph(
                              ox.graph_from_point((lat, lon), dist=half_m, network_type="drive"),
                              to_crs=utm), centre)
    except Exception as e:
        print("Note: no road network found, the picture will have no roads:", e)

    # 4. demand grid (Phase 2)
    centre_xy = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(utm).iloc[0]
    grid = _make_grid(b, (centre_xy.x, centre_xy.y), half_m, cell_m)

    side_m = 2 * half_m
    built_up = float(grid["built_m2"].sum() / side_m ** 2)

    # Almost no buildings: the name probably pointed at open land (for example the middle of a
    # country) or OpenStreetMap is empty there. Planning that would be misleading, so stop.
    if built_up < MIN_BUILT_UP:
        raise ValueError(
            f"only {built_up:.1%} of this {size_km:g} km square is covered by buildings "
            f"(centre {lat:.4f}, {lon:.4f}), so there is almost no demand to plan for. "
            f"The name probably points at open land, or OpenStreetMap has almost no buildings "
            f"here. Try a city centre, a district name, or coordinates inside a built-up area "
            f"such as '30.047, 31.205'. Nothing was saved, so you can simply try another name.")

    info = {"buildings_in_area": int(len(b)), "centre_lat": round(lat, 5), "centre_lon": round(lon, 5),
            "requested_size_km": size_km, "place_search_used": search_used,
            "data_source": "OpenStreetMap, downloaded now"}
    if built_up < LOW_BUILT_UP:
        info["data_quality_warning"] = (
            f"only {built_up:.0%} of the area is covered by buildings: it may be parks, water or desert, "
            "or OpenStreetMap may be missing buildings here. Demand will look low. Check the map "
            "before trusting the plan.")

    # 5. keep a copy so the next time is instant
    os.makedirs(folder, exist_ok=True)
    b.to_file(f"{folder}/buildings.gpkg", driver="GPKG")
    grid.to_file(f"{folder}/demand_grid.gpkg", driver="GPKG")
    if roads is not None:
        ox.save_graphml(roads, f"{folder}/roads.graphml")
    with open(f"{folder}/meta.json", "w") as f:
        json.dump({"place": place, "aliases": candidate_queries(place) + [search_used],
                   "info": info}, f)

    return _planner(place, slug, grid, b, roads, info)