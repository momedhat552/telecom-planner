"""Tools for the planning agent (Phase 7).

Every number the agent reports comes from the functions below (area_summary, plan_network,
towers_needed, compare_methods, what_if, draw_plan, change_area). They work on the current area:
Berlin at the start, or any place the user asks for. Run `py planner_tools.py` to test them
without any LLM.
"""
import os
import shutil

NAME = "Berlin, Germany"    # area loaded at the start (the files made in Phases 1 and 2, in data/)

CALL_LOG = []               # every tool call and result, used by the agent for its number check
_state = {"area": None}


def current_area():
    """The area the tools work on. Loads Berlin the first time it is needed."""
    if _state["area"] is None:
        import area_builder
        _state["area"] = area_builder.load_saved_area(NAME)
    return _state["area"]


def set_area(planner):
    """Use another AreaPlanner (for example from the Streamlit app)."""
    _state["area"] = planner


def _run(name, args, fn, *a):
    """Run a tool, never crash the agent, and log the call and its result."""
    for earlier in CALL_LOG:                     # same tool, same arguments, same question: do not repeat
        if earlier["tool"] == name and earlier["args"] == args and "error" not in earlier["result"]:
            again = dict(earlier["result"])
            again["note"] = ("You already called this tool with these arguments. "
                             "Use this result and write your answer now.")
            return again
    try:
        result = fn(*a)
    except Exception as e:                       # the agent gets the error as data
        result = {"error": f"{type(e).__name__}: {e}"}
    CALL_LOG.append({"tool": name, "args": args, "result": result})
    return result


def area_summary() -> dict:
    """Describe the current study area, its demand data and the radio assumptions behind every
    result. Call this first when the user asks what the tool does or what it assumes."""
    return _run("area_summary", {}, lambda: current_area().describe())


def plan_network(n_towers: int, method: str = "greedy") -> dict:
    """Plan a network of n_towers towers, each with 3 sectors, and report how much demand
    and area it covers and how many frequency groups it needs.

    Args:
        n_towers: number of towers (1 to 80 for greedy, 5 to 40 for honeycomb).
        method: "greedy" (demand-aware placement) or "honeycomb" (best regular hexagonal layout).
    """
    return _run("plan_network", {"n_towers": n_towers, "method": method},
                lambda: current_area().plan(n_towers, method))


def towers_needed(target_demand_percent: float = 95.0) -> dict:
    """How many towers greedy placement and the best honeycomb need to cover a given
    share of demand (baseline assumptions). The honeycomb search can take a few seconds.

    Args:
        target_demand_percent: share of demand that must have signal, for example 95.
    """
    return _run("towers_needed", {"target_demand_percent": target_demand_percent},
                lambda: current_area().towers_needed(target_demand_percent))


def compare_methods(n_towers: int) -> dict:
    """Compare greedy placement with the best honeycomb at the same number of towers, with
    real sector antennas. Can take several seconds the first time.

    Args:
        n_towers: number of towers to compare (5 to 40).
    """
    return _run("compare_methods", {"n_towers": n_towers},
                lambda: current_area().compare(n_towers))


def what_if(antenna_gain_dbi: float = 15.0, coverage_threshold_dbm: float = -100.0,
            conflict_margin_db: float = 10.0) -> dict:
    """Replan with greedy placement until 95% of demand is covered, under different
    assumptions, and report towers needed and frequency groups. Takes a few seconds.

    Args:
        antenna_gain_dbi: sector antenna gain, 0 to 30 (baseline 15).
        coverage_threshold_dbm: RSRP needed to count as covered, -130 to -70 (baseline -100).
        conflict_margin_db: how close in dB two sectors must be to conflict, 0 to 30 (baseline 10).
    """
    return _run("what_if", {"antenna_gain_dbi": antenna_gain_dbi,
                            "coverage_threshold_dbm": coverage_threshold_dbm,
                            "conflict_margin_db": conflict_margin_db},
                lambda: current_area().what_if(antenna_gain_dbi, coverage_threshold_dbm,
                                               conflict_margin_db))


def draw_plan(n_towers: int, method: str = "greedy") -> dict:
    """Draw the planned network on top of a real street map (towers, sector directions,
    frequency group of every sector, coloured coverage) and save it as a PNG file. Returns the
    file path together with the coverage numbers.

    Args:
        n_towers: number of towers (1 to 80 for greedy, 5 to 40 for honeycomb).
        method: "greedy" or "honeycomb".
    """
    return _run("draw_plan", {"n_towers": n_towers, "method": method},
                lambda: current_area().draw(n_towers, method))


def change_area(place_or_coordinates: str, size_km: float = 3.0) -> dict:
    """Switch the study area to another place. Downloads OpenStreetMap buildings and roads and
    builds the demand grid. Takes about a minute the first time. All later tools then work on
    the new area. If the place cannot be found, or almost nothing is built there (open land,
    or missing map data), nothing is changed and the current area stays.

    Args:
        place_or_coordinates: a place name such as "Paris, France", or coordinates "48.86, 2.35".
        size_km: side of the square study area in km, 1 to 5 (default 3).
    """
    def go():
        import area_builder
        before = current_area().name
        try:
            area = area_builder.build_area(place_or_coordinates, size_km)
        except Exception as e:
            # The model gets a plain error and the area stays as it was.
            return {"error": f"{type(e).__name__}: {e}", "area_still_in_use": before}
        set_area(area)
        out = area.describe()
        out.pop("assumptions", None)             # unchanged, area_summary shows them
        out.pop("limitations", None)
        out["previous_area"] = before
        return out
    return _run("change_area", {"place_or_coordinates": place_or_coordinates, "size_km": size_km}, go)


if __name__ == "__main__":
    import json
    print("Self-test, no LLM involved. Area:", current_area().name)
    for label, res in [("area_summary", area_summary()),
                       ("plan_network 26 greedy", plan_network(26, "greedy")),
                       ("plan_network 26 honeycomb", plan_network(26, "honeycomb")),
                       ("towers_needed 95", towers_needed(95)),
                       ("compare_methods 26", compare_methods(26)),
                       ("what_if gain 12", what_if(12, -100, 10)),
                       ("draw_plan 26 greedy", draw_plan(26, "greedy")),
                       ("draw_plan 26 honeycomb", draw_plan(26, "honeycomb"))]:
        print(f"\n--- {label}")
        print(json.dumps(res, indent=2)[:1200])
    os.makedirs("images", exist_ok=True)            # short names for the README
    slug = current_area().slug
    for m in ("greedy", "honeycomb"):
        src = f"plans/{slug}_plan_{m}_26_towers.png"
        if os.path.exists(src):
            shutil.copy(src, f"images/plan_{m}_26.png")
    print("\nPictures copied to images/plan_greedy_26.png and images/plan_honeycomb_26.png")