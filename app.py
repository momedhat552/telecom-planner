"""Phase 8: the Streamlit app.

Pick a study area (name or coordinates), plan a network and look at the picture and the numbers,
then ask the Gemini assistant questions about the same area.

"""
import os

import streamlit as st

# set_page_config must be the first Streamlit call in the script.
st.set_page_config(page_title="5G network planner", page_icon="📡", layout="wide")

try:
    from dotenv import load_dotenv
    load_dotenv()                       # local: keys come from .env
except ImportError:
    pass


def _secret(name):
    """On Streamlit Cloud the keys live in the app secrets; locally they come from .env."""
    try:
        return st.secrets[name]
    except Exception:
        return None


# The agent reads these from the environment, so fill them in before importing it.
for _name in ("GEMINI_API_KEY", "GEMINI_MODEL", "CARTO_API_KEY"):
    _value = _secret(_name)
    if _value:
        os.environ[_name] = str(_value)

import area_builder          # noqa: E402  (imports after the secrets are in place on purpose)
import planner_tools as pt   # noqa: E402
import phase7_agent as ag    # noqa: E402

BERLIN = "Berlin, Germany"


# --------------------------------------------------------------------------- caching
@st.cache_resource(show_spinner=False)
def load_area(place: str, size_km: float):
    """One planner per (place, size), kept in memory. A downloaded area is never downloaded
    twice in the same running app: build_area also saves it in areas/ for the next start.
    Berlin at 3 km is answered from the saved Phase 1 and 2 files, which is instant."""
    return area_builder.build_area(place, size_km)


# --------------------------------------------------------------------------- session state
st.session_state.setdefault("area", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("chat", None)


def start_new_area(new_area):
    """A new area means a new plan and a new chat (the old chat talked about another place)."""
    st.session_state["area"] = new_area
    st.session_state.pop("plan", None)
    st.session_state.pop("compare", None)
    st.session_state["messages"] = []
    st.session_state["chat"] = None


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Study area")

    # An area can also change from the chat. The new name is applied here, before the widgets
    # are created, because a widget value cannot be changed after the widget exists.
    if "_new_place" in st.session_state:
        st.session_state["place_box"] = st.session_state.pop("_new_place")
        st.session_state["size_box"] = float(st.session_state.pop("_new_size", 3.0))

    st.session_state.setdefault("place_box", BERLIN)
    st.session_state.setdefault("size_box", 3.0)

    place = st.text_input("Place name or coordinates", key="place_box",
                          help="For example 'Paris, France' or '48.8566, 2.3522'. Use the name of "
                               "a city or district; a description such as 'the capital of the "
                               "Netherlands' is not a place name.")
    size = st.number_input("Side of the square (km)", min_value=1.0, max_value=5.0,
                           step=0.5, key="size_box")
    if st.button("Load area", type="primary"):
        with st.spinner("Loading the area. A new place needs an OpenStreetMap download, "
                        "which usually takes about a minute..."):
            try:
                found = load_area(place.strip(), float(size))
            except Exception as e:
                found = None
                st.error(f"Could not load '{place}': {e}")
        if found is not None:
            start_new_area(found)
            st.success(f"Loaded {found.name}.")

    saved = area_builder.saved_areas()
    if saved:
        with st.expander("Saved areas (open without a download)"):
            labels = [f"{a['place']} · {a['size_km']:g} km" for a in saved]
            choice = st.selectbox("Area saved in the areas/ folder", labels, key="saved_box")
            if st.button("Open saved area"):
                item = saved[labels.index(choice)]
                try:
                    found = area_builder.load_cached(item["place"], item["size_km"])
                except Exception as e:
                    st.error(f"Could not open the saved area: {e}")
                else:
                    start_new_area(found)
                    st.success(f"Opened the saved area {found.name}. Nothing was downloaded.")

    st.divider()

# First run (or after a fresh start): load Berlin from the saved Phase 1 and 2 files.
if st.session_state["area"] is None:
    with st.spinner("Loading the saved Berlin data..."):
        try:
            st.session_state["area"] = load_area(BERLIN, 3.0)
        except Exception as e:
            st.error(f"Could not load the Berlin data: {e}")
            st.stop()

area = st.session_state["area"]
d = area.describe()
pt.set_area(area)          # the assistant's tools now work on the area shown here

with st.sidebar:
    st.caption("Working on")
    st.markdown(f"**{d['area']}**")
    st.caption(f"{d['side_km']:g} x {d['side_km']:g} km · {d['grid_cells']} cells of "
               f"{d['cell_size_m']} m · demand from {d.get('buildings_in_area', '?')} buildings")
    if str(d.get("data_source", "")).startswith("saved copy"):
        st.caption("Opened from the saved copy in areas/ — no download was needed.")
    if d.get("data_quality_warning"):
        st.warning(d["data_quality_warning"])

# --------------------------------------------------------------------------- header
st.title("5G network planner (3.5 GHz, n78)")
st.caption("Demand is estimated from OpenStreetMap buildings. Coverage uses the 3GPP TR 38.901 "
           "Urban Macro path loss with 3-sector antennas. Everything runs locally, except the "
           "optional chat.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Buildings", d.get("buildings_in_area", "?"))
c2.metric("Built-up area", f"{d['built_up_fraction_in_grid']:.1%}")
c3.metric("Demand cells", f"{d['grid_cells']}", help="100 m squares that make up the study area.")
c4.metric("Building footprint", f"{d['total_building_footprint_km2']} km²",
          help="Total building area inside the square. Used as the demand proxy.")

if d.get("data_quality_warning"):
    st.warning(d["data_quality_warning"])

with st.expander("How this works"):
    st.markdown(
        "1. Buildings from OpenStreetMap are cut into 100 m squares; the building area in each "
        "square is the demand there.\n"
        "2. A tower's signal at every square comes from the 3GPP path-loss model (line of sight "
        "mixed with non-line of sight), the transmit power and the antenna pattern of its "
        "3 sectors.\n"
        "3. Towers are placed greedily (the tower that covers the most uncovered demand is added "
        "each time) or on the best regular hexagonal grid, and the sectors then get frequency "
        "groups by graph colouring.\n\n"
        "No machine-learning model is trained in this project, on purpose: this is a physics and "
        "optimisation problem. The chat assistant is a language model that calls these calculations "
        "as tools; it never does the arithmetic itself.")

# --------------------------------------------------------------------------- planning
st.subheader("Plan a network")

left, middle, right = st.columns([1.1, 1.6, 1.0])
with left:
    method = st.radio("Placement", ("greedy", "honeycomb"), horizontal=True,
                      help="Greedy adds the tower that covers the most uncovered demand. "
                           "Honeycomb tries thousands of regular layouts and keeps the best.")
with middle:
    lo, hi, default = (5, 40, 26) if method == "honeycomb" else (1, 60, 26)
    n_towers = st.slider("Towers", lo, hi, default, key=f"towers_{method}")
with right:
    st.write("")
    plan_now = st.button("Plan and draw", type="primary")

compare_now = st.checkbox("Also compare greedy placement with the honeycomb at this tower count",
                          value=False,
                          help="Runs both placements with the same number of towers (5 to 40).")

if plan_now:
    with st.spinner(f"Planning {n_towers} towers ({method}) and drawing the map..."):
        try:
            result = area.draw(n_towers, method)
        except Exception as e:
            result = None
            st.error(f"Could not plan this network: {e}")
    if result is not None:
        st.session_state["plan"] = {"key": (area.slug, method, n_towers), "result": result}
        st.session_state.pop("compare", None)          # the old comparison belongs to the old plan
        if compare_now:
            if 5 <= n_towers <= 40:
                with st.spinner("Comparing greedy placement with the honeycomb..."):
                    try:
                        st.session_state["compare"] = {
                            "key": (area.slug, n_towers), "result": area.compare(n_towers)}
                    except Exception as e:
                        st.error(f"Could not compare: {e}")
            else:
                st.info("The comparison works for 5 to 40 towers.")

plan = st.session_state.get("plan")
if plan is not None and plan["key"] != (area.slug, method, n_towers):
    st.caption(f"The last picture was for {plan['key'][2]} towers ({plan['key'][1]}). "
               "Press **Plan and draw** to update it for the current settings.")
    plan = None
if plan is None:
    st.caption("Choose the placement and the number of towers, then press **Plan and draw**. "
               "Planning and drawing take a few seconds.")
else:
    r = plan["result"]
    picture_col, number_col = st.columns([1.7, 1.0])
    with picture_col:
        st.image(r["picture_file"], width="stretch")
        with open(r["picture_file"], "rb") as f:
            st.download_button("Download the picture", f.read(),
                               file_name=os.path.basename(r["picture_file"]), mime="image/png")
        st.caption("Triangles are towers, coloured lines show where the 3 sectors of each tower "
                   "point, and the letters F1, F2, ... are their frequency groups. The tint of each "
                   "square is the frequency group of the sector that serves it; no tint means no "
                   "signal. Background: " + str(r.get("background")) + ".")
    with number_col:
        st.metric("Demand covered", f"{r['demand_covered_percent']}%",
                  help="Share of the building footprint that has signal above the threshold.")
        st.metric("Grid squares covered", f"{r['squares_covered_percent']}%",
                  help="Share of the 100 m squares with signal. Squares with no buildings count "
                       "here too, so this is usually lower than demand coverage.")
        st.metric("Towers / sectors", f"{r['towers']} / {r['sectors']}")
        st.metric("Frequency groups", r["frequency_groups"],
                  help=f"The colouring used {r['frequency_groups']} groups. The lower bound from "
                       f"the largest set of mutually conflicting sectors is "
                       f"{r['frequency_groups_lower_bound']}.")
        st.metric("Sector conflicts", r["sector_conflicts"],
                  help="Pairs of sectors that would disturb each other; each pair must not share "
                       "a frequency group.")
        if "honeycomb_spacing_m" in r:
            st.caption(f"Honeycomb spacing {r['honeycomb_spacing_m']} m, "
                       f"rotation {r['honeycomb_rotation_deg']}°.")
        st.caption("Baseline assumptions: 15 dBi antenna, -100 dBm threshold, 10 dB conflict "
                   "margin. Demand is a proxy from building footprints, not real traffic.")

cmp = st.session_state.get("compare")
if cmp is not None and cmp["key"] == (area.slug, n_towers):
    c = cmp["result"]
    st.markdown(f"**Greedy against the honeycomb, {c['towers']} towers**")
    st.table({"Placement": ["Greedy", "Honeycomb"],
              "Demand covered %": [c["greedy_demand_percent"], c["honeycomb_demand_percent"]],
              "Grid squares covered %": [c["greedy_squares_percent"], c["honeycomb_squares_percent"]]})
    st.caption(c["note"] + ". Small differences should not be read as a clear winner.")

# --------------------------------------------------------------------------- chat
st.subheader("Ask the assistant")
st.caption("The assistant chooses a tool, the tool does the calculation, and the assistant "
           "explains the result. Every number in its answer comes from a tool run on the area "
           "above. It can also switch the area for you.")

have_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
if not have_key:
    st.info("No Gemini API key found, so the chat is off. Planning works without it. Put "
            "GEMINI_API_KEY in the .env file (local) or in the app secrets (Streamlit Cloud).")
else:
    _, chat_clear = st.columns([4, 1])
    with chat_clear:
        if st.session_state["messages"] and st.button("Clear chat"):
            st.session_state["messages"] = []
            st.session_state["chat"] = None
            st.rerun()

    if st.session_state["chat"] is None:
        st.session_state["chat"] = ag.make_chat(verbose=False)
    if st.session_state["chat"] is None:
        st.error("The chat could not start. Check GEMINI_API_KEY and GEMINI_MODEL.")


def show_calls(calls):
    """The tool calls behind an answer, so the numbers can be checked."""
    if not calls:
        return
    with st.expander(f"Tools used ({len(calls)})"):
        for call in calls:
            args = ", ".join(f"{k}={v}" for k, v in call["args"].items())
            st.markdown(f"`{call['tool']}({args})`")
            problem = call["result"].get("error") if isinstance(call["result"], dict) else None
            if problem:
                st.error(problem)


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for picture in message.get("pictures", []):
            if os.path.exists(picture):
                st.image(picture, width="stretch")
        show_calls(message.get("calls", []))
        if message.get("warning"):
            st.warning(message["warning"])

if have_key and st.session_state.get("chat") is not None:
    question = st.chat_input("For example: how many towers do we need for 95% of demand?")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            pt.set_area(area)      # make sure the tools work on the area this page shows
            with st.spinner("Thinking. The assistant may run several tools, and a new area can "
                            "take about a minute to download..."):
                try:
                    answer, calls = ag.ask(st.session_state["chat"], question, verbose=False)
                    api_error = None
                except Exception as e:
                    answer, calls, api_error = "", [], f"{type(e).__name__}: {e}"

            if api_error:
                st.error(f"The assistant could not answer: {api_error}")
                answer = "Something went wrong before I got an answer. Please try again."
                calls = []

            st.markdown(answer or "_(the model returned no text)_")
            pictures = [c["result"]["picture_file"] for c in calls
                        if isinstance(c["result"], dict) and c["result"].get("picture_file")]
            for picture in pictures:
                if os.path.exists(picture):
                    st.image(picture, width="stretch")
            show_calls(calls)

            warning = None
            bad = ag.unverified_numbers(answer, question) if answer else []
            if bad:
                warning = ("These numbers were not in any tool result of this turn: "
                           + ", ".join(bad) + ". The check is rough, but verify them.")
                st.warning(warning)

        st.session_state["messages"].append({
            "role": "assistant", "content": answer or "(no text)",
            "calls": calls, "pictures": pictures, "warning": warning})

        # The assistant may have switched the area with change_area. The panel and the sidebar
        # must follow it, and the old plan picture no longer belongs to the area.
        current = pt.current_area()
        if getattr(current, "slug", None) != area.slug:
            st.session_state["area"] = current
            st.session_state.pop("plan", None)
            st.session_state.pop("compare", None)
            st.session_state["_new_place"] = current.name
            st.session_state["_new_size"] = round(current.half * 2 / 1000, 1)
            st.rerun()
