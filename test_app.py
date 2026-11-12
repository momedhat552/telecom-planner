"""Offline check of the web app (Phase 8). It starts app.py the way Streamlit does, without a
browser and without an API call: the saved Berlin files in data/ are loaded, the "Plan and draw"
button is pressed, and the numbers are compared with the README.

Run it with:  py test_app.py
It needs the Streamlit package (see requirements.txt) and the Phase 1 and 2 files in data/.
"""
from streamlit.testing.v1 import AppTest

EXPECTED = {"Buildings": "5185", "Built-up area": "29.9%", "Demand cells": "900"}


def button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise AssertionError(f"button '{label}' not found; found {[b.label for b in at.button]}")


def metrics(at):
    return {m.label: m.value for m in at.metric}


at = AppTest.from_file("app.py", default_timeout=180)
at.run()
assert not at.exception, at.exception
before = metrics(at)
for label, value in EXPECTED.items():
    assert before[label] == value, f"{label}: expected {value}, got {before[label]}"

# Default settings: greedy placement, 26 towers.
button(at, "Plan and draw").click().run()
assert not at.exception, at.exception
after = metrics(at)
assert after["Towers / sectors"] == "26 / 78", after["Towers / sectors"]
assert after["Demand covered"] == "95.1%", after["Demand covered"]
assert after["Grid squares covered"] == "90.8%", after["Grid squares covered"]
assert after["Frequency groups"] == "4", after["Frequency groups"]
assert after["Sector conflicts"] == "175", after["Sector conflicts"]

# Saved areas open without a download; a name written a little differently is matched to the
# saved copy ('Mohandseen, Giza, Egypt' -> 'Mohandseen, Dokki, Giza, Egypt').
options = [o for s in at.selectbox if s.label.startswith("Area saved") for o in s.options]
saved_option = next((o for o in options if "Mohandseen" in o), None)
if saved_option:
    at.selectbox[0].set_value(saved_option).run()
    button(at, "Open saved area").click().run()
    assert not at.exception, at.exception
    after_saved = metrics(at)
    assert after_saved["Buildings"] == "3651", after_saved["Buildings"]
    print("Saved area opened:", saved_option)
else:
    print("No saved Mohandseen area in areas/, skipped that check.")

print("App check passed:", {k: after[k] for k in ("Demand covered", "Grid squares covered",
                                                 "Towers / sectors", "Frequency groups")})
