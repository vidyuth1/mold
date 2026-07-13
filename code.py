"""
Mold Well Tracker
=================
Streamlit app that lets you upload a photo of a mold (or plate) laid out as
a 15 x 8 grid (120 coordinates), click anywhere inside a coordinate's cell
on the image to mark it Empty or Present, and autosaves every click as it
happens.

Coordinate naming: columns A-O (15), rows 1-8 (8), e.g. "B3" = column B,
row 3.

How it works
------------
1. Upload an image.
2. Calibrate the grid ONCE per image: click the OUTER TOP-LEFT corner of
   the grid (just outside cell A1), then the OUTER BOTTOM-RIGHT corner
   (just outside cell O8). That rectangle is divided evenly into 15 x 8
   cells.
3. Switch to "Mark wells" mode and click ANYWHERE inside a cell to toggle
   it between Present (green) and Empty (red) -- no need to hit a precise
   point, the whole cell area is clickable. Every click is written to disk
   immediately -- no save button needed.

All state (which wells are empty + the grid calibration) is stored per-image
(keyed by a hash of the image bytes) in the `well_data/` folder as JSON, so
re-uploading the same image later restores exactly where you left off.
"""

import hashlib
import json
import os
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

# --------------------------------------------------------------------------
# CONFIG - tweak these to match your mold
# --------------------------------------------------------------------------
COLS = 15                       # 15 columns -> labeled A-O
ROWS = 8                        # 8 rows     -> labeled 1-8   (15 x 8 = 120)
COL_LABELS = "ABCDEFGHIJKLMNO"  # 15 letters

MAX_DISPLAY_WIDTH = 900         # working image width in pixels

STATE_DIR = "well_data"         # autosave folder (per-image JSON files)
os.makedirs(STATE_DIR, exist_ok=True)

PRESENT_FILL = (46, 204, 113)   # green
EMPTY_FILL = (231, 76, 60)      # red
CALIB_COLOR = (52, 152, 219)    # blue
GRID_LINE = (0, 0, 0, 180)

st.set_page_config(page_title="Mold Well Tracker", layout="wide")

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def well_ids():
    """All 120 well ids: A1..O1, A2..O2, ... A8..O8 (row-major)."""
    return [f"{COL_LABELS[c]}{r + 1}" for r in range(ROWS) for c in range(COLS)]


def image_key(file_bytes: bytes) -> str:
    """Stable id for an uploaded image, used to namespace its saved state."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def state_path(key: str) -> str:
    return os.path.join(STATE_DIR, f"{key}.json")


def load_state(key: str) -> dict:
    path = state_path(key)
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    if "wells" not in data or set(data["wells"].keys()) != set(well_ids()):
        data["wells"] = {w: "present" for w in well_ids()}
    if "calibration" not in data:
        data["calibration"] = None  # -> [[x1, y1], [x2, y2]] once calibrated
    return data


def save_state(key: str, data: dict) -> bool:
    """Write state to disk immediately (this IS the autosave). Returns
    False (and warns) instead of crashing if the filesystem is read-only."""
    try:
        with open(state_path(key), "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        st.warning(
            "Could not write autosave file to disk (read-only filesystem?). "
            "Your changes are still kept for this session — use the "
            "'Download state (JSON)' button in the sidebar to save manually.",
            icon="⚠️",
        )
        return False


def default_calibration(img_w, img_h):
    """Outer corners of the whole grid rectangle when nothing is calibrated."""
    margin_x = img_w * 0.04
    margin_y = img_h * 0.06
    return [[margin_x, margin_y], [img_w - margin_x, img_h - margin_y]]


def compute_cell_bounds(calibration, img_w, img_h):
    """Divide the calibrated rectangle into COLS x ROWS equal cells.

    Returns (bounds, grid_rect):
      bounds     -> {well_id: (x0, y0, x1, y1)} pixel box for each cell
      grid_rect  -> (x1, y1, x2, y2) outer edges of the whole grid
    """
    if not calibration or len(calibration) != 2:
        calibration = default_calibration(img_w, img_h)
    (x1, y1), (x2, y2) = calibration
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    col_w = (x2 - x1) / COLS
    row_h = (y2 - y1) / ROWS
    bounds = {}
    for r in range(ROWS):
        for c in range(COLS):
            wid = f"{COL_LABELS[c]}{r + 1}"
            cx0 = x1 + c * col_w
            cy0 = y1 + r * row_h
            bounds[wid] = (cx0, cy0, cx0 + col_w, cy0 + row_h)
    return bounds, (x1, y1, x2, y2)


def find_cell(x, y, grid_rect):
    """Return the well id whose cell contains point (x, y), or None if the
    click landed outside the calibrated grid rectangle entirely."""
    x1, y1, x2, y2 = grid_rect
    if x < x1 or x > x2 or y < y1 or y > y2:
        return None
    col_w = (x2 - x1) / COLS
    row_h = (y2 - y1) / ROWS
    col = min(int((x - x1) // col_w), COLS - 1) if col_w > 0 else 0
    row = min(int((y - y1) // row_h), ROWS - 1) if row_h > 0 else 0
    return f"{COL_LABELS[col]}{row + 1}"


def draw_grid_overlay(base_img, bounds, wells, show_labels, calibration_points=None):
    img = base_img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.load_default()
    for wid, (x0, y0, x1, y1) in bounds.items():
        present = wells.get(wid, "present") == "present"
        fill = (*PRESENT_FILL, 80) if present else (*EMPTY_FILL, 130)
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=GRID_LINE)
        if show_labels:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            draw.text((cx, cy), wid, fill=(0, 0, 0, 255), font=font, anchor="mm")
    if calibration_points:
        for (x, y) in calibration_points:
            r = 9
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(*CALIB_COLOR, 255), width=3)
    return Image.alpha_composite(img, layer).convert("RGB")


def resize_working(img: Image.Image, max_w: int) -> Image.Image:
    if img.width <= max_w:
        return img.copy()
    ratio = max_w / img.width
    return img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🧫 Mold Well Tracker")
st.caption(
    f"Upload a mold photo, calibrate the {COLS}x{ROWS} grid once, then click "
    "anywhere inside a coordinate's cell to mark it Empty / Present. "
    "Every click is saved automatically."
)

uploaded = st.file_uploader("Upload mold image", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

file_bytes = uploaded.getvalue()
key = image_key(file_bytes)
state = load_state(key)

raw_img = Image.open(BytesIO(file_bytes))
work_img = resize_working(raw_img, MAX_DISPLAY_WIDTH)
img_w, img_h = work_img.size

# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Mark wells", "Calibrate grid"], index=0)
    show_labels = st.checkbox("Show cell labels", value=True)

    st.divider()
    present_count = sum(1 for v in state["wells"].values() if v == "present")
    empty_count = len(state["wells"]) - present_count
    c1, c2 = st.columns(2)
    c1.metric("Present", present_count)
    c2.metric("Empty", empty_count)

    st.divider()
    if st.button("Reset all wells to Present", use_container_width=True):
        state["wells"] = {w: "present" for w in well_ids()}
        save_state(key, state)
        st.rerun()
    if st.button("Reset calibration", use_container_width=True):
        state["calibration"] = None
        save_state(key, state)
        st.rerun()

    st.divider()
    export_df = pd.DataFrame(
        [{"well": w, "status": s} for w, s in state["wells"].items()]
    )
    st.download_button(
        "Download results (CSV)",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"mold_{key}_wells.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download state (JSON)",
        json.dumps(state, indent=2).encode("utf-8"),
        file_name=f"mold_{key}_state.json",
        mime="application/json",
        use_container_width=True,
    )

    restore_file = st.file_uploader(
        "Restore a previously downloaded state (JSON)", type=["json"], key="restore"
    )
    if restore_file is not None:
        try:
            restored = json.load(restore_file)
            if "wells" in restored:
                state["wells"].update(restored["wells"])
            if restored.get("calibration"):
                state["calibration"] = restored["calibration"]
            save_state(key, state)
            st.success("State restored.")
            st.rerun()
        except (json.JSONDecodeError, KeyError):
            st.error("That file doesn't look like a valid state export.")

bounds, grid_rect = compute_cell_bounds(state["calibration"], img_w, img_h)

# ---- main panel ------------------------------------------------------
if mode == "Calibrate grid":
    st.subheader("Step 1 — Calibrate the grid")
    st.write(
        "Click the **outer top-left corner of the grid** (just outside "
        "coordinate A1), then click the **outer bottom-right corner** "
        f"(just outside coordinate {COL_LABELS[-1]}{ROWS}). The rectangle "
        f"between those two clicks is divided evenly into {COLS} x {ROWS} "
        "cells. Click again afterwards to re-calibrate from scratch."
    )
    calib_points = state["calibration"] or []
    overlay = draw_grid_overlay(work_img, bounds, state["wells"], show_labels, calib_points)
    click = streamlit_image_coordinates(overlay, key=f"calib_{key}")

    last_key = f"last_calib_click_{key}"
    if click is not None:
        click_sig = (click.get("x"), click.get("y"))
        if click_sig != (None, None) and st.session_state.get(last_key) != click_sig:
            st.session_state[last_key] = click_sig
            pts = state["calibration"] or []
            if len(pts) >= 2:
                pts = []
            pts.append([click_sig[0], click_sig[1]])
            state["calibration"] = pts
            save_state(key, state)
            st.rerun()

    if state["calibration"] and len(state["calibration"]) == 2:
        st.success("Calibration complete — switch to 'Mark wells' in the sidebar.")
    elif state["calibration"] and len(state["calibration"]) == 1:
        st.info("First corner recorded. Now click the bottom-right outer corner.")

else:
    st.subheader("Click anywhere inside a coordinate's cell to toggle it")
    overlay = draw_grid_overlay(work_img, bounds, state["wells"], show_labels)
    click = streamlit_image_coordinates(overlay, key=f"mark_{key}")

    last_key = f"last_mark_click_{key}"
    if click is not None:
        click_sig = (click.get("x"), click.get("y"))
        if click_sig != (None, None) and st.session_state.get(last_key) != click_sig:
            st.session_state[last_key] = click_sig
            wid = find_cell(click_sig[0], click_sig[1], grid_rect)
            if wid:
                state["wells"][wid] = "empty" if state["wells"][wid] == "present" else "present"
                save_state(key, state)
                st.rerun()
            else:
                st.toast("That click landed outside the calibrated grid.")

st.caption(
    f"Legend: 🟢 present &nbsp;&nbsp; 🔴 empty &nbsp;&nbsp; "
    f"Grid: {COLS} cols (A-{COL_LABELS[-1]}) x {ROWS} rows (1-{ROWS}) = {COLS * ROWS} wells."
)

with st.expander("Show empty well list"):
    empty_wells = [w for w, s in state["wells"].items() if s == "empty"]
    st.write(", ".join(empty_wells) if empty_wells else "None marked empty yet.")
