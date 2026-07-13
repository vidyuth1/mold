"""
Mold Well Tracker
=================
Streamlit app that lets you upload many photos of molds (or plates) laid out
as a 15 x 8 grid (120 coordinates), click anywhere inside a coordinate's
cell on the image to mark it Empty or Present, and autosaves every click as
it happens.

Coordinate naming: columns A-O (15), rows 1-8 (8), e.g. "B3" = column B,
row 3.

How it works
------------
1. Upload up to MAX_IMAGES images at once (or add more later up to the cap).
2. Select an image from the picker to work on it.
3. Calibrate the grid ONCE per image: click the OUTER TOP-LEFT corner of
   the grid (just outside cell A1), then the OUTER BOTTOM-RIGHT corner
   (just outside cell O8). That rectangle is divided evenly into 15 x 8
   cells.
4. Switch to "Mark wells" mode and click ANYWHERE inside a cell to toggle
   it between Present (green) and Empty (red). Every click is written to
   disk immediately — no save button needed.

All state (which wells are empty + the grid calibration) is stored per-image
(keyed by a cheap name+size signature) in the `well_data/` folder as JSON,
so re-uploading the same image later restores exactly where you left off.

Performance design
------------------
• Decode/resize is cached via st.cache_data with a cheap string key so each
  unique image is decoded ONCE no matter how many times the script reruns.
• The expensive .getvalue() / PIL decode is ONLY called on the currently
  selected image — never looped across all uploads.
• Image identity is computed from metadata (name + file size), not full-
  content hashing, so building the picker list costs O(n) cheap string ops.
• The cache is bounded to MAX_IMAGES entries so memory can't grow unboundedly
  even across a long session of switching between many images.
• Picker options are stable signatures; only the display label changes via
  format_func, avoiding Streamlit widget state resets on every annotation.
• A combined CSV export reads only the tiny per-image JSON files — no image
  decoding required.
"""

import hashlib
import json
import os
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
COLS = 15                       # columns labeled A-O
ROWS = 8                        # rows labeled 1-8   (15 x 8 = 120 wells)
COL_LABELS = "ABCDEFGHIJKLMNO"

MAX_DISPLAY_WIDTH = 900         # working image width in pixels

# Maximum number of simultaneously uploaded images.
#
# Why 50?  Each decoded working copy is ~900 × 600 × 3 ≈ 1.6 MB.
# st.cache_data(max_entries=MAX_IMAGES) keeps at most this many decoded
# copies in memory (LRU eviction after that).  Streamlit also holds the
# original upload bytes in session state (~3-5 MB per file).
#
# Budget at 50 images:
#   cache : 50 × 1.6 MB  ≈  80 MB
#   session bytes: 50 × 4 MB ≈ 200 MB
#   total ≈ 280 MB — well inside the ~800 MB Streamlit Cloud limit.
#
# In practice the cache stays much smaller because only the ACTIVE image is
# decoded per rerun; the rest are served from cache hits or not touched at
# all.  Raising this limit to 75-100 is safe if you host on a machine with
# more RAM; lower it to 20-30 on shared/low-memory deployments.
MAX_IMAGES = 50

STATE_DIR = "well_data"
os.makedirs(STATE_DIR, exist_ok=True)

PRESENT_FILL = (46, 204, 113)   # green
EMPTY_FILL   = (231, 76,  60)   # red
CALIB_COLOR  = (52,  152, 219)  # blue
GRID_LINE    = (0,   0,   0, 180)

st.set_page_config(page_title="Mold Well Tracker", layout="wide")


# ---------------------------------------------------------------------------
# HELPERS — pure / cheap
# ---------------------------------------------------------------------------

def well_ids():
    """All 120 well ids in row-major order: A1..O1, A2..O2, … A8..O8."""
    return [f"{COL_LABELS[c]}{r + 1}" for r in range(ROWS) for c in range(COLS)]


def file_signature(uploaded_file) -> str:
    """Cheap O(1) identity for an uploaded file — uses metadata only.

    Uses name + size instead of hashing the full byte content.  This means
    two *different* files that happen to share both name and size would
    collide; acceptable trade-off for this use case (called out in README).
    """
    raw = f"{uploaded_file.name}:{uploaded_file.size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def picker_label(sig: str, name: str, state: dict) -> str:
    """Human-readable label for the image picker's format_func."""
    wells  = state.get("wells", {})
    empty  = sum(1 for v in wells.values() if v == "empty")
    calib  = "✓ calibrated" if state.get("calibration") else "needs calibration"
    return f"{name}  [{empty} empty · {calib}]"


# ---------------------------------------------------------------------------
# STATE PERSISTENCE
# ---------------------------------------------------------------------------

def state_path(sig: str) -> str:
    return os.path.join(STATE_DIR, f"{sig}.json")


def load_state(sig: str) -> dict:
    path = state_path(sig)
    data: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    if "wells" not in data or set(data["wells"].keys()) != set(well_ids()):
        data["wells"] = {w: "present" for w in well_ids()}
    if "calibration" not in data:
        data["calibration"] = None
    return data


def save_state(sig: str, data: dict) -> bool:
    """Write state to disk (autosave). Returns False on read-only filesystem."""
    try:
        with open(state_path(sig), "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        st.warning(
            "Could not write autosave file (read-only filesystem?).  "
            "Your changes are kept for this session — use the sidebar's "
            "'Download state (JSON)' button to save manually.",
            icon="⚠️",
        )
        return False


# ---------------------------------------------------------------------------
# IMAGE DECODE — cached, lazy (active image only), bounded
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=MAX_IMAGES, show_spinner=False)
def _decode_and_resize(_file_bytes: bytes, _sig: str, max_w: int) -> bytes:
    """Open, resize, and return the working image as PNG bytes.

    The leading underscore on `_file_bytes` tells Streamlit's cache
    machinery NOT to hash that argument when computing the cache key —
    only the cheap `_sig` string (and `max_w`) is hashed.  This is
    Streamlit's documented pattern for excluding large/unhashable
    arguments from cache-key computation.

    Returning PNG bytes (not a PIL object) keeps the cached value
    serialisable and lets the caller re-open cheaply with no re-decode.
    """
    img = Image.open(BytesIO(_file_bytes))
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_working_image(uploaded_file, sig: str) -> Image.Image:
    """Return the resized PIL image for the ACTIVE file only.

    .getvalue() is called exactly once per unique image (the result is
    cached).  For all other uploaded files we never call .getvalue() at
    all — just the cheap metadata-based `sig`.
    """
    raw_bytes = uploaded_file.getvalue()
    png_bytes  = _decode_and_resize(raw_bytes, sig, MAX_DISPLAY_WIDTH)
    return Image.open(BytesIO(png_bytes))


# ---------------------------------------------------------------------------
# GRID MATH
# ---------------------------------------------------------------------------

def default_calibration(img_w: int, img_h: int):
    mx = img_w  * 0.04
    my = img_h  * 0.06
    return [[mx, my], [img_w - mx, img_h - my]]


def compute_cell_bounds(calibration, img_w: int, img_h: int):
    """Return per-well pixel bounding boxes and the outer grid rectangle."""
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
            wid  = f"{COL_LABELS[c]}{r + 1}"
            cx0  = x1 + c * col_w
            cy0  = y1 + r * row_h
            bounds[wid] = (cx0, cy0, cx0 + col_w, cy0 + row_h)
    return bounds, (x1, y1, x2, y2)


def find_cell(x: float, y: float, grid_rect) -> str | None:
    """Return the well id whose cell contains (x, y), or None if outside."""
    x1, y1, x2, y2 = grid_rect
    if x < x1 or x > x2 or y < y1 or y > y2:
        return None
    col_w = (x2 - x1) / COLS
    row_h = (y2 - y1) / ROWS
    col = min(int((x - x1) // col_w), COLS - 1) if col_w > 0 else 0
    row = min(int((y - y1) // row_h), ROWS - 1) if row_h > 0 else 0
    return f"{COL_LABELS[col]}{row + 1}"


# ---------------------------------------------------------------------------
# OVERLAY RENDERING
# ---------------------------------------------------------------------------

def draw_grid_overlay(
    base_img: Image.Image,
    bounds: dict,
    wells: dict,
    show_labels: bool,
    calib_points=None,
) -> Image.Image:
    img   = base_img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)
    font  = ImageFont.load_default()

    for wid, (x0, y0, x1, y1) in bounds.items():
        present = wells.get(wid, "present") == "present"
        fill    = (*PRESENT_FILL, 80) if present else (*EMPTY_FILL, 130)
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=GRID_LINE)
        if show_labels:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            draw.text((cx, cy), wid, fill=(0, 0, 0, 255), font=font, anchor="mm")

    if calib_points:
        for (x, y) in calib_points:
            r = 9
            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                outline=(*CALIB_COLOR, 255),
                width=3,
            )

    return Image.alpha_composite(img, layer).convert("RGB")


# ---------------------------------------------------------------------------
# COMBINED EXPORT — reads only tiny JSON files, no image decoding
# ---------------------------------------------------------------------------

def build_combined_csv(sigs_and_names: list[tuple[str, str]]) -> bytes:
    rows = []
    for sig, name in sigs_and_names:
        data = load_state(sig)
        for wid, status in data["wells"].items():
            rows.append({"image": name, "well": wid, "status": status})
    if not rows:
        return b"image,well,status\n"
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# ===========================================================================
# UI
# ===========================================================================

st.title("🧫 Mold Well Tracker")
st.caption(
    f"Upload up to {MAX_IMAGES} mold photos, calibrate each grid once, then "
    "click anywhere inside a coordinate's cell to mark it Empty / Present. "
    "Every click is saved automatically."
)

# ---------------------------------------------------------------------------
# File upload (multiple files) with hard cap
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    f"Upload mold images (max {MAX_IMAGES})",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more images to get started.")
    st.stop()

# Enforce hard cap — truncate with a visible warning
if len(uploaded_files) > MAX_IMAGES:
    st.warning(
        f"You uploaded {len(uploaded_files)} images but the limit is "
        f"{MAX_IMAGES}.  Only the first {MAX_IMAGES} will be used.  "
        "Lower the limit constant MAX_IMAGES in app.py if you need a "
        "smaller cap for memory-constrained deployments.",
        icon="⚠️",
    )
    uploaded_files = uploaded_files[:MAX_IMAGES]

# Build cheap O(1)-per-file signature list — NO .getvalue() calls here
sigs_and_names: list[tuple[str, str]] = [
    (file_signature(f), f.name) for f in uploaded_files
]

# Map sig -> UploadedFile for O(1) lookup later
sig_to_file = {sig: f for (sig, _), f in zip(sigs_and_names, uploaded_files)}

# Load per-image state for ALL images (tiny JSON reads, no image decoding)
all_states: dict[str, dict] = {sig: load_state(sig) for sig, _ in sigs_and_names}

# ---------------------------------------------------------------------------
# Image picker — stable options (signatures), mutable display via format_func
#
# Why stable options?  If the options list itself contained the mutable label
# text (e.g. "B3.jpg [2 empty · calibrated]"), Streamlit's selectbox would
# reset the user's selection to item 0 every time they toggled a well and the
# label text changed underneath it.  Using opaque signatures as options and
# rendering the human-readable label only in format_func avoids this.
# ---------------------------------------------------------------------------
sig_list = [sig for sig, _ in sigs_and_names]
name_map  = {sig: name for sig, name in sigs_and_names}

def _format_option(sig: str) -> str:
    return picker_label(sig, name_map[sig], all_states[sig])

active_sig: str = st.selectbox(
    "Active image",
    options=sig_list,
    format_func=_format_option,
    label_visibility="collapsed" if len(sig_list) == 1 else "visible",
)

active_file  = sig_to_file[active_sig]
active_state = all_states[active_sig]

# Decode + resize ONLY the active image (lazy; all others are never touched)
work_img = get_working_image(active_file, active_sig)
img_w, img_h = work_img.size

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    mode        = st.radio("Mode", ["Mark wells", "Calibrate grid"], index=0)
    show_labels = st.checkbox("Show cell labels", value=True)

    st.divider()
    present_count = sum(1 for v in active_state["wells"].values() if v == "present")
    empty_count   = len(active_state["wells"]) - present_count
    c1, c2 = st.columns(2)
    c1.metric("Present", present_count)
    c2.metric("Empty",   empty_count)

    st.divider()
    if st.button("Reset all wells → Present", use_container_width=True):
        active_state["wells"] = {w: "present" for w in well_ids()}
        save_state(active_sig, active_state)
        st.rerun()
    if st.button("Reset calibration", use_container_width=True):
        active_state["calibration"] = None
        save_state(active_sig, active_state)
        st.rerun()

    st.divider()
    # Per-image downloads
    export_df = pd.DataFrame(
        [{"well": w, "status": s} for w, s in active_state["wells"].items()]
    )
    st.download_button(
        "Download this image's results (CSV)",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"mold_{active_sig}_wells.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download this image's state (JSON)",
        json.dumps(active_state, indent=2).encode("utf-8"),
        file_name=f"mold_{active_sig}_state.json",
        mime="application/json",
        use_container_width=True,
    )

    # Combined export across ALL uploaded images (no image decoding)
    st.divider()
    if len(uploaded_files) > 1:
        combined_csv = build_combined_csv(sigs_and_names)
        st.download_button(
            f"Download ALL {len(uploaded_files)} images (combined CSV)",
            combined_csv,
            file_name="mold_all_wells.csv",
            mime="text/csv",
            use_container_width=True,
        )

    restore_file = st.file_uploader(
        "Restore a previously downloaded state (JSON)",
        type=["json"],
        key="restore",
    )
    if restore_file is not None:
        try:
            restored = json.load(restore_file)
            if "wells" in restored:
                active_state["wells"].update(restored["wells"])
            if restored.get("calibration"):
                active_state["calibration"] = restored["calibration"]
            save_state(active_sig, active_state)
            st.success("State restored.")
            st.rerun()
        except (json.JSONDecodeError, KeyError):
            st.error("That file doesn't look like a valid state export.")

# ---------------------------------------------------------------------------
# Main panel — grid interaction
# ---------------------------------------------------------------------------
bounds, grid_rect = compute_cell_bounds(active_state["calibration"], img_w, img_h)

if mode == "Calibrate grid":
    st.subheader("Calibrate the grid")
    st.write(
        "Click the **outer top-left corner of the grid** (just outside "
        "coordinate A1), then click the **outer bottom-right corner** "
        f"(just outside coordinate {COL_LABELS[-1]}{ROWS}).  "
        f"The rectangle between those two clicks is divided evenly into "
        f"{COLS} × {ROWS} cells.  Click again to re-calibrate from scratch."
    )
    calib_pts = active_state["calibration"] or []
    overlay   = draw_grid_overlay(
        work_img, bounds, active_state["wells"], show_labels, calib_pts
    )
    click = streamlit_image_coordinates(overlay, key=f"calib_{active_sig}")

    last_key = f"last_calib_click_{active_sig}"
    if click is not None:
        sig = (click.get("x"), click.get("y"))
        if sig != (None, None) and st.session_state.get(last_key) != sig:
            st.session_state[last_key] = sig
            pts = active_state["calibration"] or []
            if len(pts) >= 2:
                pts = []
            pts.append([sig[0], sig[1]])
            active_state["calibration"] = pts
            save_state(active_sig, active_state)
            st.rerun()

    if active_state["calibration"] and len(active_state["calibration"]) == 2:
        st.success("Calibration complete — switch to 'Mark wells' in the sidebar.")
    elif active_state["calibration"] and len(active_state["calibration"]) == 1:
        st.info("First corner recorded.  Now click the outer bottom-right corner.")

else:  # Mark wells
    st.subheader("Click anywhere inside a coordinate's cell to toggle it")
    overlay = draw_grid_overlay(
        work_img, bounds, active_state["wells"], show_labels
    )
    click = streamlit_image_coordinates(overlay, key=f"mark_{active_sig}")

    last_key = f"last_mark_click_{active_sig}"
    if click is not None:
        sig = (click.get("x"), click.get("y"))
        if sig != (None, None) and st.session_state.get(last_key) != sig:
            st.session_state[last_key] = sig
            wid = find_cell(sig[0], sig[1], grid_rect)
            if wid:
                current = active_state["wells"][wid]
                active_state["wells"][wid] = "empty" if current == "present" else "present"
                save_state(active_sig, active_state)
                st.rerun()
            else:
                st.toast("That click landed outside the calibrated grid.")

st.caption(
    f"Legend: 🟢 present  🔴 empty  ·  "
    f"Grid: {COLS} cols (A–{COL_LABELS[-1]}) × {ROWS} rows (1–{ROWS}) = "
    f"{COLS * ROWS} wells  ·  "
    f"Image {sig_list.index(active_sig) + 1} of {len(sig_list)} uploaded"
)

with st.expander("Show empty well list"):
    empty_wells = [w for w, s in active_state["wells"].items() if s == "empty"]
    st.write(", ".join(empty_wells) if empty_wells else "None marked empty yet.")
