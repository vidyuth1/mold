"""
Mold Well Tracker
=================
Streamlit app that lets you upload up to 40 photos of molds (or plates) laid
out as a 15 x 8 grid (120 coordinates), auto-prompts calibration on the first
image, then steps through each image with Next/Previous buttons.

Coordinate naming: columns A-O (15), rows 1-8 (8), e.g. "B3" = column B, row 3.

How it works
------------
1. Upload up to MAX_IMAGES images at once.
2. The app automatically opens the first image and prompts calibration.
3. Click the OUTER TOP-LEFT corner of the grid (just outside A1), then the
   OUTER BOTTOM-RIGHT corner (just outside O8). That rectangle is divided
   evenly into 15 × 8 cells.
4. After calibration the mode switches to "Mark wells". Click ANYWHERE inside
   a cell to toggle it between Present (green) and Empty (red). Every click is
   written to disk immediately.
5. Click "Next →" to move to the next image (or "← Previous" to go back).
6. After all images are reviewed, use the Export tab to download:
   - A heatmap image showing missing-coordinate frequency across all molds.
   - An Excel workbook with two sheets: the heatmap data and a frequency table.
"""

import hashlib
import io
import json
import os
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import openpyxl
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
COLS = 15
ROWS = 8
COL_LABELS = "ABCDEFGHIJKLMNO"
MAX_DISPLAY_WIDTH = 900
MAX_IMAGES = 40

STATE_DIR = "well_data"
os.makedirs(STATE_DIR, exist_ok=True)

PRESENT_FILL = (46, 204, 113)
EMPTY_FILL   = (231, 76,  60)
CALIB_COLOR  = (52,  152, 219)
GRID_LINE    = (0,   0,   0, 180)

st.set_page_config(page_title="Mold Well Tracker", layout="wide")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def well_ids():
    """All 120 well ids in row-major order."""
    return [f"{COL_LABELS[c]}{r + 1}" for r in range(ROWS) for c in range(COLS)]


def file_signature(uploaded_file) -> str:
    raw = f"{uploaded_file.name}:{uploaded_file.size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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
    try:
        with open(state_path(sig), "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        st.warning(
            "Could not write autosave file (read-only filesystem?). "
            "Your changes are kept for this session.",
            icon="⚠️",
        )
        return False


# ---------------------------------------------------------------------------
# IMAGE DECODE — cached, lazy
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=MAX_IMAGES, show_spinner=False)
def _decode_and_resize(_file_bytes: bytes, _sig: str, max_w: int) -> bytes:
    img = Image.open(BytesIO(_file_bytes))
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_working_image(uploaded_file, sig: str) -> Image.Image:
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
# FREQUENCY / EXPORT DATA
# ---------------------------------------------------------------------------

def compute_frequency(sigs_and_names: list[tuple[str, str]]) -> dict:
    """Count how many images have each well marked empty."""
    freq = {w: 0 for w in well_ids()}
    n_images = len(sigs_and_names)
    for sig, _ in sigs_and_names:
        data = load_state(sig)
        for wid, status in data["wells"].items():
            if status == "empty":
                freq[wid] += 1
    return freq, n_images


def build_heatmap_figure(freq: dict, n_images: int) -> plt.Figure:
    """Build a matplotlib heatmap of missing-well frequency."""
    grid = np.zeros((ROWS, COLS), dtype=float)
    for r in range(ROWS):
        for c in range(COLS):
            wid = f"{COL_LABELS[c]}{r + 1}"
            grid[r, c] = freq.get(wid, 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.YlOrRd
    im = ax.imshow(grid, cmap=cmap, aspect="auto",
                   vmin=0, vmax=max(n_images, 1))

    # Annotate each cell with count and %
    for r in range(ROWS):
        for c in range(COLS):
            val = int(grid[r, c])
            pct = (val / n_images * 100) if n_images > 0 else 0
            color = "white" if val > n_images * 0.6 else "black"
            ax.text(c, r, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(COLS))
    ax.set_xticklabels(list(COL_LABELS), fontsize=9)
    ax.set_yticks(range(ROWS))
    ax.set_yticklabels([str(i + 1) for i in range(ROWS)], fontsize=9)
    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Row", fontsize=11)
    ax.set_title(
        f"Missing Coordinate Frequency Heatmap\n"
        f"{n_images} mold{'s' if n_images != 1 else ''} analyzed",
        fontsize=13, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("# Times Missing", fontsize=9)
    plt.tight_layout()
    return fig


def build_excel_export(freq: dict, n_images: int) -> bytes:
    """
    Build a two-sheet Excel workbook:
      Sheet 1 – Heatmap (color-coded grid of missing frequency)
      Sheet 2 – Frequency Table (coordinate, count, %)
    """
    wb = openpyxl.Workbook()

    # ---- SHEET 1: Heatmap ------------------------------------------------
    ws_heat = wb.active
    ws_heat.title = "Heatmap"

    thin = Side(style="thin", color="AAAAAA")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    header_font  = Font(name="Arial", bold=True, size=11, color="FFFFFF")
    label_font   = Font(name="Arial", bold=True, size=10)
    cell_font    = Font(name="Arial", size=9)
    center       = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Title row
    ws_heat.merge_cells("A1:Q1")
    title_cell = ws_heat["A1"]
    title_cell.value = f"Missing Coordinate Frequency Heatmap — {n_images} Mold(s) Analyzed"
    title_cell.font  = Font(name="Arial", bold=True, size=13)
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    title_cell.fill = PatternFill("solid", fgColor="2C3E50")
    title_cell.font = Font(name="Arial", bold=True, size=13, color="FFFFFF")
    ws_heat.row_dimensions[1].height = 30

    # Sub-header row (column letters A-O in columns B-P, row 2)
    ws_heat["A2"].value = "Row \\ Col"
    ws_heat["A2"].font  = header_font
    ws_heat["A2"].fill  = PatternFill("solid", fgColor="34495E")
    ws_heat["A2"].alignment = center
    ws_heat["A2"].border = border

    for c_idx, col_letter in enumerate(COL_LABELS):
        cell = ws_heat.cell(row=2, column=c_idx + 2)
        cell.value = col_letter
        cell.font  = header_font
        cell.fill  = PatternFill("solid", fgColor="34495E")
        cell.alignment = center
        cell.border = border

    ws_heat.row_dimensions[2].height = 22

    # Max frequency for color scaling
    max_freq = max(freq.values()) if freq else 1

    def freq_to_hex(count, maximum):
        """Map count → red gradient hex (white=0, deep red=max)."""
        if maximum == 0:
            return "FFFFFF"
        ratio = count / maximum
        # White (255,255,255) → Red (231,76,60)
        r = int(255 - ratio * (255 - 231))
        g = int(255 - ratio * (255 - 76))
        b = int(255 - ratio * (255 - 60))
        return f"{r:02X}{g:02X}{b:02X}"

    # Data rows (rows 3..10 → grid rows 1..8)
    for r_idx in range(ROWS):
        row_num = r_idx + 3
        ws_heat.row_dimensions[row_num].height = 28

        # Row label
        label_cell = ws_heat.cell(row=row_num, column=1)
        label_cell.value = str(r_idx + 1)
        label_cell.font  = label_font
        label_cell.fill  = PatternFill("solid", fgColor="ECF0F1")
        label_cell.alignment = center
        label_cell.border = border

        for c_idx in range(COLS):
            wid   = f"{COL_LABELS[c_idx]}{r_idx + 1}"
            count = freq.get(wid, 0)
            pct   = (count / n_images * 100) if n_images > 0 else 0
            hex_color = freq_to_hex(count, max_freq)

            data_cell = ws_heat.cell(row=row_num, column=c_idx + 2)
            data_cell.value = f"{count} ({pct:.0f}%)"
            data_cell.font  = Font(
                name="Arial", size=8, bold=count > 0,
                color="FFFFFF" if count > max_freq * 0.6 else "000000"
            )
            data_cell.fill  = PatternFill("solid", fgColor=hex_color)
            data_cell.alignment = center
            data_cell.border = border

    # Column widths
    ws_heat.column_dimensions["A"].width = 10
    for c_idx in range(COLS):
        ws_heat.column_dimensions[get_column_letter(c_idx + 2)].width = 10

    # Legend note
    note_row = ROWS + 4
    ws_heat.merge_cells(f"A{note_row}:Q{note_row}")
    note = ws_heat.cell(row=note_row, column=1)
    note.value = (
        f"Each cell shows: count of molds where this coordinate was missing "
        f"(percentage of {n_images} total molds). "
        f"Color intensity reflects frequency — deeper red = missing more often."
    )
    note.font = Font(name="Arial", italic=True, size=9, color="555555")
    note.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    ws_heat.row_dimensions[note_row].height = 30

    # ---- SHEET 2: Frequency Table ----------------------------------------
    ws_table = wb.create_sheet("Frequency Table")

    # Title
    ws_table.merge_cells("A1:E1")
    t = ws_table["A1"]
    t.value = f"Coordinate Flag Frequency Table — {n_images} Mold(s) Analyzed"
    t.font  = Font(name="Arial", bold=True, size=13, color="FFFFFF")
    t.fill  = PatternFill("solid", fgColor="2C3E50")
    t.alignment = Alignment(horizontal="center", vertical="center")
    ws_table.row_dimensions[1].height = 30

    # Column headers
    headers = ["Coordinate", "Column", "Row", "Times Missing", "% Missing"]
    header_fills = ["34495E"] * 5
    for col_i, (h, fill) in enumerate(zip(headers, header_fills), start=1):
        cell = ws_table.cell(row=2, column=col_i)
        cell.value = h
        cell.font  = Font(name="Arial", bold=True, size=10, color="FFFFFF")
        cell.fill  = PatternFill("solid", fgColor=fill)
        cell.alignment = center
        cell.border = border
    ws_table.row_dimensions[2].height = 22

    # Data — sorted by frequency descending
    all_wids = well_ids()
    sorted_wids = sorted(all_wids, key=lambda w: -freq.get(w, 0))

    for row_i, wid in enumerate(sorted_wids, start=3):
        count = freq.get(wid, 0)
        pct   = (count / n_images * 100) if n_images > 0 else 0
        col_letter = wid[0]
        row_number = wid[1:]

        hex_color = freq_to_hex(count, max_freq)
        text_color = "FFFFFF" if count > max_freq * 0.6 else "000000"

        vals = [wid, col_letter, row_number, count, round(pct, 1)]
        for col_i, val in enumerate(vals, start=1):
            cell = ws_table.cell(row=row_i, column=col_i)
            cell.value = val
            cell.border = border
            cell.alignment = center

            if col_i == 4:  # Times Missing
                cell.fill = PatternFill("solid", fgColor=hex_color)
                cell.font = Font(name="Arial", size=10, bold=True, color=text_color)
            elif col_i == 5:  # % Missing
                cell.fill = PatternFill("solid", fgColor=hex_color)
                cell.font = Font(name="Arial", size=10, color=text_color)
                cell.number_format = "0.0%"
                cell.value = pct / 100  # store as fraction for proper % format
            else:
                cell.font = Font(name="Arial", size=10)
                # Alternate row shading
                if row_i % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor="F8F9FA")

        ws_table.row_dimensions[row_i].height = 18

    # Column widths
    col_widths = [14, 10, 8, 16, 14]
    for col_i, w in enumerate(col_widths, start=1):
        ws_table.column_dimensions[get_column_letter(col_i)].width = w

    # Summary footer
    footer_row = len(sorted_wids) + 4
    ws_table.merge_cells(f"A{footer_row}:E{footer_row}")
    footer = ws_table.cell(row=footer_row, column=1)
    total_missing = sum(freq.values())
    footer.value = (
        f"Total missing observations across all coordinates: {total_missing}  |  "
        f"Molds analyzed: {n_images}  |  Sorted by frequency (highest first)"
    )
    footer.font = Font(name="Arial", italic=True, size=9, color="555555")
    footer.alignment = Alignment(horizontal="left", vertical="center")
    ws_table.row_dimensions[footer_row].height = 22

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ===========================================================================
# UI
# ===========================================================================

st.title("🧫 Mold Well Tracker")
st.caption(
    f"Upload up to {MAX_IMAGES} mold photos. The app will guide you through "
    "calibrating and marking each image, then export a frequency heatmap."
)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    f"Upload mold images (max {MAX_IMAGES})",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more images to get started.")
    st.stop()

if len(uploaded_files) > MAX_IMAGES:
    st.warning(
        f"You uploaded {len(uploaded_files)} images but the limit is {MAX_IMAGES}. "
        f"Only the first {MAX_IMAGES} will be used.",
        icon="⚠️",
    )
    uploaded_files = uploaded_files[:MAX_IMAGES]

sigs_and_names: list[tuple[str, str]] = [
    (file_signature(f), f.name) for f in uploaded_files
]
sig_to_file = {sig: f for (sig, _), f in zip(sigs_and_names, uploaded_files)}
all_states: dict[str, dict] = {sig: load_state(sig) for sig, _ in sigs_and_names}
sig_list  = [sig for sig, _ in sigs_and_names]
name_map  = {sig: name for sig, name in sigs_and_names}

n_total = len(sig_list)

# ---------------------------------------------------------------------------
# Session state: current image index
# ---------------------------------------------------------------------------
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
# Clamp in case files changed
st.session_state.current_idx = max(0, min(st.session_state.current_idx, n_total - 1))

# ---------------------------------------------------------------------------
# Top navigation tabs: Annotation | Export
# ---------------------------------------------------------------------------
tab_annotate, tab_export = st.tabs(["📷 Annotate Images", "📊 Export Results"])

# ===========================================================================
# TAB 1 — ANNOTATE
# ===========================================================================
with tab_annotate:

    idx = st.session_state.current_idx
    active_sig   = sig_list[idx]
    active_file  = sig_to_file[active_sig]
    active_state = all_states[active_sig]

    # Decode active image only
    work_img     = get_working_image(active_file, active_sig)
    img_w, img_h = work_img.size
    bounds, grid_rect = compute_cell_bounds(active_state["calibration"], img_w, img_h)

    # --- Header row: image progress + nav buttons -------------------------
    h_col1, h_col2, h_col3 = st.columns([1, 4, 1])
    with h_col1:
        if st.button("← Previous", disabled=(idx == 0), use_container_width=True):
            st.session_state.current_idx -= 1
            st.rerun()
    with h_col2:
        st.markdown(
            f"<div style='text-align:center;font-size:1.05rem;padding-top:6px;'>"
            f"<b>Image {idx + 1} of {n_total}</b> &nbsp;·&nbsp; "
            f"<code>{name_map[active_sig]}</code>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with h_col3:
        if st.button("Next →", disabled=(idx == n_total - 1), use_container_width=True):
            st.session_state.current_idx += 1
            st.rerun()

    st.progress((idx + 1) / n_total)

    # --- Sidebar controls -------------------------------------------------
    with st.sidebar:
        st.header("Controls")
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
        st.download_button(
            "Download this image (CSV)",
            pd.DataFrame(
                [{"well": w, "status": s} for w, s in active_state["wells"].items()]
            ).to_csv(index=False).encode("utf-8"),
            file_name=f"mold_{active_sig}_wells.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download this image (JSON)",
            json.dumps(active_state, indent=2).encode("utf-8"),
            file_name=f"mold_{active_sig}_state.json",
            mime="application/json",
            use_container_width=True,
        )

        restore_file = st.file_uploader(
            "Restore state (JSON)",
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
                st.error("Invalid state file.")

    # --- Auto-determine mode: calibrate first if not done -----------------
    needs_calibration = (
        not active_state["calibration"]
        or len(active_state["calibration"]) < 2
    )

    if needs_calibration:
        # ---- CALIBRATION MODE ----------------------------------------
        st.subheader("🎯 Step 1 — Calibrate the grid")

        calib_pts = active_state.get("calibration") or []
        if not calib_pts:
            st.info(
                "This image needs calibration. "
                "Click the **outer top-left corner** of the grid "
                f"(just outside coordinate A1)."
            )
        else:
            st.info(
                "First corner recorded ✓  "
                "Now click the **outer bottom-right corner** "
                f"(just outside coordinate {COL_LABELS[-1]}{ROWS})."
            )

        overlay = draw_grid_overlay(
            work_img, bounds, active_state["wells"], show_labels, calib_pts
        )
        click = streamlit_image_coordinates(overlay, key=f"calib_{active_sig}")

        last_key = f"last_calib_click_{active_sig}"
        if click is not None:
            sig_xy = (click.get("x"), click.get("y"))
            if sig_xy != (None, None) and st.session_state.get(last_key) != sig_xy:
                st.session_state[last_key] = sig_xy
                pts = active_state["calibration"] or []
                if len(pts) >= 2:
                    pts = []
                pts.append([sig_xy[0], sig_xy[1]])
                active_state["calibration"] = pts
                save_state(active_sig, active_state)
                st.rerun()

    else:
        # ---- MARK WELLS MODE -----------------------------------------
        st.subheader("Step 2 — Click a cell to toggle Empty / Present")

        col_info, col_nav = st.columns([5, 1])
        with col_info:
            st.caption(
                f"🟢 present &nbsp; 🔴 empty &nbsp;·&nbsp; "
                f"Grid: {COLS} cols (A–{COL_LABELS[-1]}) × {ROWS} rows"
            )

        overlay = draw_grid_overlay(
            work_img, bounds, active_state["wells"], show_labels
        )
        click = streamlit_image_coordinates(overlay, key=f"mark_{active_sig}")

        last_key = f"last_mark_click_{active_sig}"
        if click is not None:
            sig_xy = (click.get("x"), click.get("y"))
            if sig_xy != (None, None) and st.session_state.get(last_key) != sig_xy:
                st.session_state[last_key] = sig_xy
                wid = find_cell(sig_xy[0], sig_xy[1], grid_rect)
                if wid:
                    current = active_state["wells"][wid]
                    active_state["wells"][wid] = "empty" if current == "present" else "present"
                    save_state(active_sig, active_state)
                    st.rerun()
                else:
                    st.toast("Click landed outside the calibrated grid.")

        # Quick empty-well list
        with st.expander("Show empty well list"):
            empty_wells = [w for w, s in active_state["wells"].items() if s == "empty"]
            st.write(", ".join(empty_wells) if empty_wells else "None marked empty yet.")

        # Bottom navigation
        st.divider()
        nav1, nav2, nav3 = st.columns([1, 6, 1])
        with nav1:
            if st.button("← Prev", disabled=(idx == 0), use_container_width=True):
                st.session_state.current_idx -= 1
                st.rerun()
        with nav2:
            # Image thumbnail strip
            progress_labels = []
            for i, (s, n) in enumerate(sigs_and_names):
                state_i = all_states[s]
                calib_ok = state_i.get("calibration") and len(state_i["calibration"]) == 2
                icon = "✅" if calib_ok else "⏳"
                progress_labels.append(f"{icon} {i+1}")
            st.caption("  ".join(progress_labels))
        with nav3:
            if st.button("Next →", disabled=(idx == n_total - 1), use_container_width=True):
                st.session_state.current_idx += 1
                st.rerun()

            if idx == n_total - 1:
                st.success("All images reviewed! Head to the **Export Results** tab.")


# ===========================================================================
# TAB 2 — EXPORT
# ===========================================================================
with tab_export:

    freq, n_images = compute_frequency(sigs_and_names)
    n_calibrated = sum(
        1 for sig, _ in sigs_and_names
        if all_states[sig].get("calibration") and len(all_states[sig]["calibration"]) == 2
    )

    st.subheader(f"📊 Results Summary — {n_images} mold(s) uploaded, {n_calibrated} calibrated")

    if n_images == 0:
        st.info("No images uploaded yet.")
    else:
        # ---- Heatmap preview -------------------------------------------
        st.markdown("### Missing Coordinate Frequency Heatmap")
        st.caption(
            "Color intensity shows how often each coordinate was marked empty "
            "across all analyzed molds."
        )
        fig = build_heatmap_figure(freq, n_images)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Save heatmap as PNG for download
        heatmap_buf = BytesIO()
        fig2 = build_heatmap_figure(freq, n_images)
        fig2.savefig(heatmap_buf, format="PNG", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        heatmap_buf.seek(0)

        st.download_button(
            "⬇️ Download Heatmap (PNG)",
            heatmap_buf.getvalue(),
            file_name="mold_heatmap.png",
            mime="image/png",
            use_container_width=False,
        )

        st.divider()

        # ---- Frequency table preview ------------------------------------
        st.markdown("### Coordinate Flag Frequency Table")
        rows = []
        for wid in well_ids():
            count = freq.get(wid, 0)
            pct   = (count / n_images * 100) if n_images > 0 else 0
            rows.append({
                "Coordinate": wid,
                "Column": wid[0],
                "Row": wid[1:],
                "Times Missing": count,
                "% Missing": round(pct, 1),
            })

        df_freq = pd.DataFrame(rows).sort_values("Times Missing", ascending=False)
        st.dataframe(
            df_freq,
            use_container_width=True,
            hide_index=True,
            column_config={
                "% Missing": st.column_config.NumberColumn(format="%.1f%%"),
                "Times Missing": st.column_config.ProgressColumn(
                    min_value=0, max_value=n_images, format="%d"
                ),
            },
        )

        st.divider()

        # ---- Excel export -----------------------------------------------
        st.markdown("### Download Excel Report")
        st.caption(
            "The Excel workbook contains two sheets: **Heatmap** (color-coded grid) "
            "and **Frequency Table** (sorted list of all 120 coordinates)."
        )

        excel_bytes = build_excel_export(freq, n_images)
        st.download_button(
            "⬇️ Download Excel Report (.xlsx)",
            excel_bytes,
            file_name="mold_well_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
        )
