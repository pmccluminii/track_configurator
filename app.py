# app.py — v2.5.2
# - Text-only dimensions with outward normals
# - Leg dimension labels are two lines ("Leg N" and "<len> m")
# - Segment length labels rotate 90° on vertical legs (left/right)
# - Keeps: cover strip, mounting hardware per meter, U legs, Excel stock checkboxes, shortest-donor cuts, label backers

import json, math, os, re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from shared_logic import (
    LayoutSpec, MidComponent, pack_segments,
    path_points_for_shape, path_lengths, place_mid_components,
    MIN_SEGMENT_HARD_M, MIN_SEGMENT_WARN_M
)

st.set_page_config(page_title="Track Layout Maker (Streamlit)", layout="wide")
st.title("Track Layout Maker — (Metric) v2.5.1")

with st.expander("Instructions", expanded=False):
    st.markdown(
        """
        **Before you start**
        - Gather the run dimensions you need (overall length, leg lengths, widths).
        - Know which stock lengths you can cut from (e.g. 2 m, 1 m sticks).

        **Using the calculator**
        1. Pick the layout shape in the sidebar and type in every required dimension. For U-shapes, fill in all three legs.
        2. Tick the stock lengths you actually have on hand so the app knows which pieces it can use.
        3. Scroll down the sidebar to select end feeds, corners, inline joins, and enter any mid-run parts using `position:PARTNO`.
        4. Watch the warnings banner over the preview. Anything under 0.18 m is an error; aim for 0.36 m or longer to leave room for luminaires. Adjust dimensions or stock selections if warnings appear.
        **Electrical loading note**
        - This calculator does **not** check fixture power draw. If your layout exceeds the remote AC→48 V DC supply limit, split the run with an isolator (inline or corner) and re-feed, or break the track with end caps and feed each section separately.
        - Always confirm with your electrical guidelines that feeds, breakers, and cables are sized correctly.
        """
    )

scroll_helper = """
<div style='margin: 8px 0;'>
  <a href="#bom" style="text-decoration:none;">
    <strong>Need the parts list?</strong> Click here to jump to the Bill of Materials table and download button.
  </a>
</div>
"""
st.markdown(scroll_helper, unsafe_allow_html=True)

# =========================================================
# Excel-driven Options
# =========================================================
EXCEL_FILE = "MultisysBOM_Options.xlsx"

if os.path.exists(EXCEL_FILE):
    try:
        xls = pd.ExcelFile(EXCEL_FILE)
        df_opts = None
        for s in xls.sheet_names:
            df_try = pd.read_excel(xls, sheet_name=s)
            cols = [c.strip().upper() for c in df_try.columns.astype(str)]
            if {"NAME","TYPE"}.issubset(set(cols)):
                df_opts = df_try
                break
        if df_opts is None:
            raise ValueError("No sheet with columns Name/Type found")
        df_mount = pd.read_excel(xls, sheet_name="MountingHardware") if "MountingHardware" in xls.sheet_names else None
    except Exception as e:
        st.error(f"Failed to read {EXCEL_FILE}: {e}")
        df_opts = pd.DataFrame(columns=["Name","Type","BOM SURFACE","BOM RECESSED","BOM RECESSED TRIMLESS","BOM SUSPENDED"])
        df_mount = None
else:
    st.warning(f"Options file not found: {EXCEL_FILE}. Using minimal fallbacks.")
    df_opts = pd.DataFrame(columns=["Name","Type","BOM SURFACE","BOM RECESSED","BOM RECESSED TRIMLESS","BOM SUSPENDED"])
    df_mount = None

df_opts = df_opts.fillna("")
if df_mount is not None:
    df_mount = df_mount.fillna("")
for col in df_opts.columns:
    if df_opts[col].dtype == object:
        df_opts[col] = df_opts[col].astype(str).str.strip()

def options_for_type(t):
    return sorted(df_opts.loc[df_opts["Type"].str.lower() == str(t).lower(), "Name"].unique().tolist())

options_by_type = {
    "Track":  options_for_type("Track"),
    "End":    options_for_type("End"),
    "Join":   options_for_type("Join"),
    "Corner": options_for_type("Corner"),
}

PROFILE_TO_COL = {
    "Surface": "BOM SURFACE",
    "Recessed": "BOM RECESSED",
    "Recessed Trimless": "BOM RECESSED TRIMLESS",
    "Suspended": "BOM SUSPENDED",
}

def apply_finish_tokens(bom_cell, finish_token):
    if not bom_cell:
        return []
    parts = [p.strip() for p in str(bom_cell).split("|") if p.strip()]
    return [p.replace("**", finish_token).replace("/*", finish_token) for p in parts]

def _norm(s): return str(s).strip().lower()
def _safe_index(options, value, default_idx=0):
    if not options:
        return 0
    try:
        return options.index(value)
    except ValueError:
        cmap = {_norm(o): i for i, o in enumerate(options)}
        return min(cmap.get(_norm(value), default_idx), len(options)-1)

# Parse "Track" names -> numeric metres
track_name_to_len_all = {}
for nm in options_by_type.get("Track", []):
    m = re.search(r'(\d+(?:\.\d+)?)\s*m\b', nm.lower())
    if m:
        track_name_to_len_all[nm] = float(m.group(1))
available_track_lengths = sorted(set(track_name_to_len_all.values()))

# =========================================================
# Sidebar — System
# =========================================================
with st.sidebar:
    st.header("System")
    track_profile = st.radio("Track profile", list(PROFILE_TO_COL.keys()), index=0, horizontal=True)
    finish = st.radio("Finish", ["Black","White"], index=0, horizontal=True)
    finish_token = "BK" if finish.lower().startswith("b") else "WH"

# =========================================================
# Sidebar — Geometry & Stock
# =========================================================
with st.sidebar:
    st.header("Inputs")
    name = st.text_input("Layout Name", "Layout 01")
    shape = st.selectbox("Shape", ["Straight","L","Rectangle","U"], index=0)

    # U with independent legs
    if shape == "U":
        leg1 = st.number_input("U — Leg 1 (m)", min_value=0.1, value=1.50, step=0.01, format="%.2f")
        base = st.number_input("U — Base (m)",  min_value=0.1, value=2.00, step=0.01, format="%.2f")
        leg3 = st.number_input("U — Leg 3 (m)", min_value=0.1, value=1.50, step=0.01, format="%.2f")
        length = base; width = None; depth = (leg1, base, leg3)
    else:
        length = st.number_input("Length (m)", min_value=0.1, value=5.0, step=0.01, format="%.2f")
        width = depth = None
        if shape in ("L","Rectangle"):
            width = st.number_input("Width (m)", min_value=0.1, value=2.0, step=0.01, format="%.2f")

    # Stock lengths from Excel as checkboxes
    st.subheader("Stock lengths (from Excel)")
    stock_selected = []
    if available_track_lengths:
        cols = st.columns(min(3, len(available_track_lengths)))
        for idx, L in enumerate(available_track_lengths):
            with cols[idx % len(cols)]:
                if st.checkbox(f"{L:.2f} m", value=True, key=f"stock_{L}"):
                    stock_selected.append(float(L))
    else:
        st.info("No track lengths found in Excel ‘Track’ options. Using fallback 2 m and 1 m.")
        stock_selected = [2.0, 1.0]

    maxrun = st.text_input("Max run (m) [optional]", "")

# Accessories
with st.sidebar:
    st.header("Accessories")
    cover_strip_on = st.toggle("Include cover strip (linear, matches total track length)", value=False)
    cover_name_ui  = st.text_input("Cover strip name (BOM row)", "Cover Strip (linear)")
    cover_part_ui  = st.text_input("Cover strip part no (optional)", "")

    st.subheader("Mounting hardware (per meter)")
    mh_auto = None
    if df_mount is not None and not df_mount.empty:
        def _col(df, name):
            for c in df.columns:
                if c.strip().lower() == name.lower(): return c
            return None
        c_prof = _col(df_mount, "Profile"); c_name = _col(df_mount, "Name")
        c_pn = _col(df_mount, "PartNo"); c_sp = _col(df_mount, "Spacing_m")
        if all([c_prof, c_name, c_pn, c_sp]):
            mh_candidates = df_mount[df_mount[c_prof].astype(str).str.strip().str.lower() == track_profile.strip().lower()]
            if not mh_candidates.empty:
                row0 = mh_candidates.iloc[0]
                try: spacing_val = float(row0[c_sp]) if str(row0[c_sp]).strip() else None
                except: spacing_val = None
                mh_auto = {"name": str(row0[c_name]).strip(), "part": str(row0[c_pn]).strip(), "spacing_m": spacing_val}

    use_mount  = st.toggle("Include mounting hardware", value=bool(mh_auto))
    mh_name    = st.text_input("Mounting hardware name", mh_auto["name"] if (use_mount and mh_auto) else "")
    mh_part    = st.text_input("Mounting hardware part no", mh_auto["part"] if (use_mount and mh_auto) else "")
    mh_spacing = st.number_input("Spacing (m) between supports", min_value=0.1, value=float(mh_auto["spacing_m"]) if (use_mount and mh_auto and mh_auto["spacing_m"]) else 1.00, step=0.1, format="%.2f")

# Base spec
base_spec = LayoutSpec(
    name=name, shape=shape, length_m=length, width_m=width, depth_m=depth,
    stock=stock_selected or [2.0, 1.0],
    max_run_m=float(maxrun) if str(maxrun).strip() else None,
    start_end=None, end_end=None,
    corner1_join=None, corner2_join=None, corner3_join=None,
    mid_components=[]
)

# =========================================================
# Helpers
# =========================================================
def pts_for_spec(spec):
    if spec.shape == "U" and isinstance(spec.depth_m, (list, tuple)) and len(spec.depth_m) == 3:
        L1, B, L3 = float(spec.depth_m[0]), float(spec.depth_m[1]), float(spec.depth_m[2])
        # Build path like an "n"; we'll flip for rendering
        return [(0.0, 0.0), (0.0, L1), (B, L1), (B, L1 - L3)]
    return path_points_for_shape(spec.shape, spec.length_m, spec.width_m, spec.depth_m)

def prefer_smallest_cut(segs, stock):
    if not segs: return segs
    stock_sorted = sorted(float(s) for s in (stock or []))
    out = []
    for s in segs:
        try:
            kind = getattr(s, "kind", None); length_m = float(getattr(s, "length_m"))
        except Exception:
            out.append(s); continue
        if kind == "cut":
            if stock_sorted:
                candidates = [L for L in stock_sorted if L + 1e-9 >= length_m]
                best = candidates[0] if candidates else stock_sorted[-1]
            else:
                best = length_m
            try: setattr(s, "cut_from", float(best))
            except Exception: pass
        out.append(s)
    return out

def donor_length_for_segment(s):
    cf = getattr(s, "cut_from", None)
    try:
        return float(cf) if isinstance(cf, (int, float)) else float(getattr(s, "length_m", 0.0))
    except Exception:
        return float(getattr(s, "length_m", 0.0))

def compute_plan(spec):
    """
    Compute geometry + basic cut stats and apply minimum-length rules:
      - Standalone run (single-leg Straight): must be ≥ 0.34 m to be useful (end feed + 1 light).
      - Any leg in a multi-leg shape: hard minimum ≥ 0.18 m; to fit a light in that leg, ≥ 0.36 m.
      - Individual segments (cuts/sticks) follow the same thresholds as legs.
    We return 'rules' with items of form {'level': 'error'|'warn', 'msg': '...'}.
    """
    pts = pts_for_spec(spec)
    seg_lens, total_len = path_lengths(pts)
    total_cuts = 0
    total_waste = 0.0
    rules = []
    leg_min_hard = MIN_SEGMENT_HARD_M
    leg_min_warn = MIN_SEGMENT_WARN_M

    # Cut stats (unchanged)
    for i in range(len(pts) - 1):
        leg_len = seg_lens[i]
        segs = prefer_smallest_cut(pack_segments(leg_len, spec.stock, spec.max_run_m), spec.stock)
        seg_count = len(segs)
        for seg_idx, s in enumerate(segs, start=1):
            seg_length = float(getattr(s, "length_m", 0.0))
            donor = donor_length_for_segment(s)
            waste = max(0.0, donor - seg_length)
            total_waste += waste
            if getattr(s, "kind", "") == "cut":
                total_cuts += 1
            same_as_leg = seg_count == 1 and abs(seg_length - leg_len) < 1e-6
            if seg_length < leg_min_hard - 1e-9 and not same_as_leg:
                rules.append({
                    "level": "error",
                    "msg": f"Leg {i+1} segment {seg_idx} is {seg_length:.2f} m, below the minimum {leg_min_hard:.2f} m. Combine it with an adjacent segment or cut a longer piece."
                })
            elif seg_length < leg_min_warn - 1e-9 and not same_as_leg:
                rules.append({
                    "level": "warn",
                    "msg": f"Leg {i+1} segment {seg_idx} is {seg_length:.2f} m. It clears {leg_min_hard:.2f} m, but to fit a light allow ≥ {leg_min_warn:.2f} m."
                })

    # Minimum-length validations
    num_legs = max(0, len(pts) - 1)
    if spec.shape == "Straight" and num_legs == 1:
        run_len = seg_lens[0] if seg_lens else 0.0
        if run_len < 0.34 - 1e-9:
            rules.append({
                "level": "error",
                "msg": f"Standalone run is {run_len:.2f} m, below the minimum 0.34 m needed for an end feed + 1 light."
            })
        elif run_len < 0.36 - 1e-9:
            rules.append({
                "level": "warn",
                "msg": f"Standalone run is {run_len:.2f} m. It meets 0.34 m minimum, but to fit a light comfortably use ≥ 0.36 m."
            })
    else:
        # Multi-leg shapes
        for i in range(num_legs):
            Lm = seg_lens[i]
            if Lm < leg_min_hard - 1e-9:
                rules.append({
                    "level": "error",
                    "msg": f"Leg {i+1} is {Lm:.2f} m, below the hard minimum {leg_min_hard:.2f} m."
                })
            elif Lm < leg_min_warn - 1e-9:
                rules.append({
                    "level": "warn",
                    "msg": f"Leg {i+1} is {Lm:.2f} m. It clears {leg_min_hard:.2f} m, but is too short to fit a light. Use ≥ {leg_min_warn:.2f} m to allow a luminaire."
                })

    # Mid-component validations
    if getattr(spec, "mid_components", None):
        for idx, mc in enumerate(spec.mid_components, start=1):
            try:
                pos_m = float(mc.pos_m)
            except (TypeError, ValueError):
                continue
            if pos_m > total_len + 1e-6:
                rules.append({
                    "level": "warn",
                    "msg": f"Mid component {idx} is placed at {pos_m:.2f} m, beyond the total layout length {total_len:.2f} m. It will be clamped to the end."
                })
            elif pos_m < -1e-6:
                rules.append({
                    "level": "warn",
                    "msg": f"Mid component {idx} is placed at {pos_m:.2f} m, before the start of the run. It will be clamped to the beginning."
                })

    return dict(
        pts=pts,
        seg_lens=seg_lens,
        total_len=total_len,
        total_cuts=total_cuts,
        total_excess=total_waste,
        rules=rules
    )

plan = compute_plan(base_spec)

# Quick banner for rule status
if plan.get("rules"):
    n_err = sum(1 for r in plan["rules"] if r.get("level") == "error")
    n_warn = sum(1 for r in plan["rules"] if r.get("level") == "warn")
    st.warning(f"Validation: {n_err} error(s), {n_warn} warning(s) found.")
else:
    st.success("Validation: no minimum-length issues.")

def title_text_for_spec(spec, total_len, cover_on):
    parts = [spec.name, "—", spec.shape]
    if spec.shape in ("L","Rectangle"):
        if spec.length_m is not None and spec.width_m is not None:
            parts.append(f"({spec.length_m:.2f} m × {spec.width_m:.2f} m)")
        elif spec.length_m is not None:
            parts.append(f"({spec.length_m:.2f} m)")
    elif spec.shape == "U":
        if isinstance(spec.depth_m, (list, tuple)) and len(spec.depth_m) == 3:
            L1,B,L3 = spec.depth_m
            parts.append(f"(L1 {float(L1):.2f} m × Base {float(B):.2f} m × L3 {float(L3):.2f} m)")
        elif spec.length_m is not None and spec.depth_m is not None:
            parts.append(f"({spec.length_m:.2f} m × {float(spec.depth_m):.2f} m)")
    else:
        if spec.length_m is not None:
            parts.append(f"({spec.length_m:.2f} m)")
    parts.append(f"• Total {total_len:.2f} m")
    parts.append("(with cover strip)" if cover_on else "(without cover strip)")
    return " ".join(parts)

# =========================================================
# Label backer (multiline + rotation)
# =========================================================
def draw_text_with_backer(
    x, y, text, anchor="middle", cls="lenLabel",
    px=12, pad_x=6, pad_y=3, back_opacity=0.5, rotate_deg=0
):
    lines = str(text).split("\n")
    if not lines:
        lines = [""]

    est_w_per_char = 0.6 * px
    line_w = [max(12.0, est_w_per_char * len(line)) for line in lines]
    w = max(line_w) + 2 * pad_x
    line_h = px * 1.25
    h = line_h * len(lines) + 2 * pad_y

    # Anchor rect around (x,y)
    x_left = x - (w / 2.0) if anchor == "middle" else x
    y_top = y - (h * 0.5)

    rect = (
        f'<rect x="{x_left:.2f}" y="{y_top:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="#fff" fill-opacity="{back_opacity:.2f}" rx="3" ry="3"/>'
    )

    start_baseline = y_top + pad_y + px
    text_open = f'<text class="{cls}" x="{x:.2f}" y="{start_baseline:.2f}" text-anchor="{anchor}">'
    tspans = [lines[0] if lines[0] else " "]
    for line in lines[1:]:
        tspans.append(f'<tspan x="{x:.2f}" dy="{line_h:.2f}">{line if line else " "}</tspan>')
    text_close = "</text>"

    if abs(rotate_deg) > 1e-3:
        return f'<g transform="rotate({rotate_deg:.2f} {x:.2f} {y:.2f})">{rect}{text_open}{"".join(tspans)}{text_close}</g>'
    else:
        return rect + text_open + "".join(tspans) + text_close

# =========================================================
# Geometry helpers for outward normals
# =========================================================
def node_normal(pts_px, idx):
    n = len(pts_px)
    if n <= 1: return (0.0, -1.0)
    if idx <= 0:
        x1,y1 = pts_px[0]; x2,y2 = pts_px[1]
    elif idx >= n-1:
        x1,y1 = pts_px[-2]; x2,y2 = pts_px[-1]
    else:
        x1,y1 = pts_px[idx-1]; x2,y2 = pts_px[idx+1]
    vx, vy = x2 - x1, y2 - y1
    L = math.hypot(vx, vy) or 1.0
    nx, ny = -vy / L, vx / L
    return nx, ny

def outward_normal(nx, ny, px, py, cx, cy):
    vx, vy = (px - cx), (py - cy)
    return (nx, ny) if (nx * vx + ny * vy) >= 0 else (-nx, -ny)

# =========================================================
# Renderer (text-only dimensions; outward normals; rotated vertical segment labels)
# =========================================================
def fit_layout(pts_m, target_w_units, padding_units=24, y_down=False):
    xs = [p[0] for p in pts_m] or [0.0]
    ys = [p[1] for p in pts_m] or [0.0]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    span_x = max(maxx - minx, 1e-9); span_y = max(maxy - miny, 1e-9)
    S = (target_w_units - 2*padding_units) / span_x
    S = min(S, 150.0)
    W = span_x * S + 2*padding_units; H = span_y * S + 2*padding_units; H = max(H, 240)
    def to_units(p):
        x_m, y_m = p
        x = padding_units + (x_m - minx) * S
        y = padding_units + (maxy - y_m) * S if y_down else padding_units + (y_m - miny) * S
        return x, y
    return [to_units(p) for p in pts_m], W, H, S, to_units

def render_track_svg(spec, plan, style, max_w_px=900):
    pad = style.get("pad", 28)
    pts_m = plan["pts"]
    seg_lens = plan["seg_lens"]
    total_len = plan["total_len"]
    rotate_side_labels = spec.shape in {"L", "Rectangle", "U"}
    side_label_angle = -90.0
    dim_side_extra = style.get("dim_side_extra", 0.0)

    # Visual flip for U to render as "U" (not "n")
    if spec.shape == "U":
        pts_m = [(x, -y) for (x, y) in pts_m]

    pts_px, w, h, S, to_px = fit_layout(pts_m, target_w_units=max_w_px, padding_units=pad, y_down=True)

    # canvas center for outward tests
    xs_px = [p[0] for p in pts_px]; ys_px = [p[1] for p in pts_px]
    cx = (min(xs_px) + max(xs_px)) * 0.5
    cy = (min(ys_px) + max(ys_px)) * 0.5

    auto_extra = 0
    if style.get("auto_bottom_buffer", True):
        auto_extra = int(style["dim_off"] + max(style["seg_label_off"], style["join_label_off"], style["corner_label_off"], style["end_label_off"]) + style["font_px"]*1.6 + style["node_size"]*0.6)

    w_out = w
    h_out = h + style.get("extra_top", 0) + style.get("extra_bottom", 0) + auto_extra

    def _svg_header(w, h):
        return f'''<svg viewBox="0 0 {w:.2f} {h:.2f}" width="100%" height="{h:.0f}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
  <style>
    svg {{ overflow: visible; }}
    .track {{ stroke:#111; stroke-width:{style["track_stroke"]}; fill:none; stroke-linecap:round; stroke-linejoin:round; }}
    .node  {{ fill:#111; }}
    .muted {{ fill:#333; }}
    .title       {{ font-family: "SF Pro Text",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; font-weight:700; font-size:16px; fill:#111; }}
    .lenLabel    {{ font-family: Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; font-weight:500; font-size:11px; fill:#111; font-variant-numeric: tabular-nums; }}
    .joinLabel   {{ font-family: Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; font-style:italic; font-size:11px; fill:#111; }}
    .cornerLabel {{ font-family: Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; font-style:italic; font-size:11px; fill:#111; }}
    .endLabel    {{ font-family: Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; font-style:italic; font-size:11px; fill:#111; }}
    .midLabel    {{ font-family: Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; font-style:italic; font-size:11px; fill:#111; }}
    .dimLabel    {{ font-family: "SF Mono","Roboto Mono",Menlo,Consolas,monospace; font-weight:500; font-size:11px; fill:#111; letter-spacing:0.2px; }}
    .sumLabel    {{ font-family: Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; font-weight:600; font-size:12px; fill:#111; }}
  </style>'''
    def _svg_footer(): return "</svg>"
    def _line(x1,y1,x2,y2, cls="track"): return f'<line class="{cls}" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>'
    def _circle(x,y,r=3, cls="node"): return f'<circle class="{cls}" cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}"/>'
    def _square(x,y,s=6, cls="node"): return f'<rect class="{cls}" x="{x-s/2:.2f}" y="{y-s/2:.2f}" width="{s:.2f}" height="{s:.2f}"/>'
    def _triangle(x,y,s=7, cls="node"):
        htri = s * 0.866
        p1 = (x, y - 2*htri/3); p2 = (x - s/2, y + htri/3); p3 = (x + s/2, y + htri/3)
        return f'<polygon class="{cls}" points="{p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}" />'

    parts = [_svg_header(w_out, h_out)]

    # Title + summary
    parts.append(draw_text_with_backer(w_out/2, style["title_y"] + style.get("extra_top", 0), title_text_for_spec(spec, total_len, style.get("cover_strip_on")), "middle", "title", px=16, pad_x=8, pad_y=4))
    summary = f"Total track length: {plan['total_len']:.2f} m   •   Field cuts: {plan['total_cuts']}   •   Excess: {plan['total_excess']:.2f} m"
    parts.append(draw_text_with_backer(w_out/2, style["title_y"] + style.get("extra_top", 0) + 22, summary, "middle", "sumLabel", px=12, pad_x=8, pad_y=3))

    # Draw legs + outward labels
    join_counter = 1
    for i in range(len(pts_m)-1):
        (x1,y1), (x2,y2) = pts_px[i], pts_px[i+1]
        parts.append(_line(x1,y1,x2,y2,"track"))

        leg_len_m = seg_lens[i]
        segs = prefer_smallest_cut(pack_segments(leg_len_m, base_spec.stock, base_spec.max_run_m), base_spec.stock)

        dx, dy = x2 - x1, y2 - y1
        L = (dx*dx + dy*dy) ** 0.5 or 1.0
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux  # normal

        # Leg midpoint + outward
        midx_leg = (x1 + x2) / 2; midy_leg = (y1 + y2) / 2
        onx_leg, ony_leg = outward_normal(nx, ny, midx_leg, midy_leg, cx, cy)

        is_vertical_leg = abs(dy) > abs(dx)

        # Joins (outward)
        if len(segs) > 1:
            cum = 0.0
            for s in segs[:-1]:
                cum += s.length_m
                bx = x1 + ux * (cum * S); by = y1 + uy * (cum * S)
                onx, ony = outward_normal(nx, ny, bx, by, cx, cy)
                if style.get("show_ticks", True):
                    half = style.get("tick_len",10)/2.0
                    parts.append(_line(bx + onx*half, by + ony*half, bx - onx*half, by - ony*half, "track"))
                parts.append(_circle(bx, by, r=style["node_size"]/2))
                join_label = f"Join {join_counter}"
                if style.get("show_element_labels", True):
                    join_rotate = side_label_angle if (rotate_side_labels and is_vertical_leg) else 0.0
                    parts.append(draw_text_with_backer(bx + onx*style["join_label_off"], by + ony*style["join_label_off"], join_label, "middle", "joinLabel", px=style["font_px"], rotate_deg=join_rotate))
                join_counter += 1

        # Segment labels (outward + rotate on vertical legs)
        cursor_m = 0.0
        seg_rotate = (-90.0 if is_vertical_leg else 0.0)
        for s in segs:
            mid_m = cursor_m + s.length_m/2
            mx = x1 + ux * (mid_m * S); my = y1 + uy * (mid_m * S)
            onx, ony = outward_normal(nx, ny, mx, my, cx, cy)
            donor_len = donor_length_for_segment(s)
            label_text = (f"{s.length_m:.2f}m" if getattr(s, "kind", "") != "cut"
                          else f"{s.length_m:.2f}m ({donor_len:.2f}\u2009→\u2009)")
            parts.append(
                draw_text_with_backer(
                    mx + onx*style["seg_label_off"],
                    my + ony*style["seg_label_off"],
                    label_text, "middle", "lenLabel", px=style["font_px"],
                    rotate_deg=seg_rotate
                )
            )
            cursor_m += s.length_m

        # Leg dimension — text only, outward (two lines)
        leg_label = f"Leg {i+1}\n{leg_len_m:.2f} m"
        dim_base_offset = style["dim_off"] + style["font_px"]*0.2
        dim_total_offset = dim_base_offset + (dim_side_extra if (rotate_side_labels and is_vertical_leg) else 0.0)
        parts.append(
            draw_text_with_backer(
                midx_leg + onx_leg*dim_total_offset,
                midy_leg + ony_leg*dim_base_offset,
                leg_label, "middle", "dimLabel", px=style["font_px"]
            )
        )

    # Ends (outward)
    sx, sy = pts_px[0]; ex, ey = pts_px[-1]
    parts.append(_triangle(sx, sy, s=style["node_size"]))
    parts.append(_triangle(ex, ey, s=style["node_size"]))
    if style.get("show_element_labels", True):
        nx0, ny0 = node_normal(pts_px, 0); nxN, nyN = node_normal(pts_px, len(pts_px)-1)
        on0x, on0y = outward_normal(nx0, ny0, sx, sy, cx, cy)
        onNx, onNy = outward_normal(nxN, nyN, ex, ey, cx, cy)
        start_vertical = rotate_side_labels and len(pts_px) >= 2 and abs(pts_px[1][0] - pts_px[0][0]) < abs(pts_px[1][1] - pts_px[0][1])
        end_vertical = rotate_side_labels and len(pts_px) >= 2 and abs(pts_px[-1][0] - pts_px[-2][0]) < abs(pts_px[-1][1] - pts_px[-2][1])
        start_rotate = side_label_angle if start_vertical else 0.0
        end_rotate = side_label_angle if end_vertical else 0.0
        parts.append(draw_text_with_backer(sx + on0x*style["end_label_off"], sy + on0y*style["end_label_off"], "End 1", "middle", "endLabel", px=style["font_px"], rotate_deg=start_rotate))
        parts.append(draw_text_with_backer(ex + onNx*style["end_label_off"], ey + onNy*style["end_label_off"], "End 2", "middle", "endLabel", px=style["font_px"], rotate_deg=end_rotate))

    # Corners (outward)
    internal_nodes_idx = list(range(1, max(0, len(pts_px)-1)))
    for idx, node_i in enumerate(internal_nodes_idx, start=1):
        px_i, py_i = pts_px[node_i]
        parts.append(_circle(px_i, py_i, r=style["node_size"]/2))
        if style.get("show_element_labels", True):
            nxc, nyc = node_normal(pts_px, node_i)
            onx, ony = outward_normal(nxc, nyc, px_i, py_i, cx, cy)
            horizontal_bias = abs(onx) > abs(ony)
            corner_rotate = side_label_angle if (rotate_side_labels and horizontal_bias) else 0.0
            parts.append(draw_text_with_backer(px_i + onx*style["corner_label_off"], py_i + ony*style["corner_label_off"], f"Corner {idx}", "middle", "cornerLabel", px=style["font_px"], rotate_deg=corner_rotate))

    # Mid components
    placed = place_mid_components(base_spec.mid_components, plan["pts"])
    for (xm, ym, mc) in placed:
        xpx, ypx = to_px((xm, ym))
        parts.append(_square(xpx, ypx, s=max(6, style["node_size"]-1)))
        parts.append(draw_text_with_backer(xpx + style["mid_label_off"] + style["font_px"]*0.2, ypx - (style["font_px"]//2), mc.part_no, "start", "midLabel", px=style["font_px"]))

    parts.append(_svg_footer())
    return "\n".join(parts), int(h_out), {}

# =========================================================
# Sidebar — Mid-run + Style
# =========================================================
with st.sidebar:
    st.subheader("Mid-run components")
    mid_str = st.text_area("Enter components (one per line as `pos_m:PARTNO`)", "1.0:FEED-TEE")

    st.header("Style")
    show_style = st.toggle("Show style options", value=False)
    if show_style:
        font_px = st.slider("Font size (px)", 9, 24, 12)
        track_stroke = st.slider("Track stroke (px)", 1, 8, 4)
        dim_stroke = st.slider("(unused) Dimension stroke (px)", 1, 4, 1)  # compatibility only
        node_size = st.slider("Node size (px)", 4, 16, 14)

        st.subheader("Label Offsets (perpendicular to track)")
        seg_label_off_px   = st.slider("Track LENGTH labels offset (px)", 2, 40, 18)
        join_label_off_px  = st.slider("JOIN labels offset (px)", 2, 40, 35)
        corner_label_off_px= st.slider("CORNER labels offset (px)", 30, 70, 50)
        end_label_off_px   = st.slider("END labels offset (px)", 30, 70, 50)
        mid_label_offset_px= st.slider("MID-COMPONENT label offset (px)", -60, 70, 20)
        dim_side_extra_px  = st.slider("Extra padding for left/right dimension labels (px)", 0, 80, 20)

        st.subheader("Other")
        dim_offset_px = st.slider("Dimension offset from line (px)", 6, 70, 55)
        title_offset_px = st.slider("Title offset from top (px)", -60, 150, 0)
        show_segment_ticks = st.checkbox("Show segment boundary ticks", True)
        tick_len_px = st.slider("Tick length (px)", 4, 24, 10)
        show_element_labels = st.checkbox("Show element labels (End/Corner/Join text)", True)

        st.divider()
        st.subheader("Canvas / Cropping")
        canvas_padding_px = st.slider("Canvas padding (px)", 100, 200, 140)
        extra_top_px      = st.slider("Extra top space (px)", 0, 200, 30)
        extra_bottom_px   = st.slider("Extra bottom space (px)", 0, 200, 30)
        auto_bottom_buffer= st.checkbox("Auto add bottom buffer for dims/labels", True)
        scroll_preview    = st.checkbox("Make preview scrollable", True)
    else:
        font_px = 12
        track_stroke = 4
        dim_stroke = 1
        node_size = 14
        seg_label_off_px = 18
        join_label_off_px = 35
        corner_label_off_px = 50
        end_label_off_px = 50
        mid_label_offset_px = 20
        dim_side_extra_px = 20
        dim_offset_px = 55
        title_offset_px = 0
        show_segment_ticks = True
        tick_len_px = 10
        show_element_labels = True
        canvas_padding_px = 140
        extra_top_px = 30
        extra_bottom_px = 30
        auto_bottom_buffer = True
        scroll_preview = True

# Parse mids
base_spec.mid_components = []
for line in [l.strip() for l in mid_str.splitlines() if l.strip()]:
    try:
        pos, pn = line.split(":",1)
        base_spec.mid_components.append(MidComponent(float(pos.strip()), pn.strip()))
    except: pass

# Style dict
style = dict(
    font_px=font_px, track_stroke=track_stroke, dim_stroke=dim_stroke, node_size=node_size,
    seg_label_off=seg_label_off_px, join_label_off=join_label_off_px, corner_label_off=corner_label_off_px,
    end_label_off=end_label_off_px, mid_label_off=mid_label_offset_px,
    dim_off=dim_offset_px, dim_side_extra=dim_side_extra_px, title_y=title_offset_px,
    show_ticks=show_segment_ticks, tick_len=tick_len_px, show_element_labels=show_element_labels,
    inline_join_types={}, pad=canvas_padding_px, extra_top=extra_top_px, extra_bottom=extra_bottom_px,
    auto_bottom_buffer=auto_bottom_buffer, scroll_preview=scroll_preview, cover_strip_on=cover_strip_on
)

# Inline joins for UI (positions)
def compute_inline_joins_for_ui(pts_m, seg_lens, stock, max_run):
    joins, counter = [], 1
    for i in range(len(pts_m)-1):
        leg_len = seg_lens[i]
        segs = prefer_smallest_cut(pack_segments(leg_len, stock, max_run), stock)
        cum = 0.0
        for s in segs[:-1]:
            cum += s.length_m
            joins.append({'label': f"Join {counter}", 'leg_i': i, 'pos_m': cum})
            counter += 1
    return joins

pts_ui = pts_for_spec(base_spec)
seg_lens_ui, _ = path_lengths(pts_ui)
inline_joins = compute_inline_joins_for_ui(pts_ui, seg_lens_ui, base_spec.stock, base_spec.max_run_m)

# Connections
with st.sidebar:
    st.header("Connections (per element)")
    base_key = f"{name}|{shape}|{length}|{(width if width is not None else '')}|{(depth if depth is not None else '')}|{track_profile}|{finish}"
    end_options = options_by_type.get("End", [])
    key_e1 = f"{base_key}:End 1"; key_e2 = f"{base_key}:End 2"
    default_end = end_options[0] if end_options else ""
    sel_end1 = st.selectbox("End 1", end_options or [""], index=_safe_index(end_options or [""], default_end), key=key_e1)
    sel_end2 = st.selectbox("End 2", end_options or [""], index=_safe_index(end_options or [""], default_end), key=key_e2)

    corner_options = options_by_type.get("Corner") or options_by_type.get("Join") or []
    internal_nodes = list(range(1, max(0, len(pts_ui)-1)))
    corner_labels = [f"Corner {i}" for i,_ in enumerate(internal_nodes, start=1)]
    corner_selections = []
    for label in corner_labels:
        corner_selections.append(st.selectbox(label, corner_options or [""], index=_safe_index(corner_options or [""], corner_options[0] if corner_options else "")))

    st.subheader("Inline joins (stock boundaries)")
    join_options = options_by_type.get("Join", [])
    join_type_by_label = {}
    if inline_joins:
        for j in inline_joins:
            lbl = j['label']
            join_type_by_label[lbl] = st.selectbox(lbl, join_options or [""], index=0, key=f"{base_key}:{lbl}")
    else:
        st.caption("No inline joins for current geometry/stock.")
    style["inline_join_types"] = join_type_by_label

# Update spec selections
base_spec = LayoutSpec(
    name=name, shape=shape, length_m=length, width_m=width, depth_m=depth,
    stock=stock_selected or [2.0, 1.0],
    max_run_m=float(maxrun) if str(maxrun).strip() else None,
    start_end=sel_end1, end_end=sel_end2,
    corner1_join=corner_selections[0] if len(corner_selections) >= 1 else "",
    corner2_join=corner_selections[1] if len(corner_selections) >= 2 else "",
    corner3_join=corner_selections[2] if len(corner_selections) >= 3 else "",
    mid_components=base_spec.mid_components
)

# Recompute plan & render
plan = compute_plan(base_spec)

svg, svg_h, _ = render_track_svg(base_spec, plan, style)
components.html(svg, height=svg_h, scrolling=style.get("scroll_preview", True))

st.download_button(
    "Download diagram as SVG",
    data=svg.encode("utf-8"),
    file_name=f"{base_spec.name}_diagram.svg",
    mime="image/svg+xml"
)

# Minimum-length rule messages
if plan.get("rules"):
    for r in plan["rules"]:
        if r.get("level") == "error":
            st.error(r.get("msg", ""))
        elif r.get("level") == "warn":
            st.warning(r.get("msg", ""))
        else:
            st.info(r.get("msg", ""))

# =========================================================
# Build BOM
# =========================================================
profile_col = PROFILE_TO_COL[track_profile]

track_name_to_len = {}
for nm in options_by_type.get("Track", []):
    m = re.search(r'(\d+(?:\.\d+)?)\s*m\b', nm.lower())
    if m:
        track_name_to_len[nm] = float(m.group(1))

def choose_track_name_for_donor(donor_len):
    if not track_name_to_len:
        return None
    exact = [nm for nm, L in track_name_to_len.items() if abs(L - donor_len) < 1e-6]
    if exact: return exact[0]
    bigger = sorted([(L, nm) for nm, L in track_name_to_len.items() if L + 1e-9 >= donor_len])
    if bigger: return bigger[0][1]
    return max(track_name_to_len.items(), key=lambda kv: kv[1])[0]

rows = []  # [Reference label, Name, Part no, QTY]

def add_excel_rows(ref_label, display_name, expected_type=None):
    dn = str(display_name).strip()
    candidates = df_opts[df_opts["Name"].str.lower() == dn.lower()] if dn else pd.DataFrame()
    if expected_type and not candidates.empty:
        candidates = candidates[candidates["Type"].str.lower() == expected_type.lower()]
    if candidates is None or candidates.empty:
        rows.append([ref_label, dn if dn else "--", "--", "--"]); return
    parts = apply_finish_tokens(candidates.iloc[0].get(profile_col, ""), finish_token)
    if not parts:
        rows.append([ref_label, dn, "--", "--"]); return
    for p in parts:
        rows.append([ref_label, dn, p, 1])

# Inline joins
join_counter = 1
pts_bom = plan["pts"]
if base_spec.shape == "U":
    pts_bom = [(x, -y) for (x, y) in pts_bom]
seg_lens_bom, _ = path_lengths(pts_bom)
for i in range(len(pts_bom)-1):
    leg_len_m = seg_lens_bom[i]
    segs = prefer_smallest_cut(pack_segments(leg_len_m, base_spec.stock, base_spec.max_run_m), base_spec.stock)
    if len(segs) > 1:
        for _ in segs[:-1]:
            ref = f"Join {join_counter}"
            chosen = style["inline_join_types"].get(ref, "")
            add_excel_rows(ref, chosen, expected_type="Join")
            join_counter += 1

# Ends
add_excel_rows("End 1", base_spec.start_end, expected_type="End")
add_excel_rows("End 2", base_spec.end_end, expected_type="End")

# Corners
corner_choices = [base_spec.corner1_join, base_spec.corner2_join, base_spec.corner3_join]
internal_nodes_idx = list(range(1, max(0, len(pts_bom)-1)))
for idx, _node_i in enumerate(internal_nodes_idx, start=1):
    display = corner_choices[idx-1] if idx-1 < len(corner_choices) else ""
    add_excel_rows(f"Corner {idx}", display, expected_type="Corner")

# Tracks — one row per donor piece
for i in range(len(pts_bom)-1):
    leg_len_m = seg_lens_bom[i]
    segs = prefer_smallest_cut(pack_segments(leg_len_m, base_spec.stock, base_spec.max_run_m), base_spec.stock)
    seg_idx = 1
    for s in segs:
        donor_len = donor_length_for_segment(s)
        track_name = choose_track_name_for_donor(donor_len)
        ref = f"Leg {i+1} Seg {seg_idx}"
        if track_name:
            add_excel_rows(ref, track_name, expected_type="Track")
        else:
            rows.append([ref, f"{donor_len:.2f} m", "--", "--"])
        seg_idx += 1

# Mid components
for idx, mc in enumerate(base_spec.mid_components, start=1):
    rows.append([f"MID {idx}", mc.part_no, mc.part_no, 1])

# Cover strip row
if style.get("cover_strip_on"):
    rows.append(["COVER STRIP", cover_name_ui, cover_part_ui if cover_part_ui.strip() else "--", round(plan["total_len"], 2)])

# Mounting hardware row
if use_mount and mh_name.strip() and mh_spacing and mh_spacing > 0:
    from math import ceil
    qty = int(ceil(plan["total_len"] / float(mh_spacing)))
    rows.append(["MOUNTING", mh_name.strip(), mh_part.strip() if mh_part.strip() else "--", qty])

# =========================================================
# BOM table + CSV
# =========================================================
st.markdown('<div id="bom"></div>', unsafe_allow_html=True)
st.subheader("Bill of Materials")
if rows:
    df_bom = pd.DataFrame(rows, columns=["Reference label","Name","Part no","QTY"])
    st.dataframe(df_bom, use_container_width=True)
    csv_bytes = df_bom.to_csv(index=False).encode("utf-8")
    st.download_button("Download BOM CSV", data=csv_bytes, file_name=f"{name}_BOM.csv", mime="text/csv")
else:
    st.write("Nothing yet for this configuration.")
