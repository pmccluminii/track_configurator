# app.py — v2.5.3
# - Text-only dimensions with outward normals
# - Leg dimension labels are two lines ("Leg N" and "<len> m")
# - Segment length labels rotate 90° on vertical legs (left/right)
# - Keeps: cover strip, mounting hardware per meter, U legs, Excel stock checkboxes, shortest-donor cuts, label backers

import copy
import csv, io, json, math, os, re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from shared_logic import (
    LayoutSpec, MidComponent, pack_segments,
    path_points_for_shape, path_lengths, place_mid_components,
    MIN_SEGMENT_HARD_M, MIN_SEGMENT_WARN_M
)

def default_config():
    return {
        "layout_name": "Layout 01",
        "track_profile": "Surface",
        "finish": "Black",
        "measurement_system": "Metric",
        "imperial_input_mode": "decimal",
        "shape": "Straight",
        "length": 5.0,
        "width": 2.0,
        "u_leg1": 1.50,
        "u_base": 2.00,
        "u_leg3": 1.50,
        "stock_selected": [],
        "max_run_text": "",
        "cover_strip_on": False,
        "cover_choice": "Without cover strip",
        "cover_name": "Cover Strip (linear)",
        "cover_part": "",
        "use_mount": True,
        "mount_choice": "With mounting hardware",
        "mh_name": "",
        "mh_part": "",
        "mh_spacing": 1.0,
        "mh_qty_each": 1,
        "layout_mid_components": "",
        "show_style_options": False,
        "font_px": 12,
        "track_stroke": 4,
        "dim_stroke": 1,
        "node_size": 14,
        "seg_label_off": 18,
        "join_label_off": 35,
        "corner_label_off": 50,
        "end_label_off": 50,
        "mid_label_off": 20,
        "isolation_mark_len": 18,
        "dim_side_extra": 20,
        "dim_offset": 55,
        "title_offset": 0,
        "show_segment_ticks": True,
        "tick_len": 10,
        "show_element_labels": True,
        "canvas_padding": 140,
        "extra_top": 30,
        "extra_bottom": 30,
        "auto_bottom_buffer": True,
        "scroll_preview": True,
        "start_end": "",
        "end_end": "",
        "corner1": "",
        "corner2": "",
        "corner3": "",
        "inline_join_types": {},
    }

if "base_config" not in st.session_state:
    st.session_state["base_config"] = default_config()
if "config" not in st.session_state:
    st.session_state["config"] = copy.deepcopy(st.session_state["base_config"])

if "_loaded_config_source" not in st.session_state:
    st.session_state["_loaded_config_source"] = None

config = st.session_state["config"]

def cfg_get(key, default):
    if key not in config:
        config[key] = default
    return config[key]

def cfg_set(key, value):
    config[key] = value

def _sync_single(cfg, key, widget_key):
    if key in cfg:
        st.session_state[widget_key] = cfg[key]

def sync_session_state_from_config(cfg):
    _sync_single(cfg, "layout_name", "cfg_layout_name")
    _sync_single(cfg, "track_profile", "cfg_track_profile")
    _sync_single(cfg, "finish", "cfg_finish")
    _sync_single(cfg, "measurement_system", "cfg_measurement_system")
    _sync_single(cfg, "imperial_input_mode", "cfg_imperial_input_mode")
    _sync_single(cfg, "shape", "cfg_shape")
    _sync_single(cfg, "u_leg1", "cfg_u_leg1")
    _sync_single(cfg, "u_base", "cfg_u_base")
    _sync_single(cfg, "u_leg3", "cfg_u_leg3")
    _sync_single(cfg, "length", "cfg_length")
    _sync_single(cfg, "width", "cfg_width")
    _sync_single(cfg, "stock_selected", "cfg_stock_selected")
    _sync_single(cfg, "max_run_text", "cfg_max_run")
    _sync_single(cfg, "cover_choice", "cfg_cover_choice")
    _sync_single(cfg, "mount_choice", "cfg_mount_choice")
    _sync_single(cfg, "layout_mid_components", "cfg_mid_components")
    _sync_single(cfg, "show_style_options", "cfg_show_style")
    _sync_single(cfg, "font_px", "cfg_font_px")
    _sync_single(cfg, "track_stroke", "cfg_track_stroke")
    _sync_single(cfg, "dim_stroke", "cfg_dim_stroke")
    _sync_single(cfg, "node_size", "cfg_node_size")
    _sync_single(cfg, "seg_label_off", "cfg_seg_label_off")
    _sync_single(cfg, "join_label_off", "cfg_join_label_off")
    _sync_single(cfg, "corner_label_off", "cfg_corner_label_off")
    _sync_single(cfg, "end_label_off", "cfg_end_label_off")
    _sync_single(cfg, "mid_label_off", "cfg_mid_label_off")
    _sync_single(cfg, "isolation_mark_len", "cfg_isolation_mark_len")
    _sync_single(cfg, "dim_side_extra", "cfg_dim_side_extra")
    _sync_single(cfg, "dim_offset", "cfg_dim_offset")
    _sync_single(cfg, "title_offset", "cfg_title_offset")
    _sync_single(cfg, "show_segment_ticks", "cfg_show_segment_ticks")
    _sync_single(cfg, "tick_len", "cfg_tick_len")
    _sync_single(cfg, "show_element_labels", "cfg_show_element_labels")
    _sync_single(cfg, "canvas_padding", "cfg_canvas_padding")
    _sync_single(cfg, "extra_top", "cfg_extra_top")
    _sync_single(cfg, "extra_bottom", "cfg_extra_bottom")
    _sync_single(cfg, "auto_bottom_buffer", "cfg_auto_bottom_buffer")
    _sync_single(cfg, "scroll_preview", "cfg_scroll_preview")
    _sync_single(cfg, "start_end", "cfg_end1")
    _sync_single(cfg, "end_end", "cfg_end2")
    _sync_single(cfg, "corner1", "cfg_Corner 1")
    _sync_single(cfg, "corner2", "cfg_Corner 2")
    _sync_single(cfg, "corner3", "cfg_Corner 3")
    # Clear dynamic inline join keys so defaults from config apply on next render
    for key in list(st.session_state.keys()):
        if key.startswith("cfg_inline_"):
            st.session_state.pop(key, None)
st.set_page_config(page_title="Track Layout Maker (Streamlit)", layout="wide", initial_sidebar_state="expanded")
measurement_system_saved = cfg_get("measurement_system", "Metric")
measurement_system = st.session_state.get("cfg_measurement_system", measurement_system_saved)
st.title(f"Track Layout Maker — ({measurement_system}) v2.5.3")

st.markdown(
    """
    <style>
    :root {
        color-scheme: only light;
    }
    body, .stApp, .stAppViewContainer, .main, .block-container {
        background-color: #ffffff !important;
        color: #111111 !important;
    }
    .stSidebar, .stSidebar > div, .stChatFloatingInputContainer {
        background-color: #f7f7f7 !important;
        color: #111111 !important;
    }
    .stMarkdown, .stTextInput > label, .stSelectbox > label, .stNumberInput > label,
    .stCheckbox > label, .stToggle > label, .stFileUploader > label,
    .stButton button, .stDownloadButton button {
        color: #111111 !important;
    }
    .stButton button, .stDownloadButton button {
        background-color: #f0f0f0 !important;
        color: #111111 !important;
        border: 1px solid #cccccc !important;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background-color: #e5e5e5 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #111111 !important;
    }
    .stNumberInput input, .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #111111 !important;
    }
    .stCheckbox input, .stToggle input {
        accent-color: #111111 !important;
    }
    .stAlert {
        border-radius: 6px !important;
        border-left: 6px solid transparent !important;
        box-shadow: none !important;
    }
    .stAlert-success {
        background-color: #e6f4ea !important;
        border-left-color: #0b8043 !important;
        color: #0b8043 !important;
    }
    .stAlert-warning {
        background-color: #fff3e0 !important;
        border-left-color: #f57c00 !important;
        color: #e65100 !important;
    }
    .stAlert-error {
        background-color: #fdecea !important;
        border-left-color: #d93025 !important;
        color: #c5221f !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
        5. When the layout looks right, use **Download diagram as SVG** to save the drawing or grab the BOM CSV further down the page.

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

with st.sidebar:
    st.header("Configuration")
    cfg_buffer = io.StringIO()
    writer = csv.writer(cfg_buffer)
    for key in sorted(config.keys()):
        writer.writerow([key, json.dumps(config[key])])
    st.download_button(
        "Download configuration CSV",
        data=cfg_buffer.getvalue(),
        file_name=f"{cfg_get('layout_name', 'layout').replace(' ', '_')}_config.csv",
        mime="text/csv"
    )
    if "_uploaded_config_name" not in st.session_state:
        st.session_state["_uploaded_config_name"] = ""
    if "_config_loaded_id" not in st.session_state:
        st.session_state["_config_loaded_id"] = None
    uploaded_cfg = st.file_uploader("Upload configuration CSV", type="csv", key="config_file_uploader")
    if st.session_state.get("_uploaded_config_name"):
        st.caption(f"Loaded config: {st.session_state['_uploaded_config_name']}")
    if uploaded_cfg is not None:
        token = (getattr(uploaded_cfg, "id", None), uploaded_cfg.name, uploaded_cfg.size)
        if st.session_state.get("_config_loaded_id") != token:
            try:
                text = uploaded_cfg.getvalue().decode("utf-8")
                reader = csv.reader(io.StringIO(text))
                new_cfg = copy.deepcopy(st.session_state["base_config"])
                for row in reader:
                    if len(row) < 2:
                        continue
                    key, raw = row[0], row[1]
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        value = raw
                    new_cfg[key] = value
                st.success("Configuration loaded.")
                st.session_state["config"] = new_cfg
                config = new_cfg
                sync_session_state_from_config(config)
                st.session_state["_uploaded_config_name"] = uploaded_cfg.name
                st.session_state["_config_loaded_id"] = token
                st.session_state.pop("config_file_uploader", None)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load configuration: {e}")
                st.session_state["_uploaded_config_name"] = ""
                st.session_state["_config_loaded_id"] = None
                st.session_state.pop("config_file_uploader", None)
                st.rerun()
    else:
        st.session_state["_config_loaded_id"] = None
        st.session_state.pop("config_file_uploader", None)
    if st.button("Reset configuration"):
        st.session_state["confirm_reset"] = True

    if st.session_state.get("confirm_reset"):
        st.warning("Reset will revert all inputs to defaults.")
        col_reset1, col_reset2 = st.columns(2)
        with col_reset1:
            if st.button("Yes, reset", key="cfg_reset_confirm"):
                st.session_state["config"] = copy.deepcopy(st.session_state["base_config"])
                config = st.session_state["config"]
                sync_session_state_from_config(config)
                st.session_state["_uploaded_config_name"] = ""
                st.session_state["_config_loaded_id"] = None
                st.session_state.pop("config_file_uploader", None)
                st.session_state["confirm_reset"] = False
                st.rerun()
        with col_reset2:
            if st.button("Cancel", key="cfg_reset_cancel"):
                st.session_state["confirm_reset"] = False
                st.rerun()

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
    st.error(f"Options file not found: {EXCEL_FILE}. Using minimal fallbacks.")
    df_opts = pd.DataFrame(columns=["Name","Type","BOM SURFACE","BOM RECESSED","BOM RECESSED TRIMLESS","BOM SUSPENDED"])
    df_mount = None

df_opts = df_opts.fillna("")
if df_mount is not None:
    df_mount = df_mount.fillna("")
for col in df_opts.columns:
    if df_opts[col].dtype == object:
        df_opts[col] = df_opts[col].astype(str).str.strip()
if df_mount is not None:
    for col in df_mount.columns:
        if df_mount[col].dtype == object:
            df_mount[col] = df_mount[col].astype(str).str.strip()

def _ensure_region_column(df):
    if df is None:
        return None
    df = df.copy()
    region_col = next((c for c in df.columns if str(c).strip().lower() == "region"), None)
    if region_col is None:
        df["Region"] = ""
    else:
        df["Region"] = df[region_col].astype(str).str.strip()
    return df

df_opts = _ensure_region_column(df_opts)
df_mount = _ensure_region_column(df_mount)

MEASUREMENT_CHOICES = ["Metric", "Imperial"]
METERS_PER_FOOT = 0.3048
INCHES_PER_METER = 39.37007874015748
INCH_DISPLAY_RESOLUTION = 16  # nearest 1/16"
COVER_STRIP_STICK_M = 3.0

def feet_to_meters(feet_val):
    try:
        return float(feet_val) * METERS_PER_FOOT
    except (TypeError, ValueError):
        return None

def meters_to_feet(meters_val):
    try:
        return float(meters_val) / METERS_PER_FOOT
    except (TypeError, ValueError):
        return None

def meters_to_inches(meters_val):
    try:
        return float(meters_val) * INCHES_PER_METER
    except (TypeError, ValueError):
        return None

def _round_inches(value, resolution=INCH_DISPLAY_RESOLUTION):
    if resolution <= 0:
        return value
    return round(float(value) * resolution) / resolution

def _split_region_tokens(value):
    if value is None:
        return []
    raw = str(value).replace("/", ",")
    return [token.strip().lower() for token in raw.split(",") if token.strip()]

_GLOBAL_REGION_TOKENS = {"", "global", "all", "both", "any"}
_REGION_ALIASES = {
    "metric": {"metric", "m", "metre", "meter", "emea", "eu"},
    "imperial": {"imperial", "ft", "feet", "foot", "us", "usa", "na", "north america"}
}

def region_allows(value, measurement):
    tokens = _split_region_tokens(value)
    if not tokens:
        return True
    measurement_key = str(measurement).strip().lower()
    aliases = _REGION_ALIASES.get(measurement_key, {measurement_key})
    for token in tokens:
        if token in _GLOBAL_REGION_TOKENS:
            return True
        if token in aliases:
            return True
        if measurement_key in token:
            return True
    return False

def filter_by_measurement(df, measurement):
    if df is None:
        return None
    if "Region" not in df.columns:
        return df.copy()
    mask = df["Region"].apply(lambda val: region_allows(val, measurement))
    return df[mask].copy()

def options_for_type(df, t):
    if df is None or df.empty:
        return []
    mask = df["Type"].str.lower() == str(t).lower()
    return sorted(df.loc[mask, "Name"].unique().tolist())

def build_option_lookup(df):
    lookup = {}
    if df is None or df.empty:
        return lookup
    def _col_name(target):
        target_low = str(target).strip().lower()
        for col in df.columns:
            if str(col).strip().lower() == target_low:
                return col
        return None
    name_col = _col_name("Name")
    type_col = _col_name("Type")
    circuit_col = _col_name("Circuit")
    for _, row in df.iterrows():
        name_val = str(row[name_col]).strip() if name_col else ""
        if not name_val:
            continue
        type_val = str(row[type_col]).strip().lower() if type_col else ""
        circuit_val = str(row[circuit_col]).strip() if circuit_col else ""
        key = (name_val.lower(), type_val if type_val else None)
        lookup[key] = {
            "name": name_val,
            "type": type_val,
            "circuit": circuit_val
        }
        fallback_key = (name_val.lower(), None)
        if fallback_key not in lookup:
            lookup[fallback_key] = lookup[key]
    return lookup

def parse_length_to_meters(name):
    text = str(name).strip()
    if not text:
        return None
    text_low = text.lower()
    match = re.search(r"(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*\"", text_low)
    if match:
        feet = int(match.group(1))
        inches = float(match.group(2))
        total_inches = feet * 12.0 + inches
        return total_inches * 0.0254
    match = re.search(r"(\d+(?:\.\d+)?)\s*'", text_low)
    if match:
        feet_val = float(match.group(1))
        return feet_val * METERS_PER_FOOT
    match = re.search(r"(\d+(?:\.\d+)?)\s*(ft|feet|foot)\b", text_low)
    if match:
        feet_val = float(match.group(1))
        return feet_val * METERS_PER_FOOT
    match = re.search(r"(\d+(?:\.\d+)?)\s*(m|meter|metre)\b", text_low)
    if match:
        return float(match.group(1))
    return None

def meters_to_feet_inches_parts(meters_val, precision=4):
    total_inches = meters_to_inches(meters_val)
    if total_inches is None:
        return 0, 0.0
    total_inches = max(0.0, _round_inches(total_inches))
    feet = int(math.floor(total_inches / 12.0))
    inches = total_inches - feet * 12.0
    inches = round(inches, precision)
    if inches >= 12.0 - (1.0 / max(1, INCH_DISPLAY_RESOLUTION)):
        feet += 1
        inches = 0.0
    if abs(inches) < (10 ** (-precision)):
        inches = 0.0
    return feet, inches

def meters_to_feet_inches_string(meters_val, precision=2, include_zero_feet=True):
    if meters_val is None:
        return ""
    feet, inches = meters_to_feet_inches_parts(meters_val, precision=precision+2)
    rounding = round(inches, precision)
    if rounding >= 12.0:
        feet += 1
        rounding = 0.0
    if precision == 0:
        inch_str = f"{int(rounding)}"
    else:
        inch_str = f"{rounding:.{precision}f}".rstrip("0").rstrip(".")
    if inch_str == "":
        inch_str = "0"
    if rounding == 0.0:
        if include_zero_feet or feet != 0:
            return f"{feet}'"
        return "0\""
    if include_zero_feet or feet != 0:
        return f"{feet}'{inch_str}\""
    return f'{inch_str}"'

def format_length(meters_val, measurement, decimals=2, imperial_precision=2, include_unit=True):
    if meters_val is None:
        return ""
    if str(measurement).strip().lower().startswith("imperial"):
        text = meters_to_feet_inches_string(meters_val, precision=imperial_precision, include_zero_feet=True)
        return text
    value = float(meters_val)
    if include_unit:
        return f"{value:.{decimals}f} m"
    return f"{value:.{decimals}f}"

def length_unit_suffix(measurement):
    return "ft" if str(measurement).strip().lower().startswith("imperial") else "m"

def meters_to_display_value(meters_val, measurement):
    if meters_val is None:
        return None
    if str(measurement).strip().lower().startswith("imperial"):
        return meters_to_feet(meters_val)
    return meters_val

def display_value_to_meters(display_val, measurement):
    if display_val is None:
        return None
    if str(measurement).strip().lower().startswith("imperial"):
        return feet_to_meters(display_val)
    try:
        return float(display_val)
    except (TypeError, ValueError):
        return None

def parse_length_string(text, measurement):
    raw = str(text).strip()
    if not raw:
        return None
    measurement_key = str(measurement).strip().lower()
    if measurement_key.startswith("imperial"):
        if any(ch in raw for ch in ("'", '"')) or re.search(r"\b(ft|in)\b", raw.lower()):
            parsed = parse_length_to_meters(raw)
            if parsed is not None:
                return parsed
        try:
            val = float(raw)
            return feet_to_meters(val)
        except (TypeError, ValueError):
            pass
        parts = re.split(r"[^0-9.]+", raw)
        parts = [p for p in parts if p]
        if len(parts) == 2:
            try:
                feet_val = float(parts[0])
                inch_val = float(parts[1])
                return feet_to_meters(feet_val + inch_val / 12.0)
            except (TypeError, ValueError):
                return None
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

def is_feed_label(text):
    if text is None:
        return False
    stripped = str(text).strip().lower()
    if not stripped:
        return False
    feed_tokens = ["feed", "feeder", "power in", "supply"]
    return any(token in stripped for token in feed_tokens)

def is_isolator_label(text):
    if text is None:
        return False
    stripped = str(text).strip().lower()
    if not stripped:
        return False
    isolator_tokens = ["isolator", "iso", "isolate", "isolated"]
    return any(token in stripped for token in isolator_tokens)

def option_lookup_entry(meta_lookup, name, expected_type=None):
    if not meta_lookup or name is None:
        return None
    name_key = str(name).strip().lower()
    if not name_key:
        return None
    type_key = str(expected_type).strip().lower() if expected_type else None
    entry = meta_lookup.get((name_key, type_key))
    if entry is None:
        entry = meta_lookup.get((name_key, None))
    return entry

def option_has_isolation(meta_lookup, name, expected_type=None):
    entry = option_lookup_entry(meta_lookup, name, expected_type=expected_type)
    if entry:
        circ = str(entry.get("circuit", "")).strip().lower()
        if "isolat" in circ:
            return True
    return is_isolator_label(name)

def _sync_imperial_widget_state(widget_key, value_m):
    if value_m is None:
        return
    ft_key = f"{widget_key}_ft"
    in_key = f"{widget_key}_in"
    feet_val, inch_val = meters_to_feet_inches_parts(value_m, precision=4)
    inch_val = min(max(_round_inches(inch_val), 0.0), 12.0 - (1.0 / INCH_DISPLAY_RESOLUTION))
    current_ft = st.session_state.get(ft_key)
    current_in = st.session_state.get(in_key)
    try:
        current_m = feet_to_meters(float(current_ft) + float(current_in)/12.0)
    except (TypeError, ValueError):
        current_m = None
    if current_m is None or abs(current_m - value_m) > 1e-6:
        st.session_state[ft_key] = int(feet_val)
        st.session_state[in_key] = round(inch_val, 4)

def length_number_input(label, config_key, widget_key, default_m, measurement, min_m=0.1, step=0.01, fmt="%.2f", imperial_mode="decimal"):
    measurement_key = str(measurement).strip().lower()
    if measurement_key.startswith("imperial") and imperial_mode == "feet_inches":
        feet_default, inches_default = meters_to_feet_inches_parts(default_m, precision=4)
        inches_default = min(max(_round_inches(inches_default), 0.0), 12.0 - (1.0 / INCH_DISPLAY_RESOLUTION))
        ft_key = f"{widget_key}_ft"
        in_key = f"{widget_key}_in"
        feet_val = st.number_input(
            f"{label} (ft)",
            min_value=0,
            value=int(feet_default),
            step=1,
            key=ft_key
        )
        inch_step = 1.0 / INCH_DISPLAY_RESOLUTION
        inch_max = 12.0 - inch_step
        inch_val = st.number_input(
            f"{label} (in)",
            min_value=0.0,
            max_value=float(inch_max),
            value=float(round(inches_default, 4)),
            step=float(inch_step),
            format="%.4f",
            key=in_key
        )
        inch_val = min(max(_round_inches(inch_val), 0.0), inch_max)
        value_m = feet_to_meters(float(feet_val) + inch_val/12.0)
        if value_m < min_m:
            value_m = min_m
            _sync_imperial_widget_state(widget_key, value_m)
        cfg_set(config_key, value_m)
        _sync_imperial_widget_state(widget_key, value_m)
        return value_m

    unit = length_unit_suffix(measurement)
    default_disp = meters_to_display_value(default_m, measurement)
    if default_disp is None:
        default_disp = meters_to_display_value(min_m, measurement)
    if default_disp is None:
        default_disp = 0.0
    min_disp = meters_to_display_value(min_m, measurement)
    if min_disp is None:
        min_disp = 0.0
    value_disp = st.number_input(
        f"{label} ({unit})",
        min_value=float(round(min_disp, 4)),
        value=float(round(default_disp, 4)),
        step=step,
        format=fmt,
        key=widget_key
    )
    value_m = display_value_to_meters(value_disp, measurement)
    if value_m is None:
        value_m = default_m
    if value_m < min_m:
        value_m = min_m
        st.session_state[widget_key] = meters_to_display_value(value_m, measurement)
    cfg_set(config_key, value_m)
    if measurement_key.startswith("imperial"):
        _sync_imperial_widget_state(widget_key, value_m)
    return value_m

options_by_type = {}

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

available_track_lengths = []
max_run_meters = None
option_meta_lookup = {}

# =========================================================
# Sidebar — System
# =========================================================
with st.sidebar:
    st.header("System")
    measurement_default = cfg_get("measurement_system", MEASUREMENT_CHOICES[0])
    measurement_choice = st.selectbox(
        "Measurement system",
        MEASUREMENT_CHOICES,
        index=_safe_index(MEASUREMENT_CHOICES, measurement_default),
        key="cfg_measurement_system"
    )
    cfg_set("measurement_system", measurement_choice)
    measurement_system = measurement_choice
    imperial_input_mode = cfg_get("imperial_input_mode", "decimal")
    if measurement_system == "Imperial":
        mode_labels = ["Decimal feet", "Feet + inches"]
        label_to_mode = {"Decimal feet": "decimal", "Feet + inches": "feet_inches"}
        current_label = next((lbl for lbl, mode in label_to_mode.items() if mode == imperial_input_mode), mode_labels[0])
        imperial_label = st.selectbox(
            "Imperial input style",
            mode_labels,
            index=_safe_index(mode_labels, current_label),
            key="cfg_imperial_input_mode"
        )
        imperial_input_mode = label_to_mode.get(imperial_label, "decimal")
    else:
        imperial_input_mode = "decimal"
    cfg_set("imperial_input_mode", imperial_input_mode)

    df_opts_filtered = filter_by_measurement(df_opts, measurement_system)
    df_mount_filtered = filter_by_measurement(df_mount, measurement_system)
    options_by_type = {
        "Track":  options_for_type(df_opts_filtered, "Track"),
        "End":    options_for_type(df_opts_filtered, "End"),
        "Join":   options_for_type(df_opts_filtered, "Join"),
        "Corner": options_for_type(df_opts_filtered, "Corner"),
        "Cover Strip": options_for_type(df_opts_filtered, "Cover Strip"),
    }
    option_meta_lookup = build_option_lookup(df_opts_filtered)
    track_name_to_len = {}
    if df_opts_filtered is not None and not df_opts_filtered.empty:
        track_mask = df_opts_filtered["Type"].str.lower() == "track"
        track_rows = df_opts_filtered.loc[track_mask, ["Name"]]
        for _, row in track_rows.iterrows():
            nm = row["Name"]
            length_m = parse_length_to_meters(nm)
            if length_m is not None:
                track_name_to_len[nm] = round(length_m, 6)
    available_track_lengths = sorted(set(track_name_to_len.values()))

    profile_options = list(PROFILE_TO_COL.keys())
    track_profile_default = cfg_get("track_profile", profile_options[0] if profile_options else "")
    track_profile = st.selectbox(
        "Track profile",
        profile_options,
        index=_safe_index(profile_options, track_profile_default),
        key="cfg_track_profile"
    )
    cfg_set("track_profile", track_profile)

    finish_options = ["Black", "White"]
    finish_default = cfg_get("finish", finish_options[0])
    finish = st.selectbox(
        "Finish",
        finish_options,
        index=_safe_index(finish_options, finish_default),
        key="cfg_finish"
    )
    cfg_set("finish", finish)
    finish_token = "BK" if finish.lower().startswith("b") else "WH"

# =========================================================
# Sidebar — Geometry & Stock
# =========================================================
with st.sidebar:
    st.header("Inputs")
    name = st.text_input("Layout Name", cfg_get("layout_name", "Layout 01"), key="cfg_layout_name")
    cfg_set("layout_name", name)

    shape_options = ["Straight", "L", "Rectangle", "U"]
    shape = cfg_get("shape", shape_options[0])
    shape = st.selectbox("Shape", shape_options, index=_safe_index(shape_options, shape), key="cfg_shape")
    cfg_set("shape", shape)

    if shape == "U":
        leg1 = length_number_input("U — Leg 1", "u_leg1", "cfg_u_leg1", float(config.get("u_leg1", 1.50)), measurement_system, imperial_mode=imperial_input_mode)
        base = length_number_input("U — Base", "u_base", "cfg_u_base", float(config.get("u_base", 2.00)), measurement_system, imperial_mode=imperial_input_mode)
        leg3 = length_number_input("U — Leg 3", "u_leg3", "cfg_u_leg3", float(config.get("u_leg3", 1.50)), measurement_system, imperial_mode=imperial_input_mode)
        length = base
        width = None
        depth = (leg1, base, leg3)
        cfg_set("length", length)
    else:
        length = length_number_input("Length", "length", "cfg_length", float(config.get("length", 5.0)), measurement_system, imperial_mode=imperial_input_mode)
        width = None
        depth = None
        if shape in ("L", "Rectangle"):
            width_val = length_number_input("Width", "width", "cfg_width", float(config.get("width", 2.0)), measurement_system, imperial_mode=imperial_input_mode)
            width = width_val
        else:
            width = width

    st.subheader("Stock lengths (from Excel)")
    if available_track_lengths:
        default_stock_config = [float(x) for x in config.get("stock_selected", [])]
        default_stock = []
        for val in default_stock_config:
            match = next((opt for opt in available_track_lengths if abs(opt - val) < 1e-6), None)
            if match is not None:
                default_stock.append(match)
        if not default_stock:
            default_stock = available_track_lengths
        stock_selected = st.multiselect(
            "Select lengths to use",
            available_track_lengths,
            default=default_stock,
            key="cfg_stock_selected",
            format_func=lambda v: format_length(v, measurement_system, decimals=2, imperial_precision=2)
        )
    else:
        fallback_lengths = [2.0, 1.0]
        formatted_fallback = ", ".join(format_length(val, measurement_system, decimals=2, imperial_precision=2) for val in fallback_lengths)
        st.warning(f"No track lengths found in Excel ‘Track’ options. Using fallback {formatted_fallback}.")
        stock_selected = config.get("stock_selected", fallback_lengths) or fallback_lengths
    stock_selected = [float(x) for x in stock_selected]
    cfg_set("stock_selected", stock_selected)

    maxrun_label = f"Max run ({length_unit_suffix(measurement_system)}) [optional]"
    maxrun = st.text_input(maxrun_label, config.get("max_run_text", ""), key="cfg_max_run")
    cfg_set("max_run_text", maxrun)
    max_run_meters = None
    max_run_clean = str(maxrun).strip()
    if max_run_clean:
        parsed_max = parse_length_string(max_run_clean.replace(",", " "), measurement_system)
        if parsed_max is not None:
            max_run_meters = parsed_max

# Accessories
with st.sidebar:
    st.header("Accessories")
    cover_toggle_options = ["With cover strip", "Without cover strip"]
    cover_default_label = config.get("cover_choice", cover_toggle_options[1])
    cover_choice = st.selectbox(
        "Cover strip",
        cover_toggle_options,
        index=_safe_index(cover_toggle_options, cover_default_label, default_idx=1),
        key="cfg_cover_choice"
    )
    cover_catalog = options_by_type.get("Cover Strip", [])
    cover_name_selected = cover_catalog[0] if cover_catalog else ""
    include_cover_strip = (cover_choice == cover_toggle_options[0])
    if include_cover_strip and not cover_name_selected:
        st.warning("Cover strip option not found in Excel for the current configuration.")
        include_cover_strip = False
    cfg_set("cover_choice", cover_choice)
    cfg_set("cover_strip_on", include_cover_strip)
    if include_cover_strip:
        cfg_set("cover_name", cover_name_selected)
    else:
        cover_name_selected = ""
        cfg_set("cover_name", "")
    cfg_set("cover_part", "")

    st.subheader("Mounting hardware")
    mount_entry = None
    mount_df = df_mount_filtered if df_mount_filtered is not None else df_mount
    if mount_df is not None and not mount_df.empty:
        def _col(df_local, name):
            for c in df_local.columns:
                if c.strip().lower() == name.lower(): return c
            return None
        c_prof = _col(mount_df, "Profile"); c_name = _col(mount_df, "Name")
        c_pn = _col(mount_df, "PartNo"); c_sp = _col(mount_df, "Spacing_m")
        c_qty = _col(mount_df, "QTY")
        if all([c_prof, c_name, c_pn, c_sp, c_qty]):
            mh_candidates = mount_df[mount_df[c_prof].astype(str).str.strip().str.lower() == track_profile.strip().lower()]
            if not mh_candidates.empty:
                row0 = mh_candidates.iloc[0]
                name_val = str(row0[c_name]).strip()
                part_val = str(row0[c_pn]).strip()
                try:
                    spacing_val = float(row0[c_sp]) if str(row0[c_sp]).strip() else None
                except Exception:
                    spacing_val = None
                try:
                    qty_each = float(row0[c_qty]) if str(row0[c_qty]).strip() else None
                except Exception:
                    qty_each = None
                if name_val:
                    mount_entry = {
                        "name": name_val,
                        "part": part_val,
                        "spacing_m": spacing_val if spacing_val and spacing_val > 0 else None,
                        "qty_each": qty_each if qty_each and qty_each > 0 else 1.0
                    }

    mount_toggle_options = ["With mounting hardware", "Without mounting hardware"]
    mount_default_label = config.get(
        "mount_choice",
        mount_toggle_options[0] if mount_entry else mount_toggle_options[1]
    )
    default_mount_idx = _safe_index(
        mount_toggle_options,
        mount_default_label,
        default_idx=(0 if mount_entry else 1)
    )
    mount_choice = st.selectbox(
        f"Mounting hardware (per {length_unit_suffix(measurement_system)})",
        mount_toggle_options,
        index=default_mount_idx,
        key="cfg_mount_choice"
    )
    include_mounting = (mount_choice == mount_toggle_options[0]) and (mount_entry is not None)
    if mount_entry is None:
        st.caption("No mounting hardware listed for this profile.")
    cfg_set("mount_choice", mount_choice if mount_entry else mount_toggle_options[1])
    cfg_set("use_mount", include_mounting)
    if include_mounting:
        cfg_set("mh_name", mount_entry["name"])
        cfg_set("mh_part", mount_entry["part"])
        cfg_set("mh_spacing", mount_entry["spacing_m"])
        cfg_set("mh_qty_each", mount_entry["qty_each"])
        mh_name = mount_entry["name"]
        mh_part = mount_entry["part"]
        mh_spacing = mount_entry["spacing_m"]
        mh_qty_each = mount_entry["qty_each"]
    else:
        cfg_set("mh_name", "")
        cfg_set("mh_part", "")
        cfg_set("mh_spacing", None)
        cfg_set("mh_qty_each", None)
        mh_name = ""
        mh_part = ""
        mh_spacing = None
        mh_qty_each = None
    cover_part_ui = ""
    cover_name_ui = cover_name_selected
    cover_strip_on = include_cover_strip
    use_mount = include_mounting

# Base spec
base_spec = LayoutSpec(
    name=name, shape=shape, length_m=length, width_m=width, depth_m=depth,
    stock=stock_selected or [2.0, 1.0],
    max_run_m=max_run_meters,
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

def compute_plan(spec, measurement):
    """
    Compute geometry + basic cut stats and apply minimum-length rules:
      - Standalone run (single-leg Straight): must meet the configured hard/warn limits.
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
    fmt_len = lambda val: format_length(val, measurement)
    hard_text = fmt_len(leg_min_hard)
    warn_text = fmt_len(leg_min_warn)
    standalone_error_limit = 0.34
    standalone_warn_limit = 0.36
    standalone_error_text = fmt_len(standalone_error_limit)
    standalone_warn_text = fmt_len(standalone_warn_limit)

    # Cut stats and segment validation
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
                    "msg": f"Leg {i+1} segment {seg_idx} is {fmt_len(seg_length)}, below the minimum {hard_text}. Combine it with an adjacent segment or cut a longer piece."
                })
            elif seg_length < leg_min_warn - 1e-9 and not same_as_leg:
                rules.append({
                    "level": "warn",
                    "msg": f"Leg {i+1} segment {seg_idx} is {fmt_len(seg_length)}. It clears {hard_text}, but to fit a light allow ≥ {warn_text}."
                })

    # Minimum-length validations
    num_legs = max(0, len(pts) - 1)
    if spec.shape == "Straight" and num_legs == 1:
        run_len = seg_lens[0] if seg_lens else 0.0
        if run_len < 0.34 - 1e-9:
            rules.append({
                "level": "error",
                "msg": f"Standalone run is {fmt_len(run_len)}, below the minimum {standalone_error_text} needed for an end feed + 1 light."
            })
        elif run_len < 0.36 - 1e-9:
            rules.append({
                "level": "warn",
                "msg": f"Standalone run is {fmt_len(run_len)}. It meets {standalone_error_text} minimum, but to fit a light comfortably use ≥ {standalone_warn_text}."
            })
    else:
        # Multi-leg shapes
        for i in range(num_legs):
            Lm = seg_lens[i]
            if Lm < leg_min_hard - 1e-9:
                rules.append({
                    "level": "error",
                    "msg": f"Leg {i+1} is {fmt_len(Lm)}, below the hard minimum {hard_text}."
                })
            elif Lm < leg_min_warn - 1e-9:
                rules.append({
                    "level": "warn",
                    "msg": f"Leg {i+1} is {fmt_len(Lm)}. It clears {hard_text}, but is too short to fit a light. Use ≥ {warn_text} to allow a luminaire."
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
                    "msg": f"Mid component {idx} is placed at {fmt_len(pos_m)}, beyond the total layout length {fmt_len(total_len)}. It will be clamped to the end."
                })
            elif pos_m < -1e-6:
                rules.append({
                    "level": "warn",
                    "msg": f"Mid component {idx} is placed at {fmt_len(pos_m)}, before the start of the run. It will be clamped to the beginning."
                })

    return dict(
        pts=pts,
        seg_lens=seg_lens,
        total_len=total_len,
        total_cuts=total_cuts,
        total_excess=total_waste,
        rules=rules
    )

plan = compute_plan(base_spec, measurement_system)

# Quick banner for rule status
if plan.get("rules"):
    n_err = sum(1 for r in plan["rules"] if r.get("level") == "error")
    n_warn = sum(1 for r in plan["rules"] if r.get("level") == "warn")
    if n_err > 0:
        st.error(f"Validation issues: {n_err} error(s), {n_warn} warning(s) found.")
    elif n_warn > 0:
        st.warning(f"Validation warnings: {n_warn} item(s) to review.")
    else:
        st.success("Validation: no minimum-length issues.")
else:
    st.success("Validation: no minimum-length issues.")

def title_text_for_spec(spec, total_len, cover_on, measurement):
    parts = [spec.name, "—", spec.shape]
    fmt = lambda val: format_length(val, measurement)
    if spec.shape in ("L","Rectangle"):
        if spec.length_m is not None and spec.width_m is not None:
            parts.append(f"({fmt(spec.length_m)} × {fmt(spec.width_m)})")
        elif spec.length_m is not None:
            parts.append(f"({fmt(spec.length_m)})")
    elif spec.shape == "U":
        if isinstance(spec.depth_m, (list, tuple)) and len(spec.depth_m) == 3:
            L1,B,L3 = spec.depth_m
            parts.append(f"(L1 {fmt(float(L1))} × Base {fmt(float(B))} × L3 {fmt(float(L3))})")
        elif spec.length_m is not None and spec.depth_m is not None:
            parts.append(f"({fmt(spec.length_m)} × {fmt(float(spec.depth_m))})")
    else:
        if spec.length_m is not None:
            parts.append(f"({fmt(spec.length_m)})")
    parts.append(f"• Total {fmt(total_len)}")
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
    measurement = style.get("measurement_system", "Metric")

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

    iso_stroke_w = max(1.0, style["track_stroke"] * 0.6)

    def _svg_header(w, h):
        return f'''<svg viewBox="0 0 {w:.2f} {h:.2f}" width="100%" height="{h:.0f}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
  <style>
    svg {{ overflow: visible; }}
    .track {{ stroke:#111; stroke-width:{style["track_stroke"]}; fill:none; stroke-linecap:round; stroke-linejoin:round; }}
    .node  {{ fill:#111; }}
    .muted {{ fill:#333; }}
    .isoMark   {{ stroke:#111; stroke-width:{iso_stroke_w:.2f}; fill:none; stroke-linecap:round; }}
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
    parts.append(draw_text_with_backer(w_out/2, style["title_y"] + style.get("extra_top", 0), title_text_for_spec(spec, total_len, style.get("cover_strip_on"), measurement), "middle", "title", px=16, pad_x=8, pad_y=4))
    summary = (
        f"Total track length: {format_length(plan['total_len'], measurement)}   •   "
        f"Field cuts: {plan['total_cuts']}   •   Excess: {format_length(plan['total_excess'], measurement)}"
    )
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
                if style.get("inline_join_isolation", {}).get(join_label, False):
                    mark_half = max(0.0, style.get("isolation_mark_len", style.get("node_size", 14)) / 2.0)
                    parts.append(_line(bx - onx*mark_half, by - ony*mark_half, bx + onx*mark_half, by + ony*mark_half, "isoMark"))
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
            seg_text = format_length(s.length_m, measurement)
            if getattr(s, "kind", "") != "cut":
                label_text = seg_text
            else:
                donor_text = format_length(donor_len, measurement)
                label_text = f"{seg_text} ({donor_text} cut)"
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
        leg_label = f"Leg {i+1}\n{format_length(leg_len_m, measurement)}"
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
        nxc, nyc = node_normal(pts_px, node_i)
        onx, ony = outward_normal(nxc, nyc, px_i, py_i, cx, cy)
        corner_label = f"Corner {idx}"
        parts.append(_circle(px_i, py_i, r=style["node_size"]/2))
        if style.get("corner_isolation", {}).get(corner_label, False):
            mark_half = max(0.0, style.get("isolation_mark_len", style.get("node_size", 14)) / 2.0)
            parts.append(_line(px_i - onx*mark_half, py_i - ony*mark_half, px_i + onx*mark_half, py_i + ony*mark_half, "isoMark"))
        if style.get("show_element_labels", True):
            horizontal_bias = abs(onx) > abs(ony)
            corner_rotate = side_label_angle if (rotate_side_labels and horizontal_bias) else 0.0
            parts.append(draw_text_with_backer(px_i + onx*style["corner_label_off"], py_i + ony*style["corner_label_off"], corner_label, "middle", "cornerLabel", px=style["font_px"], rotate_deg=corner_rotate))

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
    mid_unit_suffix = length_unit_suffix(measurement_system)
    mid_str = st.text_area(
        f"Enter components (one per line as `pos_{mid_unit_suffix}:PARTNO`)",
        config.get("layout_mid_components", "1.0:FEED-TEE"),
        key="cfg_mid_components"
    )
    cfg_set("layout_mid_components", mid_str)

    st.header("Style")
    show_style = st.toggle("Show style options", value=bool(config.get("show_style_options", False)), key="cfg_show_style")
    cfg_set("show_style_options", show_style)

    font_px = int(config.get("font_px", 12))
    if show_style:
        font_px = st.slider("Font size (px)", 9, 24, font_px, key="cfg_font_px")
    cfg_set("font_px", font_px)

    track_stroke = int(config.get("track_stroke", 4))
    if show_style:
        track_stroke = st.slider("Track stroke (px)", 1, 8, track_stroke, key="cfg_track_stroke")
    cfg_set("track_stroke", track_stroke)

    dim_stroke = int(config.get("dim_stroke", 1))
    if show_style:
        dim_stroke = st.slider("(unused) Dimension stroke (px)", 1, 4, dim_stroke, key="cfg_dim_stroke")
    cfg_set("dim_stroke", dim_stroke)

    node_size = int(config.get("node_size", 14))
    if show_style:
        node_size = st.slider("Node size (px)", 4, 16, node_size, key="cfg_node_size")
    cfg_set("node_size", node_size)

    seg_label_off_px = int(config.get("seg_label_off", 18))
    join_label_off_px = int(config.get("join_label_off", 35))
    corner_label_off_px = int(config.get("corner_label_off", 50))
    end_label_off_px = int(config.get("end_label_off", 50))
    mid_label_offset_px = int(config.get("mid_label_off", 20))
    isolation_mark_len_px = int(config.get("isolation_mark_len", 18))
    dim_side_extra_px = int(config.get("dim_side_extra", 20))

    if show_style:
        st.subheader("Label Offsets (perpendicular to track)")
        seg_label_off_px = st.slider("Track LENGTH labels offset (px)", 2, 40, seg_label_off_px, key="cfg_seg_label_off")
        join_label_off_px = st.slider("JOIN labels offset (px)", 2, 40, join_label_off_px, key="cfg_join_label_off")
        corner_label_off_px = st.slider("CORNER labels offset (px)", 30, 70, corner_label_off_px, key="cfg_corner_label_off")
        end_label_off_px = st.slider("END labels offset (px)", 30, 70, end_label_off_px, key="cfg_end_label_off")
        mid_label_offset_px = st.slider("MID-COMPONENT label offset (px)", -60, 70, mid_label_offset_px, key="cfg_mid_label_off")
        dim_side_extra_px = st.slider("Extra padding for left/right dimension labels (px)", 0, 80, dim_side_extra_px, key="cfg_dim_side_extra")

    cfg_set("seg_label_off", seg_label_off_px)
    cfg_set("join_label_off", join_label_off_px)
    cfg_set("corner_label_off", corner_label_off_px)
    cfg_set("end_label_off", end_label_off_px)
    cfg_set("mid_label_off", mid_label_offset_px)
    cfg_set("dim_side_extra", dim_side_extra_px)

    dim_offset_px = int(config.get("dim_offset", 55))
    title_offset_px = int(config.get("title_offset", 0))
    show_segment_ticks = bool(config.get("show_segment_ticks", True))
    tick_len_px = int(config.get("tick_len", 10))
    show_element_labels = bool(config.get("show_element_labels", True))

    if show_style:
        st.subheader("Other")
        dim_offset_px = st.slider("Dimension offset from line (px)", 6, 70, dim_offset_px, key="cfg_dim_offset")
        title_offset_px = st.slider("Title offset from top (px)", -60, 150, title_offset_px, key="cfg_title_offset")
        show_segment_ticks = st.checkbox("Show segment boundary ticks", show_segment_ticks, key="cfg_show_segment_ticks")
        tick_len_px = st.slider("Tick length (px)", 4, 24, tick_len_px, key="cfg_tick_len")
        show_element_labels = st.checkbox("Show element labels (End/Corner/Join text)", show_element_labels, key="cfg_show_element_labels")
        isolation_mark_len_px = st.slider("Isolation mark length (px)", 6, 40, isolation_mark_len_px, key="cfg_isolation_mark_len")

    cfg_set("dim_offset", dim_offset_px)
    cfg_set("title_offset", title_offset_px)
    cfg_set("show_segment_ticks", show_segment_ticks)
    cfg_set("tick_len", tick_len_px)
    cfg_set("show_element_labels", show_element_labels)
    cfg_set("isolation_mark_len", isolation_mark_len_px)

    canvas_padding_px = int(config.get("canvas_padding", 140))
    extra_top_px = int(config.get("extra_top", 30))
    extra_bottom_px = int(config.get("extra_bottom", 30))
    auto_bottom_buffer = bool(config.get("auto_bottom_buffer", True))
    scroll_preview = bool(config.get("scroll_preview", True))

    if show_style:
        st.divider()
        st.subheader("Canvas / Cropping")
        canvas_padding_px = st.slider("Canvas padding (px)", 100, 200, canvas_padding_px, key="cfg_canvas_padding")
        extra_top_px = st.slider("Extra top space (px)", 0, 200, extra_top_px, key="cfg_extra_top")
        extra_bottom_px = st.slider("Extra bottom space (px)", 0, 200, extra_bottom_px, key="cfg_extra_bottom")
        auto_bottom_buffer = st.checkbox("Auto add bottom buffer for dims/labels", auto_bottom_buffer, key="cfg_auto_bottom_buffer")
        scroll_preview = st.checkbox("Make preview scrollable", scroll_preview, key="cfg_scroll_preview")

    cfg_set("canvas_padding", canvas_padding_px)
    cfg_set("extra_top", extra_top_px)
    cfg_set("extra_bottom", extra_bottom_px)
    cfg_set("auto_bottom_buffer", auto_bottom_buffer)
    cfg_set("scroll_preview", scroll_preview)

# Parse mids
parsed_mid_components = []
for line in [l.strip() for l in mid_str.splitlines() if l.strip()]:
    try:
        pos, pn = line.split(":",1)
        pos_m = parse_length_string(pos.strip(), measurement_system)
        if pos_m is None:
            continue
        parsed_mid_components.append(MidComponent(pos_m, pn.strip()))
    except Exception:
        pass
base_spec.mid_components = parsed_mid_components

# Style dict
style = dict(
    font_px=font_px, track_stroke=track_stroke, dim_stroke=dim_stroke, node_size=node_size,
    seg_label_off=seg_label_off_px, join_label_off=join_label_off_px, corner_label_off=corner_label_off_px,
    end_label_off=end_label_off_px, mid_label_off=mid_label_offset_px,
    dim_off=dim_offset_px, dim_side_extra=dim_side_extra_px, title_y=title_offset_px,
    show_ticks=show_segment_ticks, tick_len=tick_len_px, show_element_labels=show_element_labels,
    inline_join_types=config.get("inline_join_types", {}), inline_join_isolation={}, corner_isolation={},
    isolation_mark_len=isolation_mark_len_px,
    pad=canvas_padding_px, extra_top=extra_top_px, extra_bottom=extra_bottom_px,
    auto_bottom_buffer=auto_bottom_buffer, scroll_preview=scroll_preview, cover_strip_on=cover_strip_on,
    measurement_system=measurement_system
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
    default_end = end_options[0] if end_options else ""
    sel_end1_default = config.get("start_end", default_end)
    sel_end2_default = config.get("end_end", default_end)
    sel_end1 = st.selectbox("End 1", end_options or [""], index=_safe_index(end_options or [""], sel_end1_default), key="cfg_end1")
    sel_end2 = st.selectbox("End 2", end_options or [""], index=_safe_index(end_options or [""], sel_end2_default), key="cfg_end2")
    cfg_set("start_end", sel_end1)
    cfg_set("end_end", sel_end2)

    corner_options = options_by_type.get("Corner") or options_by_type.get("Join") or []
    internal_nodes = list(range(1, max(0, len(pts_ui)-1)))
    corner_labels = [f"Corner {i}" for i,_ in enumerate(internal_nodes, start=1)]
    corner_selections = []
    corner_keys = ["corner1", "corner2", "corner3"]
    for idx, label in enumerate(corner_labels):
        stored_val = config.get(corner_keys[idx], "") if idx < len(corner_keys) else ""
        choice = st.selectbox(label, corner_options or [""], index=_safe_index(corner_options or [""], stored_val if stored_val else (corner_options[0] if corner_options else "")), key=f"cfg_{label}")
        if idx < len(corner_keys):
            cfg_set(corner_keys[idx], choice)
        corner_selections.append(choice)
    for idx in range(len(corner_keys)):
        if idx >= len(corner_selections):
            cfg_set(corner_keys[idx], "")
    corner_iso = {
        label: option_has_isolation(option_meta_lookup, choice, expected_type="Corner")
        for label, choice in zip(corner_labels, corner_selections)
    }
    style["corner_isolation"] = corner_iso

    st.subheader("Inline joins (stock boundaries)")
    join_options = options_by_type.get("Join", [])
    stored_inline = config.get("inline_join_types", {})
    join_type_by_label = {}
    if inline_joins:
        for j in inline_joins:
            lbl = j['label']
            default_join = stored_inline.get(lbl, "")
            join_choice = st.selectbox(lbl, join_options or [""], index=_safe_index(join_options or [""], default_join), key=f"cfg_inline_{lbl}")
            join_type_by_label[lbl] = join_choice
    else:
        st.caption("No inline joins for current geometry/stock.")
    cfg_set("inline_join_types", join_type_by_label)
    style["inline_join_types"] = join_type_by_label
    style["inline_join_isolation"] = {
        lbl: option_has_isolation(option_meta_lookup, choice, expected_type="Join")
        for lbl, choice in join_type_by_label.items()
    }

# Update spec selections
base_spec = LayoutSpec(
    name=name, shape=shape, length_m=length, width_m=width, depth_m=depth,
    stock=stock_selected or [2.0, 1.0],
    max_run_m=max_run_meters,
    start_end=sel_end1,
    end_end=sel_end2,
    corner1_join=corner_selections[0] if len(corner_selections) >= 1 else config.get("corner1", ""),
    corner2_join=corner_selections[1] if len(corner_selections) >= 2 else config.get("corner2", ""),
    corner3_join=corner_selections[2] if len(corner_selections) >= 3 else config.get("corner3", ""),
    mid_components=parsed_mid_components
)

# Recompute plan & render
plan = compute_plan(base_spec, measurement_system)

feed_count = 0
if is_feed_label(base_spec.start_end):
    feed_count += 1
if is_feed_label(base_spec.end_end):
    feed_count += 1
for corner_choice in [base_spec.corner1_join, base_spec.corner2_join, base_spec.corner3_join]:
    if is_feed_label(corner_choice):
        feed_count += 1
for join_name in style.get("inline_join_types", {}).values():
    if is_feed_label(join_name):
        feed_count += 1
for mc in base_spec.mid_components:
    if is_feed_label(mc.part_no):
        feed_count += 1

isolator_count = 0
if option_has_isolation(option_meta_lookup, base_spec.start_end, expected_type="End"):
    isolator_count += 1
if option_has_isolation(option_meta_lookup, base_spec.end_end, expected_type="End"):
    isolator_count += 1
for corner_choice in [base_spec.corner1_join, base_spec.corner2_join, base_spec.corner3_join]:
    if option_has_isolation(option_meta_lookup, corner_choice, expected_type="Corner"):
        isolator_count += 1
for join_name in style.get("inline_join_types", {}).values():
    if option_has_isolation(option_meta_lookup, join_name, expected_type="Join"):
        isolator_count += 1
for mc in base_spec.mid_components:
    if option_has_isolation(option_meta_lookup, mc.part_no):
        isolator_count += 1

svg, svg_h, _ = render_track_svg(base_spec, plan, style)
components.html(svg, height=svg_h, scrolling=style.get("scroll_preview", True))

# Allow users to grab the raw vector artwork for external editing/printing.
st.download_button(
    "Download diagram as SVG",
    data=svg.encode("utf-8"),
    file_name=f"{base_spec.name}_diagram.svg",
    mime="image/svg+xml"
)

if feed_count == 0:
    st.error("No feeds detected in this layout. Add at least one end, inline, or mid-run feed before finalizing.")
else:
    isolator_text = f" • Isolations detected: {isolator_count}" if isolator_count > 0 else ""
    if feed_count > 1 and isolator_count == 0:
        st.warning(f"Feeds detected: {feed_count}{isolator_text or ' • No isolations detected'}")
    else:
        st.success(f"Feeds detected: {feed_count}{isolator_text or ' • No isolations detected'}")

# Minimum-length rule messages
if plan.get("rules"):
    for r in plan["rules"]:
        if r.get("level") == "error":
            st.error(r.get("msg", ""))
        elif r.get("level") == "warn":
            st.warning(r.get("msg", ""))
        else:
            st.success(r.get("msg", ""))

# =========================================================
# Build BOM
# =========================================================
profile_col = PROFILE_TO_COL[track_profile]

options_lookup_df = df_opts_filtered if df_opts_filtered is not None else df_opts

def choose_track_name_for_donor(donor_len):
    if not track_name_to_len:
        return None
    exact = [nm for nm, L in track_name_to_len.items() if abs(L - donor_len) < 1e-6]
    if exact: return exact[0]
    bigger = sorted([(L, nm) for nm, L in track_name_to_len.items() if L + 1e-9 >= donor_len])
    if bigger: return bigger[0][1]
    return max(track_name_to_len.items(), key=lambda kv: kv[1])[0]

rows = []  # [Reference label, Name, Part no, QTY]

def fetch_option_parts(display_name, expected_type=None):
    dn = str(display_name).strip()
    lookup_df = options_lookup_df if options_lookup_df is not None else pd.DataFrame()
    if not dn:
        return None
    if lookup_df is None or lookup_df.empty:
        return {"name": dn, "parts": []}
    candidates = lookup_df[lookup_df["Name"].str.lower() == dn.lower()]
    if expected_type and not candidates.empty:
        candidates = candidates[candidates["Type"].str.lower() == expected_type.lower()]
    if candidates is None or candidates.empty:
        return {"name": dn, "parts": []}
    row0 = candidates.iloc[0]
    canonical_name = str(row0["Name"]).strip()
    parts = apply_finish_tokens(row0.get(profile_col, ""), finish_token)
    circuit_col = next((c for c in row0.index if str(c).strip().lower() == "circuit"), None)
    circuit_val = str(row0[circuit_col]).strip() if circuit_col else ""
    return {"name": canonical_name, "parts": parts, "circuit": circuit_val}

def add_excel_rows(ref_label, display_name, expected_type=None):
    dn = str(display_name).strip()
    if not dn:
        rows.append([ref_label, "--", "--", "--"]); return
    result = fetch_option_parts(dn, expected_type=expected_type)
    canonical_name = result["name"] if result else dn
    parts = result["parts"] if result else []
    if not parts:
        rows.append([ref_label, canonical_name, "--", "--"]); return
    for p in parts:
        rows.append([ref_label, canonical_name, p, 1])

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
        ref_base = f"Leg {i+1} Seg {seg_idx}"
        needs_cut = str(getattr(s, "kind", "")).lower() == "cut"
        ref_label = f"{ref_base} (Cut required)" if needs_cut else ref_base
        if track_name:
            add_excel_rows(ref_label, track_name, expected_type="Track")
        else:
            rows.append([ref_label, format_length(donor_len, measurement_system), "--", "--"])
        seg_idx += 1

# Mid components
for idx, mc in enumerate(base_spec.mid_components, start=1):
    rows.append([f"MID {idx}", mc.part_no, mc.part_no, 1])

# Cover strip row
cover_part_lookup = fetch_option_parts(cover_name_ui, expected_type="Cover Strip") if style.get("cover_strip_on") else None
if style.get("cover_strip_on") and cover_name_ui:
    total_len_m = float(plan.get("total_len", 0.0) or 0.0)
    sticks_needed = int(math.ceil(total_len_m / COVER_STRIP_STICK_M)) if total_len_m > 0 else 0
    cover_parts = cover_part_lookup["parts"] if cover_part_lookup else []
    cover_display_name = cover_part_lookup["name"] if cover_part_lookup else cover_name_ui
    cover_part_ui = cover_parts[0] if cover_parts else ""
    cfg_set("cover_part", cover_part_ui)
    if sticks_needed > 0:
        if cover_parts:
            for part in cover_parts:
                rows.append(["COVER STRIP", cover_display_name, part, sticks_needed])
        else:
            rows.append(["COVER STRIP", cover_display_name or "--", "--", sticks_needed])

# Mounting hardware row
if use_mount and mh_name.strip():
    spacing_val = float(mh_spacing) if mh_spacing else None
    qty_each = float(mh_qty_each) if mh_qty_each else 1.0
    total_points = 0
    if spacing_val and spacing_val > 0:
        for leg_len in seg_lens_bom:
            if leg_len <= 0:
                continue
            segments_needed = max(1, int(math.ceil(leg_len / spacing_val)))
            total_points += segments_needed + 1
        shared_nodes = max(len(pts_bom) - 2, 0)
        total_points = max(total_points - shared_nodes, len(pts_bom))
    else:
        total_points = len(pts_bom) if pts_bom else 0
    if total_points > 0 and qty_each > 0:
        total_qty = int(math.ceil(total_points * qty_each))
        rows.append(["MOUNTING", mh_name.strip(), mh_part.strip() if mh_part.strip() else "--", total_qty])

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
