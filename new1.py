import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import time
import zipfile
import io
from PIL import Image
from sklearn.cluster import KMeans

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MAP_CONFIG = {
    "AmbroseValley": {"scale": 900,  "origin_x": -370, "origin_z": -473},
    "GrandRift":     {"scale": 581,  "origin_x": -290, "origin_z": -290},
    "Lockdown":      {"scale": 1000, "origin_x": -500, "origin_z": -500},
}

MINIMAP_PATHS = {
    "AmbroseValley": "player_data/minimaps/AmbroseValley_Minimap.png",
    "GrandRift":     "player_data/minimaps/GrandRift_Minimap.png",
    "Lockdown":      "player_data/minimaps/Lockdown_Minimap.jpg",
}

DATA_FOLDERS = {
    "February_10": "player_data/February_10",
    "February_11": "player_data/February_11",
    "February_12": "player_data/February_12",
    "February_13": "player_data/February_13",
    "February_14": "player_data/February_14",
}

MAPS = ["AmbroseValley", "GrandRift", "Lockdown"]

ZONE_STYLES = {
    "Traffic (All Movement)": {
        "events":      ["Position", "BotPosition"],
        "color":       "255,100,0",
        "label_color": "#ff6400",
        "label":       "HIGH TRAFFIC"
    },
    "Kill Zones": {
        "events":      ["Kill", "BotKill"],
        "color":       "255,0,0",
        "label_color": "#ff0000",
        "label":       "KILL ZONE"
    },
    "Death Zones": {
        "events":      ["Killed", "BotKilled", "KilledByStorm"],
        "color":       "255,60,0",
        "label_color": "#ff3c00",
        "label":       "DEATH ZONE"
    },
    "Loot Zones": {
        "events":      ["Loot"],
        "color":       "0,180,255",
        "label_color": "#00b4ff",
        "label":       "LOOT ZONE"
    },
    "Storm Deaths": {
        "events":      ["KilledByStorm"],
        "color":       "160,0,255",
        "label_color": "#aa00ff",
        "label":       "STORM ZONE"
    },
}

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def world_to_pixel(x, z, map_id):
    cfg = MAP_CONFIG[map_id]
    u  = (x - cfg["origin_x"]) / cfg["scale"]
    v  = (z - cfg["origin_z"]) / cfg["scale"]
    return u * 1024, (1 - v) * 1024

def fmt_time(s):
    return f"{int(s)//60}:{int(s)%60:02d}"

def format_day_label(day):
    return day.replace("_", " ").title()

# ─────────────────────────────────────────
# LOAD DATA — LOCAL
# ─────────────────────────────────────────
@st.cache_data
def load_all_data():
    all_frames = []
    for day_name, folder_path in DATA_FOLDERS.items():
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            try:
                df = pq.read_table(filepath).to_pandas()
                df['event'] = df['event'].apply(
                    lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                )
                df['day']      = day_name
                df['is_bot']   = df['user_id'].apply(lambda x: str(x).isdigit())
                df['match_id'] = df['match_id'].apply(
                    lambda x: x.replace('.nakama-0', '') if isinstance(x, str) else x
                )
                all_frames.append(df)
            except Exception:
                continue
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

# ─────────────────────────────────────────
# LOAD DATA — FROM ZIP
# ─────────────────────────────────────────
@st.cache_data
def load_from_zip(zip_bytes):
    all_frames     = []
    minimap_images = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            # ── Load minimaps ──
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                for map_key in ["AmbroseValley", "GrandRift", "Lockdown"]:
                    if map_key.lower() in name.lower():
                        with z.open(name) as f:
                            minimap_images[map_key] = Image.open(f).copy()
            # ── Skip directories ──
            if name.endswith('/'):
                continue
            # ── Skip files with extensions ──
            fname = name.split('/')[-1]
            if '.' in fname:
                continue
            # ── Auto detect day from parent folder ──
            parts    = name.split('/')
            day_name = parts[-2] if len(parts) >= 2 else "Unknown"
            try:
                with z.open(name) as f:
                    df = pq.read_table(f).to_pandas()
                df['event'] = df['event'].apply(
                    lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                )
                df['day']      = day_name
                df['is_bot']   = df['user_id'].apply(lambda x: str(x).isdigit())
                df['match_id'] = df['match_id'].apply(
                    lambda x: x.replace('.nakama-0', '') if isinstance(x, str) else x
                )
                all_frames.append(df)
            except Exception:
                continue
    df_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    return df_all, minimap_images

# ─────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────
def get_minimap_figure(map_id, height=600):
    if "minimap_images" in st.session_state and map_id in st.session_state.minimap_images:
        img = st.session_state.minimap_images[map_id]
    else:
        img = Image.open(MINIMAP_PATHS[map_id])
    fig = go.Figure()
    fig.add_layout_image(dict(
        source=img, xref="x", yref="y",
        x=0, y=1024, sizex=1024, sizey=1024,
        sizing="stretch", layer="below"
    ))
    fig.update_layout(
        xaxis=dict(range=[0,1024], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(range=[0,1024], showgrid=False,
                   zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        height=height,
        legend=dict(
            bgcolor="rgba(10,10,15,0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(color="#ffffff", size=11),
        )
    )
    return fig

def add_pubg_style_zones(fig, data, map_id, zone_style, n_clusters=8):
    filtered = data[data['event'].isin(zone_style["events"])]
    if filtered.empty:
        return fig
    color   = zone_style["color"]
    label_c = zone_style["label_color"]
    label   = zone_style["label"]
    coords  = filtered.apply(
        lambda r: world_to_pixel(r['x'], r['z'], map_id), axis=1
    )
    px_x   = np.array([c[0] for c in coords])
    px_y   = np.array([c[1] for c in coords])
    points = np.column_stack([px_x, px_y])
    k      = min(n_clusters, len(points))
    if k < 2:
        return fig
    kmeans    = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(points)
    centers   = kmeans.cluster_centers_
    labels_km = kmeans.labels_
    counts    = np.bincount(labels_km)
    max_count = counts.max()
    for i, (cx, cy) in enumerate(centers):
        count     = counts[i]
        intensity = count / max_count
        radius    = 30 + intensity * 80
        opacity   = 0.20 + intensity * 0.30
        theta     = np.linspace(0, 2*np.pi, 60)
        fig.add_trace(go.Scatter(
            x=cx + radius*np.cos(theta),
            y=cy + radius*np.sin(theta),
            fill="toself",
            fillcolor=f"rgba({color},{opacity:.2f})",
            line=dict(color=f"rgba({color},0.85)", width=2),
            mode="lines", hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(
            x=cx, y=cy+radius+14, text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color=label_c, size=10, family="Arial Black"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor=label_c, borderwidth=1, borderpad=3,
        )
        fig.add_annotation(
            x=cx, y=cy, text=f"<b>{count}</b>",
            showarrow=False,
            font=dict(color="#ffffff", size=11, family="Arial Black"),
            bgcolor=f"rgba({color},0.85)", borderpad=4,
        )
    return fig

def add_event_markers(fig, data, map_id):
    specs = {
        "Kill":          ("#ff4444", 22, "✖"),
        "Killed":        ("#ff8844", 22, "💀"),
        "BotKill":       ("#ff6644", 20, "✖"),
        "BotKilled":     ("#ffaa44", 20, "💀"),
        "KilledByStorm": ("#aa44ff", 26, "⚡"),
        "Loot":          ("#44aaff", 20, "📦"),
    }
    for event_type, (color, size, icon) in specs.items():
        edata = data[data['event'] == event_type]
        if edata.empty:
            continue
        coords = edata.apply(
            lambda r: world_to_pixel(r['x'], r['z'], map_id), axis=1
        )
        px_x = [c[0] for c in coords]
        px_y = [c[1] for c in coords]
        fig.add_trace(go.Scatter(
            x=px_x, y=px_y, mode="markers",
            marker=dict(color=color, size=size, opacity=1.0,
                        line=dict(width=3, color="#ffffff")),
            name=f"{icon} {event_type}",
            hovertemplate=f"<b>{event_type}</b><br>x:%{{x:.0f}} y:%{{y:.0f}}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=px_x, y=px_y, mode="text",
            text=[icon]*len(px_x),
            textposition="middle center",
            textfont=dict(size=14, color="#ffffff"),
            hoverinfo="skip", showlegend=False,
        ))
    return fig

def add_player_journey(fig, player_data, map_id, color, name):
    positions = player_data[
        player_data['event'].isin(['Position','BotPosition'])
    ].sort_values('ts')
    if positions.empty:
        return fig
    coords = positions.apply(
        lambda r: world_to_pixel(r['x'], r['z'], map_id), axis=1
    )
    px_x = [c[0] for c in coords]
    px_y = [c[1] for c in coords]
    fig.add_trace(go.Scatter(
        x=px_x, y=px_y, mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=3, color=color),
        name=name, opacity=0.85,
        hovertemplate=f"<b>{name}</b><extra></extra>"
    ))
    if px_x:
        fig.add_trace(go.Scatter(
            x=[px_x[0]], y=[px_y[0]], mode="markers",
            marker=dict(size=16, color="#00ff88", symbol="circle",
                        line=dict(width=2, color="#ffffff")),
            name="▶ Start",
        ))
        fig.add_trace(go.Scatter(
            x=[px_x[-1]], y=[px_y[-1]], mode="markers",
            marker=dict(size=16, color="#ff4444", symbol="x",
                        line=dict(width=2, color="#ffffff")),
            name="✖ End",
        ))
    return fig

# ─────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────
st.set_page_config(
    page_title="LILA BLACK — Player Journey Tool",
    page_icon="🎮", layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #0a0a0f; color: #ffffff; }
h1 { font-size:2.4rem !important; font-weight:900 !important;
     letter-spacing:2px !important; color:#ffffff !important; }
h2,h3,h4 { color:#ffffff !important; font-weight:800 !important; }
p,label,.stMarkdown { color:#cccccc !important; font-size:15px !important; }
[data-testid="stMetric"] {
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border:1px solid rgba(0,255,136,0.27); border-radius:12px;
    padding:16px !important; box-shadow:0 0 20px rgba(0,255,136,0.07);
}
[data-testid="stMetricLabel"] {
    color:#aaaaaa !important; font-size:13px !important;
    font-weight:600 !important; text-transform:uppercase; letter-spacing:1px;
}
[data-testid="stMetricValue"] {
    color:#00ff88 !important; font-size:2rem !important; font-weight:900 !important;
}
.stButton > button {
    background:linear-gradient(135deg,#1a1a2e,#16213e) !important;
    color:#00ff88 !important; border:2px solid rgba(0,255,136,0.4) !important;
    border-radius:10px !important; font-size:1.2rem !important;
    height:50px !important; transition:all 0.3s ease !important;
}
.stButton > button:hover {
    border-color:#00ff88 !important;
    box-shadow:0 0 25px rgba(0,255,136,0.33) !important;
}
.stSelectbox > div {
    background:#1a1a2e !important;
    border:1px solid rgba(0,255,136,0.27) !important;
    border-radius:8px !important;
}
hr { border-color:rgba(255,255,255,0.07) !important; }
#MainMenu { visibility:hidden; } footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px 0 10px 0;'>
    <div style='font-size:13px; letter-spacing:6px; color:#00ff88;
                font-weight:700; margin-bottom:8px;'>
        LILA GAMES — INTERNAL TOOL
    </div>
    <h1 style='font-size:2.8rem; font-weight:900; color:#ffffff;
               letter-spacing:3px; margin:0;'>🎮 LILA BLACK</h1>
    <div style='font-size:16px; color:#888; margin-top:8px; letter-spacing:2px;'>
        PLAYER JOURNEY VISUALIZATION
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-left:4px solid #00ff88; border-radius:8px;
            padding:14px 20px; margin-bottom:16px;'>
    <div style='font-size:11px; letter-spacing:4px;
                color:#00ff88; font-weight:700;'>DATA SOURCE</div>
    <div style='font-size:20px; font-weight:900; color:#ffffff;'>
        Upload Player Data</div>
    <div style='font-size:13px; color:#888; margin-top:4px;'>
        Upload your player_data ZIP file containing parquet files and minimaps.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_zip = st.file_uploader(
    "Upload player_data ZIP",
    type=["zip"],
    label_visibility="collapsed"
)

if "minimap_images" not in st.session_state:
    st.session_state.minimap_images = {}

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
if uploaded_zip is not None:
    with st.spinner("Extracting and loading data from ZIP..."):
        df, zip_minimaps = load_from_zip(uploaded_zip.read())
        if zip_minimaps:
            st.session_state.minimap_images = zip_minimaps
    if df.empty:
        st.error("No parquet data found in ZIP — check folder structure")
        st.stop()
    else:
        st.success(f"✅ Loaded {len(df):,} events from uploaded ZIP")
else:
    with st.spinner("Loading local gameplay data..."):
        df = load_all_data()
    if df.empty:
        st.markdown("""
        <div style='text-align:center; padding:60px 40px;
                    background:#111827; border-radius:12px;
                    border:1px solid rgba(0,255,136,0.2);
                    margin-top:16px;'>
            <div style='font-size:40px; margin-bottom:16px;'>📁</div>
            <div style='color:#00ff88; font-size:18px; font-weight:700;
                        letter-spacing:2px;'>NO DATA FOUND</div>
            <div style='color:#555; font-size:13px; margin-top:10px;'>
                Please upload your player_data ZIP file above to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

st.divider()

# ─────────────────────────────────────────
# TOP STATS
# ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events",   f"{len(df):,}")
col2.metric("Unique Players", f"{df[df['is_bot']==False]['user_id'].nunique():,}")
col3.metric("Unique Matches", f"{df['match_id'].nunique():,}")
col4.metric("Maps Available", f"{df['map_id'].nunique()}")

st.divider()

# ─────────────────────────────────────────
# MAP SELECTOR
# ─────────────────────────────────────────
if "map_index" not in st.session_state:
    st.session_state.map_index = 0

st.markdown("""
<div style='font-size:12px; letter-spacing:4px; color:#00ff88;
            font-weight:700; margin-bottom:4px;'>MAP EXPLORER</div>
""", unsafe_allow_html=True)

left_col, mid_col, right_col = st.columns([1, 6, 1])
with left_col:
    st.write("")
    st.write("")
    if st.button("◀", use_container_width=True):
        st.session_state.map_index = (st.session_state.map_index - 1) % len(MAPS)
with right_col:
    st.write("")
    st.write("")
    if st.button("▶", use_container_width=True):
        st.session_state.map_index = (st.session_state.map_index + 1) % len(MAPS)

selected_map = MAPS[st.session_state.map_index]

with mid_col:
    st.markdown(
        f"<h2 style='text-align:center; color:#ffffff; font-weight:900;"
        f"letter-spacing:3px;'>{selected_map.upper()}</h2>",
        unsafe_allow_html=True
    )
    dots = "".join([
        "<span style='color:#00ff88; font-size:22px;'>●</span> "
        if i == st.session_state.map_index
        else "<span style='color:#333; font-size:22px;'>●</span> "
        for i in range(len(MAPS))
    ])
    st.markdown(f"<p style='text-align:center;'>{dots}</p>",
                unsafe_allow_html=True)

map_data = df[df['map_id'] == selected_map].copy()

st.divider()

# ─────────────────────────────────────────
# DATE FILTER — DYNAMIC
# ─────────────────────────────────────────
st.markdown("""
<div style='font-size:12px; letter-spacing:4px; color:#00ff88;
            font-weight:700; margin-bottom:12px;'>FILTER BY DATE</div>
""", unsafe_allow_html=True)

available_days = sorted(map_data['day'].unique().tolist())

if not available_days:
    st.warning("No data available for this map.")
    st.stop()

date_cols     = st.columns(len(available_days))
selected_days = []

for i, day in enumerate(available_days):
    with date_cols[i]:
        day_count     = map_data[map_data['day'] == day]['match_id'].nunique()
        display_label = day.replace("_", " ").title()
        checked       = st.checkbox(
            f"**{display_label}**  \n`{day_count} matches`",
            value=True,
            key=f"day_{day}"
        )
        if checked:
            selected_days.append(day)

if not selected_days:
    st.warning("Please select at least one date.")
    st.stop()

map_data = map_data[map_data['day'].isin(selected_days)].copy()

st.markdown(
    f"<div style='font-size:13px; color:#888; margin-top:8px; margin-bottom:4px;'>"
    f"Showing <span style='color:#00ff88; font-weight:700;'>"
    f"{map_data['match_id'].nunique()}</span> matches across "
    f"<span style='color:#00ff88; font-weight:700;'>{len(selected_days)}</span>"
    f" day(s)</div>",
    unsafe_allow_html=True
)

st.divider()

# ═════════════════════════════════════════
# LEVEL 1 — HEATMAP
# ═════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-left:4px solid #00ff88; border-radius:8px;
            padding:14px 20px; margin-bottom:16px;'>
    <div style='font-size:11px; letter-spacing:4px;
                color:#00ff88; font-weight:700;'>LEVEL 1</div>
    <div style='font-size:20px; font-weight:900; color:#ffffff;'>
        HEATMAP — All Matches Combined</div>
    <div style='font-size:13px; color:#888; margin-top:4px;'>
        Where do players generally go? Big picture traffic patterns.
    </div>
</div>
""", unsafe_allow_html=True)

heatmap_col, control_col = st.columns([3, 1])

with control_col:
    st.markdown("<div style='color:#00ff88; font-weight:700; "
                "letter-spacing:2px; font-size:12px;'>ZONE TYPE</div>",
                unsafe_allow_html=True)
    heatmap_type = st.radio(
        "Select type:", list(ZONE_STYLES.keys()),
        label_visibility="collapsed"
    )
    n_clusters = st.slider("Number of hotspot zones", 4, 15, 8)
    st.divider()
    st.markdown("<div style='color:#00ff88; font-weight:700; "
                "letter-spacing:2px; font-size:12px;'>MAP STATS</div>",
                unsafe_allow_html=True)
    st.metric("Total Events",   f"{len(map_data):,}")
    st.metric("Human Players",
              f"{map_data[map_data['is_bot']==False]['user_id'].nunique():,}")
    st.metric("Matches",        f"{map_data['match_id'].nunique():,}")
    st.divider()
    st.markdown("<div style='color:#00ff88; font-weight:700; "
                "letter-spacing:2px; font-size:12px; "
                "margin-bottom:8px;'>LEGEND</div>",
                unsafe_allow_html=True)
    for clr, lbl in [
        ("#ff0000", "High activity zone"),
        ("#ffaa00", "Medium activity"),
        ("#ffffff", "Zone label + count"),
    ]:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px; "
            f"margin-bottom:6px;'>"
            f"<div style='width:14px; height:14px; border-radius:50%; "
            f"background:{clr};'></div>"
            f"<span style='color:#cccccc; font-size:12px;'>{lbl}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

with heatmap_col:
    fig1 = get_minimap_figure(selected_map, height=650)
    fig1 = add_pubg_style_zones(
        fig1, map_data, selected_map,
        ZONE_STYLES[heatmap_type], n_clusters
    )
    st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ═════════════════════════════════════════
# LEVEL 2 — SINGLE MATCH VIEW
# ═════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-left:4px solid #ffaa00; border-radius:8px;
            padding:14px 20px; margin-bottom:16px;'>
    <div style='font-size:11px; letter-spacing:4px;
                color:#ffaa00; font-weight:700;'>LEVEL 2</div>
    <div style='font-size:20px; font-weight:900; color:#ffffff;'>
        MATCH EXPLORER — Single Match View</div>
    <div style='font-size:13px; color:#888; margin-top:4px;'>
        Pick one match and see exactly what happened.
    </div>
</div>
""", unsafe_allow_html=True)

match_ids            = sorted(map_data['match_id'].unique().tolist())
match_labels         = {mid: f"Match {i+1}" for i, mid in enumerate(match_ids)}
match_col, info_col2 = st.columns([3, 1])

with info_col2:
    st.markdown("<div style='color:#ffaa00; font-weight:700; "
                "letter-spacing:2px; font-size:12px;'>SELECT MATCH</div>",
                unsafe_allow_html=True)
    selected_match = st.selectbox(
        "Match:", options=match_ids,
        format_func=lambda x: match_labels[x],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("<div style='color:#ffaa00; font-weight:700; "
                "letter-spacing:2px; font-size:12px;'>FILTERS</div>",
                unsafe_allow_html=True)
    show_humans = st.checkbox("Show Human Events",    value=True)
    show_bots   = st.checkbox("Show Bot Events",      value=False)
    show_kills  = st.checkbox("Show Kills ✖",         value=True)
    show_deaths = st.checkbox("Show Deaths 💀",       value=True)
    show_loot   = st.checkbox("Show Loot 📦",         value=True)
    show_storm  = st.checkbox("Show Storm Deaths ⚡", value=True)
    st.divider()
    st.markdown("<div style='color:#ffaa00; font-weight:700; "
                "letter-spacing:2px; font-size:12px; "
                "margin-bottom:8px;'>EVENT LEGEND</div>",
                unsafe_allow_html=True)
    for clr, lbl in [
        ("#ff4444", "✖ Kill"),
        ("#ff8844", "💀 Killed"),
        ("#ff6644", "✖ Bot Kill"),
        ("#ffaa44", "💀 Bot Killed"),
        ("#aa44ff", "⚡ Storm Death"),
        ("#44aaff", "📦 Loot"),
    ]:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px; "
            f"margin-bottom:5px;'>"
            f"<div style='width:12px; height:12px; border-radius:50%; "
            f"background:{clr};'></div>"
            f"<span style='color:#cccccc; font-size:12px;'>{lbl}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

with match_col:
    match_data     = map_data[map_data['match_id'] == selected_match].copy()
    allowed_events = []
    if show_humans:
        if show_kills:  allowed_events.extend(["Kill"])
        if show_deaths: allowed_events.extend(["Killed"])
        if show_loot:   allowed_events.extend(["Loot"])
        if show_storm:  allowed_events.extend(["KilledByStorm"])
    if show_bots:
        if show_kills:  allowed_events.extend(["BotKill"])
        if show_deaths: allowed_events.extend(["BotKilled"])

    filtered_match  = match_data[match_data['event'].isin(allowed_events)]
    humans_in_match = match_data[match_data['is_bot']==False]['user_id'].nunique()
    bots_in_match   = match_data[match_data['is_bot']==True]['user_id'].nunique()

    fig2 = get_minimap_figure(selected_map, height=600)
    fig2 = add_event_markers(fig2, filtered_match, selected_map)
    fig2.add_annotation(
        x=10, y=1010,
        text=(f"👤 {humans_in_match} Humans  "
              f"🤖 {bots_in_match} Bots  "
              f"📊 {len(match_data)} Events"),
        showarrow=False,
        font=dict(color="#ffffff", size=12),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="rgba(0,255,136,0.3)",
        borderwidth=1, xanchor="left"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ═════════════════════════════════════════
# LEVEL 3 — SINGLE PLAYER JOURNEY
# ═════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-left:4px solid #ff4444; border-radius:8px;
            padding:14px 20px; margin-bottom:16px;'>
    <div style='font-size:11px; letter-spacing:4px;
                color:#ff4444; font-weight:700;'>LEVEL 3</div>
    <div style='font-size:20px; font-weight:900; color:#ffffff;'>
        PLAYER JOURNEY — Single Player Path</div>
    <div style='font-size:13px; color:#888; margin-top:4px;'>
        Follow one player's exact path from start to finish.
    </div>
</div>
""", unsafe_allow_html=True)

players_in_match = match_data['user_id'].unique().tolist()
humans_list      = [p for p in players_in_match if not str(p).isdigit()]
bots_list        = [p for p in players_in_match if str(p).isdigit()]
journey_col, player_col = st.columns([3, 1])

with player_col:
    st.markdown("<div style='color:#ff4444; font-weight:700; "
                "letter-spacing:2px; font-size:12px;'>SELECT PLAYER</div>",
                unsafe_allow_html=True)
    player_type = st.radio("Player type:", ["Human", "Bot"], horizontal=True)
    player_list = humans_list if player_type == "Human" else bots_list

    if player_list:
        if player_type == "Human":
            player_labels = {p: f"👤 Player {i+1}"
                             for i, p in enumerate(player_list)}
        else:
            player_labels = {p: f"🤖 Bot Player {i+1}"
                             for i, p in enumerate(player_list)}
        selected_player = st.selectbox(
            "Player:", options=player_list,
            format_func=lambda x: player_labels[x],
            label_visibility="collapsed"
        )
    else:
        st.warning(f"No {player_type.lower()} players in this match")
        selected_player = None

    if selected_player:
        player_events = match_data[match_data['user_id'] == selected_player]
        st.divider()
        st.markdown("<div style='color:#ff4444; font-weight:700; "
                    "letter-spacing:2px; font-size:12px;'>PLAYER STATS</div>",
                    unsafe_allow_html=True)
        st.metric("Total Events", len(player_events))
        st.metric("Kills",
                  len(player_events[player_events['event'].isin(
                      ['Kill','BotKill'])]))
        st.metric("Deaths",
                  len(player_events[player_events['event'].isin(
                      ['Killed','BotKilled','KilledByStorm'])]))
        st.metric("Loots",
                  len(player_events[player_events['event'] == 'Loot']))

with journey_col:
    if selected_player:
        player_events = match_data[match_data['user_id'] == selected_player]
        is_bot = str(selected_player).isdigit()
        color  = "#ff4444" if is_bot else "#00ff88"
        label  = player_labels[selected_player]
        fig3   = get_minimap_figure(selected_map, height=600)
        fig3   = add_player_journey(fig3, player_events, selected_map, color, label)
        fig3   = add_event_markers(fig3, player_events, selected_map)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
        <div style='display:flex; gap:20px; padding:12px 16px;
                    background:#111827; border-radius:8px;
                    border:1px solid rgba(255,255,255,0.07); flex-wrap:wrap;'>
            <span style='color:#00ff88; font-size:13px;'>● Start</span>
            <span style='color:#ff4444; font-size:13px;'>✖ End</span>
            <span style='color:#ff4444; font-size:13px;'>✖ Kill</span>
            <span style='color:#ff8844; font-size:13px;'>✖ Death</span>
            <span style='color:#44aaff; font-size:13px;'>📦 Loot</span>
            <span style='color:#aa44ff; font-size:13px;'>⚡ Storm</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ═════════════════════════════════════════
# LEVEL 4 — TIMELINE PLAYBACK
# ═════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-left:4px solid #00aaff; border-radius:8px;
            padding:14px 20px; margin-bottom:16px;'>
    <div style='font-size:11px; letter-spacing:4px;
                color:#00aaff; font-weight:700;'>LEVEL 4</div>
    <div style='font-size:20px; font-weight:900; color:#ffffff;'>
        TIMELINE PLAYBACK — Watch Match Unfold</div>
    <div style='font-size:13px; color:#888; margin-top:4px;'>
        Watch the selected player's journey play out step by step.
    </div>
</div>
""", unsafe_allow_html=True)

if not selected_player:
    st.info("👆 Select a player in Level 3 above to enable playback")
else:
    player_events_tl = match_data[
        match_data['user_id'] == selected_player
    ].copy().sort_values('ts')

    if player_events_tl.empty:
        st.warning("No events found for this player")
    else:
        ts_min = player_events_tl['ts'].min()
        ts_max = player_events_tl['ts'].max()
        player_events_tl['ts_sec'] = (
            (player_events_tl['ts'] - ts_min)
            .dt.total_seconds()
            .astype(int)
        )
        total_seconds   = max(int((ts_max - ts_min).total_seconds()), 1)
        player_label_tl = player_labels.get(selected_player, "Player")

        st.markdown(
            f"<div style='font-size:14px; color:#00aaff; font-weight:700; "
            f"margin-bottom:12px;'>Now watching: {player_label_tl} — "
            f"{match_labels.get(selected_match,'Match')}</div>",
            unsafe_allow_html=True
        )

        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 3])
        with ctrl1:
            play_btn  = st.button("▶  PLAY",  use_container_width=True)
        with ctrl2:
            reset_btn = st.button("⏮  RESET", use_container_width=True)
        with ctrl3:
            speed = st.selectbox(
                "Speed", options=[1, 2, 4],
                format_func=lambda x: f"{x}x",
                label_visibility="collapsed"
            )

        if 'tl_time' not in st.session_state:
            st.session_state.tl_time = 0
        if reset_btn:
            st.session_state.tl_time = 0
            st.rerun()

        current_time = st.slider(
            "Timeline", min_value=0,
            max_value=total_seconds,
            value=st.session_state.tl_time,
            step=1, label_visibility="collapsed"
        )

        pct = (current_time / total_seconds) * 100
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:16px;'>
            <span style='color:#00aaff; font-size:13px; font-weight:700;
                         min-width:45px;'>{fmt_time(current_time)}</span>
            <div style='flex:1; height:6px; background:rgba(255,255,255,0.1);
                        border-radius:3px; overflow:hidden;'>
                <div style='width:{pct:.1f}%; height:100%;
                            background:linear-gradient(90deg,#00aaff,#00ff88);
                            border-radius:3px;'></div>
            </div>
            <span style='color:#888; font-size:13px; min-width:45px;
                         text-align:right;'>{fmt_time(total_seconds)}</span>
        </div>
        <div style='display:flex; gap:24px; margin-bottom:12px;'>
            <span style='color:#888; font-size:12px;'>Speed:
                <span style='color:#00aaff; font-weight:700;'>{speed}x</span>
            </span>
            <span style='color:#888; font-size:12px;'>Events shown:
                <span style='color:#00ff88; font-weight:700;'>
                {len(player_events_tl[player_events_tl['ts_sec']<=current_time])}
                </span> / {len(player_events_tl)}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if play_btn:
            step   = max(1, speed * 2)
            ph_map = st.empty()

            for t in range(current_time, total_seconds + 1, step):
                st.session_state.tl_time = t
                snap     = player_events_tl[player_events_tl['ts_sec'] <= t]
                pos_snap = snap[snap['event'].isin(['Position','BotPosition'])]
                evt_snap = snap[~snap['event'].isin(['Position','BotPosition'])]

                fig_tl = get_minimap_figure(selected_map, height=600)

                if not pos_snap.empty:
                    coords_s = pos_snap.apply(
                        lambda r: world_to_pixel(r['x'], r['z'], selected_map),
                        axis=1
                    )
                    px = [c[0] for c in coords_s]
                    py = [c[1] for c in coords_s]
                    fig_tl.add_trace(go.Scatter(
                        x=px, y=py, mode="lines",
                        line=dict(color="#00aaff", width=3),
                        name="Path", hoverinfo="skip"
                    ))
                    fig_tl.add_trace(go.Scatter(
                        x=[px[-1]], y=[py[-1]], mode="markers",
                        marker=dict(size=20, color="#00ff88",
                                    line=dict(width=3, color="#ffffff")),
                        name=f"● {player_label_tl}"
                    ))

                fig_tl = add_event_markers(fig_tl, evt_snap, selected_map)
                fig_tl.add_annotation(
                    x=10, y=1010,
                    text=f"⏱ {fmt_time(t)}  |  {player_label_tl}  |  {len(evt_snap)} events",
                    showarrow=False,
                    font=dict(color="#ffffff", size=13),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="rgba(0,170,255,0.5)",
                    borderwidth=1, xanchor="left"
                )
                ph_map.plotly_chart(fig_tl, use_container_width=True)
                time.sleep(0.1 / speed)

            st.session_state.tl_time = 0

        else:
            if current_time > 0:
                snap     = player_events_tl[player_events_tl['ts_sec'] <= current_time]
                pos_snap = snap[snap['event'].isin(['Position','BotPosition'])]
                evt_snap = snap[~snap['event'].isin(['Position','BotPosition'])]

                fig_tl = get_minimap_figure(selected_map, height=600)

                if not pos_snap.empty:
                    coords_s = pos_snap.apply(
                        lambda r: world_to_pixel(r['x'], r['z'], selected_map),
                        axis=1
                    )
                    px = [c[0] for c in coords_s]
                    py = [c[1] for c in coords_s]
                    fig_tl.add_trace(go.Scatter(
                        x=px, y=py, mode="lines",
                        line=dict(color="#00aaff", width=3),
                        name="Path so far"
                    ))
                    fig_tl.add_trace(go.Scatter(
                        x=[px[-1]], y=[py[-1]], mode="markers",
                        marker=dict(size=20, color="#00ff88",
                                    line=dict(width=3, color="#ffffff")),
                        name=f"● {player_label_tl}"
                    ))

                fig_tl = add_event_markers(fig_tl, evt_snap, selected_map)
                fig_tl.add_annotation(
                    x=10, y=1010,
                    text=f"⏱ {fmt_time(current_time)}  |  "
                         f"{player_label_tl}  |  {len(evt_snap)} events",
                    showarrow=False,
                    font=dict(color="#ffffff", size=13),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="rgba(0,170,255,0.5)",
                    borderwidth=1, xanchor="left"
                )
                st.plotly_chart(fig_tl, use_container_width=True)

            else:
                st.markdown("""
                <div style='text-align:center; padding:60px 40px;
                            background:#111827; border-radius:12px;
                            border:1px solid rgba(0,170,255,0.2);
                            margin-top:16px;'>
                    <div style='font-size:48px; margin-bottom:16px;'>▶</div>
                    <div style='color:#00aaff; font-size:18px; font-weight:700;
                                letter-spacing:2px;'>
                        PRESS PLAY TO START PLAYBACK</div>
                    <div style='color:#555; font-size:13px; margin-top:10px;'>
                        or drag the slider above to scrub through the match
                    </div>
                </div>
                """, unsafe_allow_html=True)