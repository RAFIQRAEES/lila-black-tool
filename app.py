import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import os
from PIL import Image

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

@st.cache_data
def load_all_data():
    all_frames = []
    for day_name, folder_path in DATA_FOLDERS.items():
        if not os.path.exists(folder_path):
            continue
        files = os.listdir(folder_path)
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            try:
                df = pq.read_table(filepath).to_pandas()
                df['event'] = df['event'].apply(
                    lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                )
                df['day'] = day_name
                df['is_bot'] = df['user_id'].apply(lambda x: str(x).isdigit())
                df['match_id'] = df['match_id'].apply(
                    lambda x: x.replace('.nakama-0', '') if isinstance(x, str) else x
                )
                all_frames.append(df)
            except Exception:
                continue
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

st.set_page_config(
    page_title="LILA BLACK — Player Journey Tool",
    page_icon="🎮",
    layout="wide"
)

# ── Dark theme styling ──
st.markdown("""
<style>
/* Full black background */
.stApp {
    background-color: #0a0a0f;
    color: #ffffff;
}

/* Sidebar and containers */
section[data-testid="stSidebar"] {
    background-color: #0d0d14;
}

/* Main title */
h1 {
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    letter-spacing: 2px !important;
    color: #ffffff !important;
    text-transform: uppercase;
}

/* All headings */
h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
}

/* Paragraph and labels */
p, label, .stMarkdown {
    color: #cccccc !important;
    font-size: 15px !important;
}

/* Metric boxes */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #00ff8844;
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 0 20px #00ff8811;
}

[data-testid="stMetricLabel"] {
    color: #aaaaaa !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="stMetricValue"] {
    color: #00ff88 !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
}

/* Arrow buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
    color: #00ff88 !important;
    border: 2px solid #00ff8866 !important;
    border-radius: 10px !important;
    font-size: 1.5rem !important;
    font-weight: 900 !important;
    height: 60px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 15px #00ff8822 !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00ff8822, #00ff8811) !important;
    border-color: #00ff88 !important;
    box-shadow: 0 0 25px #00ff8855 !important;
    transform: scale(1.05);
}

/* Dataframe table */
[data-testid="stDataFrame"] {
    border: 1px solid #00ff8833 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Divider */
hr {
    border-color: #ffffff11 !important;
}

/* Success message */
[data-testid="stAlert"] {
    background-color: #00ff8811 !important;
    border: 1px solid #00ff8844 !important;
    color: #00ff88 !important;
    border-radius: 10px !important;
}

/* Spinner */
.stSpinner {
    color: #00ff88 !important;
}

/* Image border */
img {
    border-radius: 12px !important;
    border: 2px solid #00ff8833 !important;
    box-shadow: 0 0 30px #00000088 !important;
}

/* Hide streamlit default menu & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <div style='font-size:13px; letter-spacing:6px; color:#00ff88; font-weight:700; margin-bottom:8px;'>
        LILA GAMES — INTERNAL TOOL
    </div>
    <h1 style='font-size:2.8rem; font-weight:900; color:#ffffff; letter-spacing:3px; margin:0;'>
        🎮 LILA BLACK
    </h1>
    <div style='font-size:16px; color:#888; margin-top:8px; letter-spacing:2px;'>
        PLAYER JOURNEY VISUALIZATION
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

with st.spinner("Loading gameplay data..."):
    df = load_all_data()

if not df.empty:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Top stats ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", f"{len(df):,}")
    col2.metric("Unique Players", f"{df[df['is_bot']==False]['user_id'].nunique():,}")
    col3.metric("Unique Matches", f"{df['match_id'].nunique():,}")
    col4.metric("Maps Available", f"{df['map_id'].nunique()}")

    st.divider()

    # ── Map index ──
    if "map_index" not in st.session_state:
        st.session_state.map_index = 0

    # ── Section header ──
    st.markdown("""
    <div style='font-size:12px; letter-spacing:4px; color:#00ff88; font-weight:700; margin-bottom:4px;'>
        MAP EXPLORER
    </div>
    """, unsafe_allow_html=True)

    # ── Arrow navigation ──
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
            f"<h2 style='text-align:center; color:#ffffff; font-weight:900; letter-spacing:3px; margin-bottom:4px;'>{selected_map.upper()}</h2>",
            unsafe_allow_html=True
        )
        dots = "".join(["<span style='color:#00ff88; font-size:22px;'>●</span> " if i == st.session_state.map_index else "<span style='color:#333; font-size:22px;'>●</span> " for i in range(len(MAPS))])
        st.markdown(f"<p style='text-align:center; margin-top:0'>{dots}</p>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Map image + stats ──
    map_col, info_col = st.columns([2, 1])

    with map_col:
        minimap_path = MINIMAP_PATHS[selected_map]
        if os.path.exists(minimap_path):
            img = Image.open(minimap_path)
            st.image(img, use_container_width=True)

    with info_col:
        map_data = df[df['map_id'] == selected_map]

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                    border: 1px solid #00ff8833;
                    border-radius: 14px;
                    padding: 20px;
                    margin-bottom: 16px;'>
            <div style='font-size:11px; letter-spacing:3px; color:#00ff88; font-weight:700; margin-bottom:12px;'>
                MAP STATISTICS
            </div>
            <div style='font-size:28px; font-weight:900; color:#ffffff;'>{selected_map}</div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Total Events", f"{len(map_data):,}")
        st.metric("Human Players", f"{map_data[map_data['is_bot']==False]['user_id'].nunique():,}")
        st.metric("Matches Played", f"{map_data['match_id'].nunique():,}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:11px; letter-spacing:3px; color:#00ff88; font-weight:700; margin-bottom:8px;'>
            EVENT BREAKDOWN
        </div>
        """, unsafe_allow_html=True)

        event_counts = map_data['event'].value_counts().reset_index()
        event_counts.columns = ['Event', 'Count']

        # Color code events
        def color_event(val):
            colors = {
                'Kill': '#ff4444',
                'Killed': '#ff8844',
                'BotKill': '#ff6644',
                'BotKilled': '#ffaa44',
                'KilledByStorm': '#aa44ff',
                'Loot': '#44aaff',
                'Position': '#00ff88',
                'BotPosition': '#00cc66',
            }
            return colors.get(val, '#ffffff')

        event_counts['Color'] = event_counts['Event'].apply(color_event)

        for _, row in event_counts.iterrows():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;
                        background:#111827; border-left: 3px solid {row["Color"]};
                        border-radius:6px; padding:8px 12px; margin-bottom:6px;'>
                <span style='color:#cccccc; font-size:13px; font-weight:600;'>{row["Event"]}</span>
                <span style='color:{row["Color"]}; font-size:15px; font-weight:900;'>{row["Count"]:,}</span>
            </div>
            """, unsafe_allow_html=True)

else:
    st.error("❌ No data loaded — check your player_data folder")