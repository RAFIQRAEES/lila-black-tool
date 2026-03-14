# 🎮 LILA BLACK — Player Journey Visualization Tool

A browser-based gameplay telemetry visualization tool built for Level Designers to analyze player journeys, heatmaps, and match replays in LILA BLACK (extraction shooter).

**🔗 Live Demo:** https://lila-black-tool.streamlit.app/  
**📁 Repository:** https://github.com/RAFIQRAEES/lila-black-tool

---

## What It Does

The tool provides 4 levels of drill-down visualization:

| Level | View | Purpose |
|---|---|---|
| **Level 1** | Heatmap | Where do players go? KMeans hotspot zones across all matches |
| **Level 2** | Match Explorer | What happened in a single match? Event markers on minimap |
| **Level 3** | Player Journey | How did one player move? Full path trace with event overlays |
| **Level 4** | Timeline Playback | Watch a player's match unfold in real time at 1x/2x/4x speed |

---

## How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/RAFIQRAEES/lila-black-tool.git
cd lila-black-tool
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your data

Place your `player_data` folder in the project root:

```
lila-black-tool/
  ├── final_code.py
  ├── requirements.txt
  ├── README.md
  ├── ARCHITECTURE.md
  ├── INSIGHTS.md
  └── player_data/
        ├── minimaps/
        │     ├── AmbroseValley_Minimap.png
        │     ├── GrandRift_Minimap.png
        │     └── Lockdown_Minimap.jpg
        ├── February_10/
        ├── February_11/
        ├── February_12/
        ├── February_13/
        └── February_14/
```

### 5. Run the app

```bash
python -m streamlit run final_code.py
```

The app will open at `http://localhost:8501`

---

## Using the Upload Feature

If you don't have the `player_data` folder locally, you can upload a ZIP file directly in the app:

```
1. Open the app
2. At the top — click "Browse files"
3. Upload your player_data.zip
4. The ZIP should contain:
   → day folders (February_10, February_11, etc.)
   → minimaps/ folder with map images
5. All 4 levels will unlock automatically
```

