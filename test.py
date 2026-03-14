import pandas as pd
import pyarrow.parquet as pq
import os

# Step 1: Check if data folder exists
data_path = "player_data"
if os.path.exists(data_path):
    print("✅ player_data folder found!")
else:
    print("❌ player_data folder NOT found — check your folder structure")

# Step 2: Try reading one file from February_10
feb10_path = "player_data/February_10"
files = os.listdir(feb10_path)
print(f"✅ Found {len(files)} files in February_10")

# Step 3: Read the first file
first_file = os.path.join(feb10_path, files[0])
df = pq.read_table(first_file).to_pandas()
df['event'] = df['event'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

print(f"✅ Successfully read a file!")
print(f"✅ Rows in this file: {len(df)}")
print(f"✅ Columns: {list(df.columns)}")
print(f"\n--- Sample Data ---")
print(df.head(3))

# Step 4: Check minimaps
minimap_path = "player_data/minimaps"
minimaps = os.listdir(minimap_path)
print(f"\n✅ Minimap images found: {minimaps}")