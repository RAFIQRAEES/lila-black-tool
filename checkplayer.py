import pandas as pd
import pyarrow.parquet as pq
import os

all_frames = []

folders = {
    "February_10": "player_data/February_10",
    "February_11": "player_data/February_11",
    "February_12": "player_data/February_12",
    "February_13": "player_data/February_13",
    "February_14": "player_data/February_14",
}

for day_name, folder in folders.items():
    if not os.path.exists(folder):
        continue
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            df = pq.read_table(filepath).to_pandas()
            df['event'] = df['event'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
            )
            df['is_bot'] = df['user_id'].apply(lambda x: str(x).isdigit())
            df['match_id'] = df['match_id'].apply(
                lambda x: x.replace('.nakama-0', '') if isinstance(x, str) else x
            )
            df['day'] = day_name
            all_frames.append(df)
        except Exception as e:
            continue

df = pd.concat(all_frames, ignore_index=True)

print("=" * 50)
print("TOTAL PLAYERS BREAKDOWN")
print("=" * 50)
total_humans = df[df['is_bot']==False]['user_id'].nunique()
total_bots = df[df['is_bot']==True]['user_id'].nunique()
print(f"Total Unique Humans : {total_humans}")
print(f"Total Unique Bots   : {total_bots}")
print(f"Total Combined      : {total_humans + total_bots}")

print("\n" + "=" * 50)
print("HUMANS PER MATCH BREAKDOWN")
print("=" * 50)
humans = df[df['is_bot']==False]
humans_per_match = humans.groupby('match_id')['user_id'].nunique()
print(f"Average humans per match : {humans_per_match.mean():.1f}")
print(f"Max humans in one match  : {humans_per_match.max()}")
print(f"Min humans in one match  : {humans_per_match.min()}")
print(f"\nDistribution:")
print(humans_per_match.value_counts().sort_index())

print("\n" + "=" * 50)
print("BOTS PER MATCH BREAKDOWN")
print("=" * 50)
bots = df[df['is_bot']==True]
bots_per_match = bots.groupby('match_id')['user_id'].nunique()
print(f"Average bots per match : {bots_per_match.mean():.1f}")
print(f"Max bots in one match  : {bots_per_match.max()}")
print(f"Min bots in one match  : {bots_per_match.min()}")