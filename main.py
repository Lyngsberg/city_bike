import src.graph as graph
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

print("data indlæses!")
load = "2023-citibike-tripdata"
# Each monthly zip holds several CSVs; read only the columns needed for a first look.
cols = ["rideable_type", "started_at", "ended_at", "member_casual",
        "start_station_id", "end_station_id",
        "start_lat", "start_lng", "end_lat", "end_lng"]
dtypes = {"start_station_id": str, "end_station_id": str}

def read_zip(path):
    dataframes = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".csv") or name.startswith("__MACOSX"):
                continue
            with archive.open(name) as file:
                dataframes.append(pd.read_csv(file, usecols=cols, parse_dates=["started_at", "ended_at"], dtype=dtypes))
    return pd.concat(dataframes) 

# Loading the daataaaa
frames = []
for filename in sorted(os.listdir(load)):
    if filename.endswith(".zip"):
        frames.append(read_zip(os.path.join(load, filename)))
df = pd.concat(frames, ignore_index=True)

df["rideable_type"] = df["rideable_type"].astype("category")
df["member_casual"] = df["member_casual"].astype("category")
df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

# Vi vil kun have en x,y for hver station
starts = df[["start_station_id", "start_lng", "start_lat"]].set_axis(["id", "x", "y"], axis=1)
ends   = df[["end_station_id",   "end_lng",   "end_lat"]].set_axis(["id", "x", "y"], axis=1)
pos_df = pd.concat([starts, ends]).dropna().groupby("id")[["x", "y"]].median()
pos = dict(zip(pos_df.index, zip(pos_df.x, pos_df.y)))

print("Her bygges grafer!")
trip_counts = df.groupby(["start_station_id", "end_station_id"], sort=False).size()
station_graph = graph.Graph()
for (start, end), n in trip_counts.items():
    station_graph.add_edge(start, end, int(n))

print(f"Graph has {station_graph.number_of_nodes()} nodes and {station_graph.number_of_edges()} edges.")

edges = [(start, end, n) for start, neighbours in station_graph.adj.items() for end, n in neighbours.items()
         if start != end and start in pos and end in pos]

top = sorted(edges, key=lambda t: t[2], reverse=True)[:10000]

segments = [(pos[start], pos[end]) for start, end, _ in top]
max_n = top[0][2]
widths = 2*[n / max_n for _, _, n in top]
print("Der plotttesssss")
fig, ax = plt.subplots(figsize=(8, 10))
ax.add_collection(LineCollection(segments, linewidths=widths, alpha=0.3))
ax.scatter(pos_df.x, pos_df.y, s=2, c="black")  
plt.show()
