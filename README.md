# Citi Bike functional neighbourhoods

The hand-in is `city_bike.ipynb`. Run it from the repository root. It asks which
NYC stations form communities in the trip graph and which stations share similar
arrival and departure patterns. Graph clustering and station-profile clustering
are the course methods. Directed Infomap is the method beyond the curriculum.

## Set up the notebook

To use Conda, run these commands from the repository root.

```bash
conda env create -f environment.yml
conda activate city-bike
python -m ipykernel install --user --name city-bike --display-name "Python (city-bike)"
jupyter lab city_bike.ipynb
```

Select **Python (city-bike)** as the notebook kernel.

Alternatively, use a virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab city_bike.ipynb
```

Select the Python kernel from this repository's `.venv`. Support code lives in
`src/citibike`. Keep it with the notebook when sharing a runnable copy.

## Prepare the data

The study window is September 2025 through August 2026. The data source is NYC
Citi Bike's [public trip-history archive](https://citibikenyc.com/system-data).
The notebook reads precomputed aggregates and never downloads trip archives
when run. To work from existing aggregates, put `stations.parquet`,
`edges.parquet`, and `profiles.parquet` under `data/processed/`. Include
`cleaning_report.csv` if available. Use files from the same preprocessing run.

To build everything from the public source (about 8.3 GB of downloads):

```bash
python -m citibike.download 2025-09 2026-08
python -m citibike.convert
python -m citibike.clean
python -m citibike.features
```

The download and conversion stages reuse complete files in `data/raw/` and
`data/parquet/`. The cleaning and feature stages rebuild outputs from those
inputs. Data and the virtual environment are ignored by Git. The data pipeline
and clustering helpers were adapted from the earlier
[CTDS project](https://github.com/Lyngsberg/CTDS_project).

## Current framework

The notebook runs a first comparison of directed Infomap, Louvain, spectral
clustering, and geographic k-means with a matched community count. Section 3
fits annual and period-specific Infomap communities on a shared station set,
with five seeds and five internal trials per seed. It reports station/trip
coverage, ARI agreement within and across periods, retention under annual and
period-specific boundaries, community counts, and geographically located station
changes. The support functions are in `src/citibike/temporal.py`.

Temporal tables are regenerated under `data/processed/temporal/` by Run All.
The comparison measures optimization variability and in-sample structure.
Sampling uncertainty and held-out temporal evaluation remain possible extensions.

Section 4 compares k-means with Ward clustering on normalized hourly net flows.
PCA provides a two-dimensional view of station profiles. The final hand-in still
needs the group discussion and individual contributions.
