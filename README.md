# Citi Bike functional neighbourhoods

The hand-in is `city_bike.ipynb`. Run it from the repository root. It asks which
NYC stations form communities in the trip graph and which stations share similar
arrival and departure patterns. Graph clustering and station-profile clustering
are the course methods. Directed Infomap is the method beyond the curriculum.

## Set up the notebook

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
clustering, and geographic k-means with a matched community count. It shows
retention by time of day for the full-year partition and compares k-means with
Ward clustering on normalized hourly net flows. PCA provides a two-dimensional
view of station profiles.

The final evaluation still needs stability checks across seeds and time periods,
and the notebook has a section for interpreting the results and recording group
contributions.
