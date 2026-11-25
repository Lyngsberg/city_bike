"""Shared paths for local data excluded from Git."""

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
RAW = DATA / "raw"
PARQUET = DATA / "parquet"  # Typed trips before cleaning.
PROCESSED = DATA / "processed"

S3_BUCKET = "https://s3.amazonaws.com/tripdata"

TRIP_COLUMNS = {
    "ride_id": "VARCHAR",
    "rideable_type": "VARCHAR",
    "started_at": "TIMESTAMP",  # NYC local time without a time zone.
    "ended_at": "TIMESTAMP",
    "start_station_name": "VARCHAR",
    "start_station_id": "VARCHAR",  # Keep IDs as text to preserve their format.
    "end_station_name": "VARCHAR",
    "end_station_id": "VARCHAR",
    "start_lat": "DOUBLE",
    "start_lng": "DOUBLE",
    "end_lat": "DOUBLE",
    "end_lng": "DOUBLE",
    "member_casual": "VARCHAR",
}
