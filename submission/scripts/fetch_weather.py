"""Fetch the project's weather covariate from the project dataset on Hugging Face.

The sprint scenario allows the project results and the project's weather data;
this team's weather covariate is the Open-Meteo hourly file collected for the
COMP3522-HKO project and published at ``NotASI/COMP3522-HKO``. This script
downloads that file once into ``data/external/weather/`` and writes a
territory-hourly mean series (one row per observation hour, Hong Kong clock) to
``data/external/weather/om_territory_hourly.csv`` so notebooks 03 and 04 can
join it to the AQHI frame on ``obs_time_hkt``.

Usage (from the repo root, inside .venv)::

    .venv\\Scripts\\python.exe scripts\\fetch_weather.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

DATASET = "NotASI/COMP3522-HKO"
REVISION = "main"
REMOTE_PATH = "open_meteo/om_historical_weather_hourly.parquet"
RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{REMOTE_PATH}"
WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "surface_pressure",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external" / "weather"
PARQUET = EXTERNAL / "om_historical_weather_hourly.parquet"
TERRITORY = EXTERNAL / "om_territory_hourly.csv"
SOURCE_NOTE = EXTERNAL / "om_territory_hourly.source.json"


def download() -> pathlib.Path:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if PARQUET.exists() and PARQUET.stat().st_size > 0:
        print("already cached:", PARQUET)
        return PARQUET
    print("downloading", RESOLVE)
    request = urllib.request.Request(RESOLVE, headers={"User-Agent": "comp3522-sprint"})
    with urllib.request.urlopen(request, timeout=300) as response, PARQUET.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    print("wrote", PARQUET, PARQUET.stat().st_size, "bytes")
    return PARQUET


def territory_hourly() -> int:
    import pandas as pd

    frame = pd.read_parquet(PARQUET)
    columns = [name for name in frame.columns]
    time_column = "time_hkt" if "time_hkt" in columns else "time"
    stamps = pd.to_datetime(frame[time_column], errors="coerce")
    if isinstance(stamps.dtype, pd.DatetimeTZDtype):
        stamps = stamps.dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
    else:
        stamps = stamps.dt.tz_localize("Asia/Hong_Kong").dt.tz_localize(None)
    frame = frame.assign(obs_time_hkt=stamps)
    present = [name for name in WEATHER_COLUMNS if name in frame.columns]
    territory = (
        frame.groupby("obs_time_hkt")[present].mean().reset_index().sort_values("obs_time_hkt")
    )
    territory.to_csv(TERRITORY, index=False)
    payload = {
        "source_dataset": DATASET,
        "remote_path": REMOTE_PATH,
        "resolve_url": RESOLVE,
        "local_parquet": str(PARQUET.relative_to(ROOT)).replace("\\", "/"),
        "rows_raw": int(len(frame)),
        "rows_territory_hourly": int(len(territory)),
        "columns": ["obs_time_hkt"] + present,
        "timezone": "Asia/Hong_Kong",
        "note": "Mean over the project's Open-Meteo station points per observation hour.",
    }
    SOURCE_NOTE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", TERRITORY, territory.shape)
    return 0


def main() -> int:
    download()
    return territory_hourly()


if __name__ == "__main__":
    sys.exit(main())