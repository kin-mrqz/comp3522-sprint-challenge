# data/processed - AQHI analysis tables

Built by `EDA/code/00_build_processed.py` from `data/raw/aqhi_data.csv` (unchanged EPD export).
Rebuild:

```bash
.venv/bin/python EDA/code/00_build_processed.py
```

## Files

| File | Grain | Rows | Key columns |
|---|---|---|---|
| `aqhi_hourly.parquet` | timestamp x station | 850,608 | `timestamp`, `station`, `aqhi`, `substituted`, `censored`, `region`, `station_type` |
| `aqhi_territory_hourly.parquet` | timestamp | 47,256 | `mean_station_mean`, `max_any_station`, `n_stations`, `n_stations_ge7` |
| `aqhi_daily_station.parquet` | date x station | 35,101 | `mean`, `max`, `n_obs`, `n_ge7`, `peak_hour`, `daily_max_file` |
| `aqhi_daily_territory.parquet` | date (full calendar) | 2,007 | `mean_station_mean`, `max_station_max`, `n_stations_ge7`, `spatial_range`, `in_file` |
| `aqhi_wide_hourly.parquet` | timestamp | 47,256 | one column per station |
| `station_metadata.csv` | station | 18 | `region`, `station_type`, `is_background` |
| `data_quality_station_year.csv` | station x year | 108 | `missing_pct`, `substituted_pct`, `censored_pct` |
| `cleaning_log.csv` | rule | 8 | `rule`, `rows_affected`, `why` |
| `t00_file_audit.csv` | metric | 18 | key-value audit |

## Conventions

- **Timezone**: dates and hours are Hong Kong local time. The export carries no timezone field.
- **Hour label**: `Hour` 1..24 maps to `hour_of_day` 0..23, i.e. hour 1 is 00:00-00:59 (hour beginning).
- **Daily Max rows** in the raw file are excluded from the hourly table. They equal the maximum of the 24 hourly values in every checked station-day (share = 1.000000), so keeping them would double-weight each day.
- **`10+`** is stored as `aqhi = 10` with `censored = True`; 10 is a lower bound, not a measurement. 304 cells.
- **`*`** marks a value borrowed from the most similar station (`substituted = True`). 2.69% of cells. Values are kept and flagged.
- **`-` and `*` alone** are missing (`aqhi = NaN`). Nothing is imputed.
- **Calendar gaps are kept.** `aqhi_daily_territory.parquet` has one row per date; `in_file = False` marks dates absent from the export or dates present but blank.
- **AQHI bands** (EPD): Low 1-3, Moderate 4-6, High 7, Very High 8-10, Serious 10+.
- **Station type** (EPD): 15 general stations, 3 roadside (Causeway Bay, Central, Mong Kok). Tap Mun is the background station.

## Known gaps

- 38 calendar dates absent: 2023-08-25..31 and 2024-12-01..31.
- 17 dates present but blank for all 18 stations: 2023-01-07, 2023-06-27, 2023-10-01..15.
- Tap Mun additionally blank 2023-09-07..25 and all of 2023-10.
- EPD updated the AQHI compilation on 2025-03-22 (ozone risk factor). The series before and after that date is not strictly one instrument.
