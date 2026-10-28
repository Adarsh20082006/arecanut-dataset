# Arecanut irrigation public-data collection pipeline

## What it retrieves

- Open-Meteo Historical Weather API (ERA5-Land): historical weather and gridded/reanalysis soil moisture.
- Open-Meteo Forecast API: current conditions plus up to several days of forecast data.
- Open-Meteo Historical Forecast API: archived forecast time series where available (coverage varies; approximately 2022 onward).
- NASA POWER Daily API: separate gridded daily weather and radiation estimates.
- ISRIC SoilGrids REST API: static mapped soil properties by soil depth.

The script is intentionally honest about the limits of public data: model/reanalysis soil moisture is not measured farm soil moisture, gridded rainfall is not a farm rain gauge, and having an irrigation system does not prove that irrigation was applied on a given date.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Edit `farms.csv` first. Replace the demo coordinates (12.34, 76.10) with the actual coordinate of each plantation/plot. Add one row per farm. Do not use the demo row as a real farm location.

Required columns: `farm_id`, `latitude`, `longitude`. Optional columns are `irrigation_method`, `farm_has_irrigation_system`, `irrigated_area_ha`, `palm_age_years`, `palm_count`, `soil_texture_field`.

## Run

```bash
python collect_arecanut_data.py --farms farms.csv --start-date 2020-01-01
```

By default, historical collection ends three days before today because near-real-time reanalysis can be delayed. Current conditions and the near-term forecast are fetched separately. To specify an end date:

```bash
python collect_arecanut_data.py --farms farms.csv --start-date 2020-01-01 --end-date 2025-12-31
```

The default `--end-date` is preferred for up-to-date collection. Passing a future end date is not useful for the historical archive.

Optional synthetic scenarios (soil-moisture state, irrigated/not, rain/no rain and forecast-rain variations):

```bash
python collect_arecanut_data.py --farms farms.csv --start-date 2020-01-01 --make-scenarios
```

Scenario values are counterfactual software/model test cases. They are NOT actual sensor observations, actual rainfall, actual irrigation, or irrigation-target labels. Keep `arecanut_scenarios.csv` separate from the factual/modelled farm-day file during training.

If a source is unavailable, the script logs the failure in `data/processed/source_status.csv` and continues with the other sources where possible. Options:

```bash
python collect_arecanut_data.py --skip-soilgrids
python collect_arecanut_data.py --skip-nasa
python collect_arecanut_data.py --skip-historical-forecast
```

## Optional field data inputs

Put these files in the same folder as `farms.csv` if available.

### `soil_moisture_observations.csv`

Required keys: `farm_id,date`. Add columns such as `soil_moisture_20cm_vwc`, `soil_moisture_40cm_vwc`, sensor depth, sensor ID and calibration/quality flags. Use units of m3/m3 for VWC columns. These values must come from farm sensors or documented manual measurement, not be copied from gridded estimates.

### `irrigation_logs.csv`

Required columns: `farm_id,date,irrigation_applied_liters`. Record the measured or carefully logged irrigation volume for that farm/day. Multiple entries per farm/date are summed. If the volume is only estimated, document that limitation separately.

## Outputs

- `data/processed/arecanut_farm_day.csv`: merged farm-day file; sources have distinct columns, so NASA POWER isn't silently blended with Open-Meteo.
- `data/processed/soilgrids_static.csv`: SoilGrids values, depths, source raw means, scale factors and units.
- `data/processed/source_status.csv`: source status per farm.
- `data/processed/data_dictionary.csv`: short data dictionary and unit reminders.
- `data/raw/*.json`: raw API responses for reproducibility.

## Interpretation and ML safety

- `soil_moisture_reanalysis_*_vwc` columns are gridded/modelled soil moisture, not farm sensor readings.
- `historical_precipitation_reanalysis_mm` and `nasa_precipitation_mm_day` are separate gridded estimates, not observed rain-gauge data.
- `forecast_*` fields are snapshots collected at run time. For a backtest, use an archived forecast whose issue time is known and was available before the simulated irrigation decision. The historical forecast API's stitched time series is not, by itself, an exact decision-time replay.
- `farm_has_irrigation_system` means a system is available; it does not mean that the farm was irrigated that day.
- The script does not output an `irrigation_target_liters` label. Generate that only from a documented water-balance method, expert-approved schedule, or reviewed field observations.
- Avoid random row-level train/test splits. Use time-based validation and, where possible, hold out entire farms.

## Official source documentation

- Open-Meteo historical weather: https://open-meteo.com/en/docs/historical-weather-api
- Open-Meteo forecast: https://open-meteo.com/en/docs
- Open-Meteo historical forecast: https://open-meteo.com/en/docs/historical-forecast-api
- NASA POWER Daily API: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- SoilGrids REST API: https://rest.isric.org/soilgrids/v2.0/docs
