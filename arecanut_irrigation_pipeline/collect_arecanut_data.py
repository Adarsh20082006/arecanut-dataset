#!/usr/bin/env python3
"""Collect public weather/soil data for an arecanut irrigation research dataset.

Sources
-------
1. Open-Meteo Historical Weather API: long-term weather reanalysis + model soil moisture.
2. Open-Meteo Forecast API: current conditions and near-term forecasts.
3. Open-Meteo Historical Forecast API: archived forecast time series (coverage varies; ~2022+).
4. NASA POWER Daily API: independent gridded meteorological / solar data.
5. ISRIC SoilGrids REST API: static mapped soil-property estimates (not field tests).

Outputs
-------
- data/processed/arecanut_farm_day.csv: one row per farm/date, factual/modelled
  weather fields kept separate by source, static farm/soil-map fields, and optional
  observed sensor/irrigation values if input logs are supplied.
- data/processed/soilgrids_static.csv: soil-map values with source scale factor/units.
- data/processed/source_status.csv: per-farm source success/failure information.
- data/processed/arecanut_scenarios.csv: optional synthetic scenario grid generated
  only when --make-scenarios is supplied. Scenario values are explicitly marked
  synthetic and must NOT be used as observed labels.
- data/raw/*.json: raw JSON responses to support reproducibility/auditing.

Important
---------
Public gridded soil moisture is modelled/reanalysis data, not a field sensor.
Public rainfall grids are not an observation from a rain gauge on the farm.
The script DOES NOT generate validated irrigation-target labels. No irrigation
recommendation should be inferred solely from synthetic scenario rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOCAL_TZ = ZoneInfo("Asia/Kolkata")
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
NASA_POWER_DAILY = "https://power.larc.nasa.gov/api/temporal/daily/point"
SOILGRIDS_PROPERTIES = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# These parameters are intentionally declared here so changes can be reviewed.
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
]
ARCHIVE_HOURLY_VARS = [
    "relative_humidity_2m",
    "wind_speed_10m",
    "vapour_pressure_deficit",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
]
FORECAST_DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_probability_max",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
]
HIST_FORECAST_HOURLY_VARS = [
    "precipitation",
    "precipitation_probability",
    "et0_fao_evapotranspiration",
]
NASA_PARAMETERS = [
    "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN", "WS10M",
]
SOILGRIDS_PROPERTIES_LIST = ["bdod", "clay", "sand", "silt", "soc", "phh2o"]
SOILGRIDS_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]


class DataCollectionError(RuntimeError):
    """Raised when a source response cannot be used safely."""


def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "ArecanutIrrigationResearch/1.0 (academic data collection)"})
    return session


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def api_json(session: requests.Session, url: str, params: Any, timeout: int = 90) -> dict[str, Any]:
    response = session.get(url, params=params, timeout=timeout)
    if not response.ok:
        excerpt = response.text[:500].replace("\n", " ")
        raise DataCollectionError(f"HTTP {response.status_code} from {response.url}: {excerpt}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise DataCollectionError(f"Non-JSON response from {response.url}") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise DataCollectionError(f"API returned error from {response.url}: {payload.get('reason', payload['error'])}")
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def date_series_to_strings(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def safe_col_name(text: str) -> str:
    text = text.lower().replace("-", "_").replace("cm", "cm")
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_")


def read_farms(path: Path) -> pd.DataFrame:
    if not path.exists():
        template = pd.DataFrame([
            {
                "farm_id": "REPLACE_WITH_FARM_ID",
                "latitude": 12.34,
                "longitude": 76.10,
                "irrigation_method": "drip",
                "farm_has_irrigation_system": True,
                "irrigated_area_ha": 1.0,
                "palm_age_years": np.nan,
                "palm_count": np.nan,
                "soil_texture_field": "unknown",
            }
        ])
        template.to_csv(path, index=False)
        raise DataCollectionError(
            f"Created template {path}. Edit it with each farm's actual plot-centre coordinates "
            "and metadata, then run the script again. The included coordinates are illustrative."
        )

    farms = pd.read_csv(path)
    required = {"farm_id", "latitude", "longitude"}
    missing = required - set(farms.columns)
    if missing:
        raise DataCollectionError(f"{path} is missing required columns: {sorted(missing)}")
    if farms.empty:
        raise DataCollectionError(f"{path} contains no farms")
    if farms["farm_id"].astype(str).duplicated().any():
        raise DataCollectionError("farm_id values must be unique in farms.csv")
    farms["farm_id"] = farms["farm_id"].astype(str)
    farms["latitude"] = pd.to_numeric(farms["latitude"], errors="coerce")
    farms["longitude"] = pd.to_numeric(farms["longitude"], errors="coerce")
    if farms["latitude"].isna().any() or not farms["latitude"].between(-90, 90).all():
        raise DataCollectionError("Latitude must be numeric and between -90 and 90")
    if farms["longitude"].isna().any() or not farms["longitude"].between(-180, 180).all():
        raise DataCollectionError("Longitude must be numeric and between -180 and 180")

    defaults = {
        "irrigation_method": "unknown",
        "farm_has_irrigation_system": np.nan,
        "irrigated_area_ha": np.nan,
        "palm_age_years": np.nan,
        "palm_count": np.nan,
        "soil_texture_field": "unknown",
    }
    for column, default in defaults.items():
        if column not in farms.columns:
            farms[column] = default
    return farms


def openmeteo_daily_and_hourly(
    session: requests.Session,
    farm: pd.Series,
    start_date: str,
    end_date: str,
    raw_dir: Path,
) -> pd.DataFrame:
    params = {
        "latitude": float(farm.latitude),
        "longitude": float(farm.longitude),
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(ARCHIVE_HOURLY_VARS),
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
        "models": "era5_land",
    }
    payload = api_json(session, OPEN_METEO_ARCHIVE, params)
    save_json(raw_dir / f"{farm.farm_id}_openmeteo_archive.json", payload)

    if not payload.get("daily", {}).get("time"):
        raise DataCollectionError("Open-Meteo archive returned no daily time series")
    daily = pd.DataFrame(payload["daily"])
    daily["date"] = date_series_to_strings(daily.pop("time"))
    daily = daily.rename(columns={
        "temperature_2m_mean": "historical_temperature_mean_c",
        "temperature_2m_max": "historical_temperature_max_c",
        "temperature_2m_min": "historical_temperature_min_c",
        "precipitation_sum": "historical_precipitation_reanalysis_mm",
        "rain_sum": "historical_rain_reanalysis_mm",
        "et0_fao_evapotranspiration": "historical_et0_mm_day",
        "shortwave_radiation_sum": "historical_shortwave_radiation_mj_m2_day",
        "wind_speed_10m_max": "historical_wind_speed_max_m_s",
    })

    hourly_payload = payload.get("hourly", {})
    if hourly_payload.get("time"):
        hourly = pd.DataFrame(hourly_payload)
        hourly["date"] = date_series_to_strings(hourly.pop("time"))
        aggregate = {
            "relative_humidity_2m": "mean",
            "wind_speed_10m": "mean",
            "vapour_pressure_deficit": "mean",
            "soil_moisture_0_to_7cm": "mean",
            "soil_moisture_7_to_28cm": "mean",
            "soil_moisture_28_to_100cm": "mean",
            "soil_temperature_0_to_7cm": "mean",
            "soil_temperature_7_to_28cm": "mean",
            "soil_temperature_28_to_100cm": "mean",
        }
        available = {k: v for k, v in aggregate.items() if k in hourly.columns}
        hourly_daily = hourly.groupby("date", as_index=False).agg(available)
        hourly_daily = hourly_daily.rename(columns={
            "relative_humidity_2m": "historical_relative_humidity_mean_pct",
            "wind_speed_10m": "historical_wind_speed_mean_m_s",
            "vapour_pressure_deficit": "historical_vpd_mean_kpa",
            "soil_moisture_0_to_7cm": "soil_moisture_reanalysis_0_7cm_vwc",
            "soil_moisture_7_to_28cm": "soil_moisture_reanalysis_7_28cm_vwc",
            "soil_moisture_28_to_100cm": "soil_moisture_reanalysis_28_100cm_vwc",
            "soil_temperature_0_to_7cm": "soil_temperature_reanalysis_0_7cm_c",
            "soil_temperature_7_to_28cm": "soil_temperature_reanalysis_7_28cm_c",
            "soil_temperature_28_to_100cm": "soil_temperature_reanalysis_28_100cm_c",
        })
        daily = daily.merge(hourly_daily, on="date", how="left")

    daily.insert(0, "farm_id", farm.farm_id)
    daily["weather_history_source"] = "Open-Meteo Historical Weather API / ERA5-Land"
    daily["weather_history_type"] = "reanalysis_grid_not_farm_station_observation"
    daily["archive_downloaded_at_utc"] = utc_now_iso()
    return daily


def nasa_power_daily(
    session: requests.Session,
    farm: pd.Series,
    start_date: str,
    end_date: str,
    raw_dir: Path,
) -> pd.DataFrame:
    params = {
        "parameters": ",".join(NASA_PARAMETERS),
        "community": "AG",
        "longitude": float(farm.longitude),
        "latitude": float(farm.latitude),
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
        "time-standard": "UTC",
    }
    payload = api_json(session, NASA_POWER_DAILY, params)
    save_json(raw_dir / f"{farm.farm_id}_nasa_power.json", payload)
    parameters = payload.get("properties", {}).get("parameter", {})
    if not parameters:
        raise DataCollectionError("NASA POWER response has no properties.parameter data")
    data = pd.DataFrame(parameters)
    data.index.name = "date_raw"
    data = data.reset_index().rename(columns={"date_raw": "date"})
    data["date"] = pd.to_datetime(data["date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    data = data.rename(columns={
        "T2M": "nasa_temperature_mean_c",
        "T2M_MAX": "nasa_temperature_max_c",
        "T2M_MIN": "nasa_temperature_min_c",
        "RH2M": "nasa_relative_humidity_mean_pct",
        "PRECTOTCORR": "nasa_precipitation_mm_day",
        "ALLSKY_SFC_SW_DWN": "nasa_shortwave_radiation_kwh_m2_day",
        "WS10M": "nasa_wind_speed_m_s",
    })
    data = data.replace(-999, np.nan)
    data.insert(0, "farm_id", farm.farm_id)
    data["nasa_weather_source"] = "NASA POWER Daily API"
    data["nasa_weather_type"] = "gridded_estimate_not_farm_station_observation"
    data["nasa_downloaded_at_utc"] = utc_now_iso()
    return data


def openmeteo_current_forecast(
    session: requests.Session,
    farm: pd.Series,
    raw_dir: Path,
    forecast_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = {
        "latitude": float(farm.latitude),
        "longitude": float(farm.longitude),
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m",
        "daily": ",".join(FORECAST_DAILY_VARS),
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
    }
    payload = api_json(session, OPEN_METEO_FORECAST, params)
    save_json(raw_dir / f"{farm.farm_id}_openmeteo_current_forecast.json", payload)
    retrieved = utc_now_iso()

    current = payload.get("current", {})
    if current:
        current_date = pd.to_datetime(current.get("time"), errors="coerce")
        current_row = {
            "farm_id": farm.farm_id,
            "date": current_date.strftime("%Y-%m-%d") if not pd.isna(current_date) else None,
            "current_conditions_time_local": current.get("time"),
            "current_temperature_c": current.get("temperature_2m"),
            "current_relative_humidity_pct": current.get("relative_humidity_2m"),
            "current_precipitation_mm_last_hour": current.get("precipitation"),
            "current_rain_mm_last_hour": current.get("rain"),
            "current_wind_speed_m_s": current.get("wind_speed_10m"),
            "current_snapshot_retrieved_at_utc": retrieved,
            "current_weather_source": "Open-Meteo Forecast API current conditions",
        }
        current_df = pd.DataFrame([current_row])
    else:
        current_df = pd.DataFrame(columns=["farm_id", "date"])

    daily_payload = payload.get("daily", {})
    if not daily_payload.get("time"):
        raise DataCollectionError("Open-Meteo forecast returned no daily forecast series")
    forecast = pd.DataFrame(daily_payload)
    forecast["date"] = date_series_to_strings(forecast.pop("time"))
    forecast = forecast.rename(columns={
        "temperature_2m_mean": "forecast_temperature_mean_c",
        "temperature_2m_max": "forecast_temperature_max_c",
        "temperature_2m_min": "forecast_temperature_min_c",
        "precipitation_sum": "forecast_precipitation_sum_mm",
        "rain_sum": "forecast_rain_sum_mm",
        "precipitation_probability_max": "forecast_precipitation_probability_max_pct",
        "et0_fao_evapotranspiration": "forecast_et0_mm_day",
        "shortwave_radiation_sum": "forecast_shortwave_radiation_mj_m2_day",
        "wind_speed_10m_max": "forecast_wind_speed_max_m_s",
    })
    forecast.insert(0, "farm_id", farm.farm_id)
    forecast["forecast_snapshot_retrieved_at_utc"] = retrieved
    forecast["forecast_source"] = "Open-Meteo Forecast API"
    forecast["forecast_is_archived"] = False
    return current_df, forecast


def openmeteo_historical_forecast(
    session: requests.Session,
    farm: pd.Series,
    start_date: str,
    end_date: str,
    raw_dir: Path,
) -> pd.DataFrame:
    if start_date > end_date:
        return pd.DataFrame(columns=["farm_id", "date"])
    params = {
        "latitude": float(farm.latitude),
        "longitude": float(farm.longitude),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HIST_FORECAST_HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }
    payload = api_json(session, OPEN_METEO_HIST_FORECAST, params, timeout=120)
    save_json(raw_dir / f"{farm.farm_id}_openmeteo_historical_forecast.json", payload)
    hourly_payload = payload.get("hourly", {})
    if not hourly_payload.get("time"):
        logging.warning("Historical forecast API returned no hourly data for %s", farm.farm_id)
        return pd.DataFrame(columns=["farm_id", "date"])
    hourly = pd.DataFrame(hourly_payload)
    hourly["date"] = date_series_to_strings(hourly.pop("time"))
    agg_spec: dict[str, str] = {}
    if "precipitation" in hourly:
        agg_spec["precipitation"] = "sum"
    if "precipitation_probability" in hourly:
        agg_spec["precipitation_probability"] = "max"
    if "et0_fao_evapotranspiration" in hourly:
        agg_spec["et0_fao_evapotranspiration"] = "sum"
    result = hourly.groupby("date", as_index=False).agg(agg_spec)
    result = result.rename(columns={
        "precipitation": "archived_forecast_precipitation_sum_mm",
        "precipitation_probability": "archived_forecast_precipitation_probability_max_pct",
        "et0_fao_evapotranspiration": "archived_forecast_et0_sum_mm",
    })
    result.insert(0, "farm_id", farm.farm_id)
    result["archived_forecast_source"] = "Open-Meteo Historical Forecast API (stitched hourly archive)"
    result["archived_forecast_downloaded_at_utc"] = utc_now_iso()
    # Important: this stitched archive is not necessarily a precise point-in-time replay
    # for a chosen farm decision time. The script does not claim issue-time alignment.
    result["archived_forecast_issue_time_alignment"] = "stitched_archive_not_exact_decision_time_replay"
    return result


def soilgrids_static(
    session: requests.Session,
    farm: pd.Series,
    raw_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # ISRIC recommends no more than 5 requests/minute. One request per farm is used.
    pairs: list[tuple[str, str]] = [
        ("lon", str(float(farm.longitude))),
        ("lat", str(float(farm.latitude))),
    ]
    for prop in SOILGRIDS_PROPERTIES_LIST:
        pairs.append(("property", prop))
    for depth in SOILGRIDS_DEPTHS:
        pairs.append(("depth", depth))
    pairs.append(("value", "mean"))

    payload = api_json(session, SOILGRIDS_PROPERTIES, pairs, timeout=90)
    save_json(raw_dir / f"{farm.farm_id}_soilgrids.json", payload)
    feature_values: dict[str, Any] = {}
    meta_rows: list[dict[str, Any]] = []
    layers = payload.get("properties", {}).get("layers", [])
    for layer in layers:
        prop = layer.get("name", "unknown")
        unit_info = layer.get("unit_measure", {}) or {}
        d_factor = unit_info.get("d_factor", 1)
        try:
            d_factor = float(d_factor) if d_factor not in (None, 0) else 1.0
        except (TypeError, ValueError):
            d_factor = 1.0
        target_units = unit_info.get("target_units") or unit_info.get("mapped_units") or "unknown"
        for depth_obj in layer.get("depths", []):
            depth_label = depth_obj.get("label", "unknown")
            raw_value = (depth_obj.get("values") or {}).get("mean")
            if raw_value is None:
                scaled_value = np.nan
            else:
                try:
                    scaled_value = float(raw_value) / d_factor
                except (TypeError, ValueError):
                    scaled_value = np.nan
            depth_name = safe_col_name(depth_label)
            col = f"soilgrids_{safe_col_name(prop)}_{depth_name}_mean"
            feature_values[col] = scaled_value
            meta_rows.append({
                "farm_id": farm.farm_id,
                "column_name": col,
                "soilgrids_property": prop,
                "depth_label": depth_label,
                "raw_mean_value": raw_value,
                "d_factor": d_factor,
                "scaled_value": scaled_value,
                "target_units": target_units,
                "source": "ISRIC SoilGrids REST API; mapped estimate not field measurement",
            })
    if not feature_values:
        raise DataCollectionError("SoilGrids response had no parseable layer values")
    return feature_values, meta_rows


def read_optional_observations(
    farm_day: pd.DataFrame,
    input_dir: Path,
) -> pd.DataFrame:
    """Merge optional field sensor and irrigation logs provided by the user."""
    out = farm_day
    sensor_path = input_dir / "soil_moisture_observations.csv"
    if sensor_path.exists():
        sensors = pd.read_csv(sensor_path)
        required = {"farm_id", "date"}
        if required.issubset(sensors.columns):
            sensors["date"] = date_series_to_strings(sensors["date"])
            sensors = sensors.drop_duplicates(["farm_id", "date"], keep="last")
            # Preserve names from the provided log, e.g. soil_moisture_20cm_vwc.
            out = out.merge(sensors, on=["farm_id", "date"], how="left", suffixes=("", "_field_log"))
        else:
            logging.warning("Ignoring %s: it must contain farm_id and date", sensor_path)
    irrigation_path = input_dir / "irrigation_logs.csv"
    if irrigation_path.exists():
        logs = pd.read_csv(irrigation_path)
        required = {"farm_id", "date", "irrigation_applied_liters"}
        if required.issubset(logs.columns):
            logs["date"] = date_series_to_strings(logs["date"])
            logs["irrigation_applied_liters"] = pd.to_numeric(logs["irrigation_applied_liters"], errors="coerce")
            logs = logs.groupby(["farm_id", "date"], as_index=False).agg({"irrigation_applied_liters": "sum"})
            logs = logs.rename(columns={"irrigation_applied_liters": "irrigation_applied_liters_observed"})
            out = out.merge(logs, on=["farm_id", "date"], how="left")
            out["irrigation_applied_today_observed"] = np.where(
                out["irrigation_applied_liters_observed"].notna(),
                out["irrigation_applied_liters_observed"].gt(0),
                np.nan,
            )
        else:
            logging.warning("Ignoring %s: it must contain farm_id, date, irrigation_applied_liters", irrigation_path)
    return out


def boolish_series(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "y": True, "n": False}
    def parse(value: Any) -> Any:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return mapping.get(str(value).strip().lower(), np.nan)
    return series.map(parse)


def generate_scenarios(
    farm_day: pd.DataFrame,
    output_path: Path,
    max_rows: int = 2_000_000,
) -> pd.DataFrame:
    """Expand each real weather row into explicitly synthetic counterfactual scenarios."""
    # Coarse, illustrative values for stress-testing pipeline logic only. Not target labels.
    soil_moisture_values = [0.12, 0.18, 0.24, 0.30, 0.36]  # m3/m3, synthetic states
    scenario_irrigated_values = [False, True]
    rainfall_values_mm = [0.0, 5.0, 15.0]
    forecast_rain_values_mm = [0.0, 5.0, 15.0]
    variants_per_row = (
        len(soil_moisture_values) * len(scenario_irrigated_values)
        * len(rainfall_values_mm) * len(forecast_rain_values_mm)
    )
    projected = len(farm_day) * variants_per_row
    if projected > max_rows:
        raise DataCollectionError(
            f"Scenario grid would create {projected:,} rows (limit {max_rows:,}). "
            "Shorten the date range or edit scenario values in the script."
        )

    scenario_axes = pd.DataFrame(
        [
            (sm, irrigated, rain, forecast_rain)
            for sm in soil_moisture_values
            for irrigated in scenario_irrigated_values
            for rain in rainfall_values_mm
            for forecast_rain in forecast_rain_values_mm
        ],
        columns=[
            "scenario_soil_moisture_vwc",
            "scenario_irrigated_today",
            "scenario_rainfall_mm",
            "scenario_forecast_rainfall_mm",
        ],
    )
    left = farm_day.copy()
    left["_join_key"] = 1
    scenario_axes["_join_key"] = 1
    scenarios = left.merge(scenario_axes, on="_join_key", how="inner").drop(columns="_join_key")
    scenarios.insert(0, "scenario_id", [f"S{i:08d}" for i in range(1, len(scenarios) + 1)])
    scenarios["row_type"] = "synthetic_counterfactual_scenario"
    scenarios["is_synthetic_scenario"] = True
    scenarios["scenario_rain_flag"] = scenarios["scenario_rainfall_mm"].gt(0.1)
    scenarios["scenario_forecast_rain_flag"] = scenarios["scenario_forecast_rainfall_mm"].gt(0.1)
    scenarios["scenario_generation_note"] = (
        "Synthetic combination for software/model pipeline testing only; not observed farm data or validated target"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_path, index=False, float_format="%.6g")
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farms", default="farms.csv", help="CSV containing farm IDs, coordinates and farm metadata")
    parser.add_argument("--start-date", default="2020-01-01", help="Historical data start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Historical data end date YYYY-MM-DD; default is local date minus 3 days")
    parser.add_argument("--outdir", default="data", help="Output root directory")
    parser.add_argument("--skip-nasa", action="store_true", help="Skip NASA POWER daily API")
    parser.add_argument("--skip-soilgrids", action="store_true", help="Skip SoilGrids static soil-map API")
    parser.add_argument("--skip-historical-forecast", action="store_true", help="Skip Open-Meteo historical forecast archive")
    parser.add_argument("--make-scenarios", action="store_true", help="Also create a synthetic counterfactual scenario CSV (not training labels)")
    parser.add_argument("--forecast-days", type=int, default=7, help="Current forecast horizon in days (Open-Meteo supports up to API limits)")
    parser.add_argument("--max-scenario-rows", type=int, default=2_000_000, help="Safety cap for synthetic scenario export")
    parser.add_argument("--soilgrids-delay-seconds", type=float, default=12.5, help="Delay between SoilGrids point requests; service recommends <=5 calls/minute")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    out_root = Path(args.outdir)
    raw_dir = out_root / "raw"
    processed_dir = out_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        farms = read_farms(Path(args.farms))
    except DataCollectionError as exc:
        logging.error("%s", exc)
        return 2

    try:
        start = date.fromisoformat(args.start_date)
    except ValueError:
        logging.error("Invalid --start-date; use YYYY-MM-DD")
        return 2
    try:
        end = date.fromisoformat(args.end_date) if args.end_date else (datetime.now(LOCAL_TZ).date() - timedelta(days=3))
    except ValueError:
        logging.error("Invalid --end-date; use YYYY-MM-DD")
        return 2
    if start > end:
        logging.error("--start-date must be on or before --end-date")
        return 2
    if end >= datetime.now(LOCAL_TZ).date():
        logging.warning("Historical end date is today or in the future; the archive may not have complete data. Forecast is collected separately.")

    session = create_session()
    all_daily: list[pd.DataFrame] = []
    soil_static_rows: list[dict[str, Any]] = []
    source_status: list[dict[str, Any]] = []
    today_str = datetime.now(LOCAL_TZ).date().isoformat()
    hist_forecast_start = max(start, date(2022, 1, 1)).isoformat()
    hist_forecast_end = min(end, datetime.now(LOCAL_TZ).date() - timedelta(days=1)).isoformat()

    for index, farm in farms.iterrows():
        farm_id = str(farm.farm_id)
        logging.info("Collecting public data for farm %s at (%s, %s)", farm_id, farm.latitude, farm.longitude)
        blocks: list[pd.DataFrame] = []
        current_df: pd.DataFrame | None = None

        # Historical weather is the core time series; fail this farm only if it is unavailable.
        try:
            archive_df = openmeteo_daily_and_hourly(session, farm, start.isoformat(), end.isoformat(), raw_dir)
            blocks.append(archive_df)
            source_status.append({"farm_id": farm_id, "source": "Open-Meteo historical weather", "status": "success", "rows": len(archive_df), "retrieved_at_utc": utc_now_iso(), "message": "ERA5-Land reanalysis; not farm station data"})
        except Exception as exc:
            logging.exception("Open-Meteo historical weather failed for %s", farm_id)
            source_status.append({"farm_id": farm_id, "source": "Open-Meteo historical weather", "status": "failed", "rows": 0, "retrieved_at_utc": utc_now_iso(), "message": str(exc)})

        if not args.skip_nasa:
            try:
                nasa_df = nasa_power_daily(session, farm, start.isoformat(), end.isoformat(), raw_dir)
                blocks.append(nasa_df)
                source_status.append({"farm_id": farm_id, "source": "NASA POWER daily", "status": "success", "rows": len(nasa_df), "retrieved_at_utc": utc_now_iso(), "message": "Gridded estimate; not farm station data"})
            except Exception as exc:
                logging.warning("NASA POWER failed for %s: %s", farm_id, exc)
                source_status.append({"farm_id": farm_id, "source": "NASA POWER daily", "status": "failed", "rows": 0, "retrieved_at_utc": utc_now_iso(), "message": str(exc)})

        try:
            current_df, forecast_df = openmeteo_current_forecast(session, farm, raw_dir, args.forecast_days)
            blocks.append(forecast_df)
            if not current_df.empty:
                blocks.append(current_df)
            source_status.append({"farm_id": farm_id, "source": "Open-Meteo current + forecast", "status": "success", "rows": len(forecast_df), "retrieved_at_utc": utc_now_iso(), "message": "Forecast snapshot recorded; current precipitation is not daily rainfall"})
        except Exception as exc:
            logging.warning("Open-Meteo current/forecast failed for %s: %s", farm_id, exc)
            source_status.append({"farm_id": farm_id, "source": "Open-Meteo current + forecast", "status": "failed", "rows": 0, "retrieved_at_utc": utc_now_iso(), "message": str(exc)})

        if not args.skip_historical_forecast and hist_forecast_start <= hist_forecast_end:
            try:
                archived_fc = openmeteo_historical_forecast(session, farm, hist_forecast_start, hist_forecast_end, raw_dir)
                if not archived_fc.empty:
                    blocks.append(archived_fc)
                source_status.append({"farm_id": farm_id, "source": "Open-Meteo historical forecast", "status": "success", "rows": len(archived_fc), "retrieved_at_utc": utc_now_iso(), "message": "Stitched archived series; not exact decision-time replay"})
            except Exception as exc:
                logging.warning("Open-Meteo historical forecast failed for %s: %s", farm_id, exc)
                source_status.append({"farm_id": farm_id, "source": "Open-Meteo historical forecast", "status": "failed", "rows": 0, "retrieved_at_utc": utc_now_iso(), "message": str(exc)})

        static_soil: dict[str, Any] = {}
        if not args.skip_soilgrids:
            try:
                static_soil, soil_rows = soilgrids_static(session, farm, raw_dir)
                soil_static_rows.extend(soil_rows)
                source_status.append({"farm_id": farm_id, "source": "ISRIC SoilGrids", "status": "success", "rows": len(soil_rows), "retrieved_at_utc": utc_now_iso(), "message": "Mapped static estimates; not field measurements"})
            except Exception as exc:
                logging.warning("SoilGrids failed for %s: %s", farm_id, exc)
                source_status.append({"farm_id": farm_id, "source": "ISRIC SoilGrids", "status": "failed", "rows": 0, "retrieved_at_utc": utc_now_iso(), "message": str(exc)})
            if index < len(farms) - 1:
                time.sleep(max(0.0, args.soilgrids_delay_seconds))

        if not blocks:
            logging.error("No data sources succeeded for farm %s; skipping farm", farm_id)
            continue

        # Outer-merge source blocks by farm-day; each source keeps its own columns.
        combined = blocks[0].copy()
        for block in blocks[1:]:
            if block.empty:
                continue
            # Each block should be unique by farm/date. A duplicate source key is a data error.
            duplicate_keys = block.duplicated(["farm_id", "date"]).any()
            if duplicate_keys:
                logging.warning("Source block has duplicate farm-date keys for %s; keeping last row", farm_id)
                block = block.drop_duplicates(["farm_id", "date"], keep="last")
            overlap_cols = [c for c in block.columns if c in combined.columns and c not in {"farm_id", "date"}]
            if overlap_cols:
                # Current/forecast and archive columns are named distinctly. For genuinely overlapping
                # metadata, preserve the existing value rather than silently suffixing columns.
                block = block.drop(columns=overlap_cols)
            combined = combined.merge(block, on=["farm_id", "date"], how="outer")

        for col in farm.index:
            if col not in {"farm_id", "latitude", "longitude"}:
                combined[col] = farm[col]
        combined["latitude"] = float(farm.latitude)
        combined["longitude"] = float(farm.longitude)
        for col, value in static_soil.items():
            combined[col] = value
        combined["timezone"] = "Asia/Kolkata"
        combined["dataset_downloaded_at_utc"] = utc_now_iso()
        combined["record_period"] = np.select(
            [pd.to_datetime(combined["date"]) < pd.Timestamp(today_str), pd.to_datetime(combined["date"]) == pd.Timestamp(today_str), pd.to_datetime(combined["date"]) > pd.Timestamp(today_str)],
            ["historical_or_past_valid_date", "current_local_date", "future_forecast_date"],
            default="unknown",
        )

        # Rain flags are based on public gridded data / forecast output, not a farm rain gauge.
        if "historical_precipitation_reanalysis_mm" in combined:
            combined["historical_rain_flag_gt_0_1mm"] = combined["historical_precipitation_reanalysis_mm"].gt(0.1).where(combined["historical_precipitation_reanalysis_mm"].notna(), np.nan)
        if "nasa_precipitation_mm_day" in combined:
            combined["nasa_rain_flag_gt_0_1mm"] = combined["nasa_precipitation_mm_day"].gt(0.1).where(combined["nasa_precipitation_mm_day"].notna(), np.nan)
        if "forecast_precipitation_sum_mm" in combined:
            combined["forecast_rain_flag_gt_0_1mm"] = combined["forecast_precipitation_sum_mm"].gt(0.1).where(combined["forecast_precipitation_sum_mm"].notna(), np.nan)

        # The farm having irrigation infrastructure is not the same thing as irrigation applied today.
        combined["irrigation_applied_today_observed"] = np.nan
        combined["irrigation_applied_liters_observed"] = np.nan
        all_daily.append(combined)

    if not all_daily:
        logging.error("No farm datasets could be assembled. Check source status and API connectivity.")
        pd.DataFrame(source_status).to_csv(processed_dir / "source_status.csv", index=False)
        return 1

    farm_day = pd.concat(all_daily, ignore_index=True, sort=False)
    farm_day["date"] = date_series_to_strings(farm_day["date"])
    farm_day = farm_day.sort_values(["farm_id", "date"]).drop_duplicates(["farm_id", "date"], keep="last")
    farm_day = read_optional_observations(farm_day, Path(args.farms).parent)

    # Recalculate the decision-facing rain flags only from respective source columns.
    for col in ["historical_rain_flag_gt_0_1mm", "nasa_rain_flag_gt_0_1mm", "forecast_rain_flag_gt_0_1mm"]:
        if col in farm_day.columns:
            farm_day[col] = farm_day[col].astype("boolean")
    if "farm_has_irrigation_system" in farm_day.columns:
        farm_day["farm_has_irrigation_system"] = boolish_series(farm_day["farm_has_irrigation_system"])

    # Make key IDs and timestamps first for easier downstream joins.
    first_cols = [
        "farm_id", "date", "latitude", "longitude", "timezone", "record_period",
        "irrigation_method", "farm_has_irrigation_system", "irrigated_area_ha",
        "palm_age_years", "palm_count", "soil_texture_field",
    ]
    ordered = [c for c in first_cols if c in farm_day.columns] + [c for c in farm_day.columns if c not in first_cols]
    farm_day = farm_day[ordered]
    main_csv = processed_dir / "arecanut_farm_day.csv"
    farm_day.to_csv(main_csv, index=False, float_format="%.6g")
    logging.info("Saved %s (%s rows, %s columns)", main_csv, f"{len(farm_day):,}", len(farm_day.columns))

    soil_csv = processed_dir / "soilgrids_static.csv"
    pd.DataFrame(soil_static_rows).to_csv(soil_csv, index=False, float_format="%.6g")
    status_csv = processed_dir / "source_status.csv"
    pd.DataFrame(source_status).to_csv(status_csv, index=False)

    if args.make_scenarios:
        scenario_csv = processed_dir / "arecanut_scenarios.csv"
        scenarios = generate_scenarios(farm_day, scenario_csv, max_rows=args.max_scenario_rows)
        logging.info("Saved %s (%s synthetic rows)", scenario_csv, f"{len(scenarios):,}")

    units_path = processed_dir / "data_dictionary.csv"
    data_dictionary = [
        ("historical_*", "See field suffix; Open-Meteo ERA5-Land reanalysis; not station measurements"),
        ("soil_moisture_reanalysis_*_vwc", "m3/m3; model/reanalysis soil moisture, not sensor VWC"),
        ("forecast_precipitation_sum_mm", "mm/day predicted by current forecast snapshot"),
        ("forecast_precipitation_probability_max_pct", "%; max hourly precipitation probability for the forecast date"),
        ("archived_forecast_*", "Archived stitched forecast series; not an exact issue-time replay"),
        ("nasa_precipitation_mm_day", "mm/day; NASA POWER gridded precipitation estimate"),
        ("nasa_shortwave_radiation_kwh_m2_day", "kWh/m2/day; NASA POWER daily solar radiation"),
        ("soilgrids_*", "ISRIC SoilGrids estimated static property; units are recorded in soilgrids_static.csv"),
        ("irrigation_applied_liters_observed", "L/day only if supplied in irrigation_logs.csv; otherwise blank"),
        ("scenario_*", "Synthetic counterfactual values only; not measured and not validated labels"),
    ]
    pd.DataFrame(data_dictionary, columns=["column_pattern", "meaning_and_units"]).to_csv(units_path, index=False)
    logging.info("Wrote source status to %s", status_csv)
    logging.info("Collection finished. Review source_status.csv and the data dictionary before modelling.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logging.error("Interrupted by user")
        raise SystemExit(130)
