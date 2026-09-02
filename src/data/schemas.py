"""Schemas and metadata definitions for the FICOS ML data pipeline."""

from typing import List, Dict

DATE_COLUMN: str = "date"

TARGET_COLUMNS: List[str] = [
    "bdi_hsi",
    "bdi_si",
    "bdi_pi",
    "bdi_ci",
]

MARKET_COLUMNS: List[str] = [
    "wti_usd_bbl",
    "brent_usd_bbl",
    "usd_inr",
]

GPR_COLUMNS: List[str] = [
    "gpr",
    "gpr_acts",
    "gpr_threats",
    "gpr_ma7",
    "gpr_ma30",
    "gpr_article_count",
]

WEATHER_COLUMNS: List[str] = [
    "wind_speed_max_kmh",
    "wind_gust_max_kmh",
    "precip_mm",
    "pressure_hpa",
]

FLAG_COLUMNS: List[str] = [
    "is_bdi_trading_day",
    "is_market_trading_day",
]

ALL_EXPECTED_COLUMNS: List[str] = [
    DATE_COLUMN,
    *TARGET_COLUMNS,
    *MARKET_COLUMNS,
    *GPR_COLUMNS,
    *WEATHER_COLUMNS,
    *FLAG_COLUMNS,
]

NUMERIC_COLUMNS: List[str] = [
    *TARGET_COLUMNS,
    *MARKET_COLUMNS,
    *GPR_COLUMNS,
    *WEATHER_COLUMNS,
]

BOOLEAN_COLUMNS: List[str] = list(FLAG_COLUMNS)
