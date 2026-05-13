import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
OBSERVATION_START = "1960-01-01"


def _fetch_series(series_id: str) -> pd.Series:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY is not set. Check your .env file.")

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": OBSERVATION_START,
    }

    response = requests.get(FRED_BASE_URL, params=params, timeout=15)
    response.raise_for_status()

    observations = response.json().get("observations", [])
    if not observations:
        raise ValueError(f"No data returned for series {series_id}")

    dates = pd.to_datetime([o["date"] for o in observations])
    values = pd.array([o["value"] for o in observations], dtype=object)
    series = (
        pd.Series(values, index=dates, name=series_id)
        .replace(".", pd.NA)
        .dropna()
        .astype(float)
    )
    series.name = series_id
    return series


def load_data() -> pd.DataFrame:
    unrate = _fetch_series("UNRATE")
    cpi = _fetch_series("CPIAUCSL")

    # Compute year-over-year CPI inflation rate (%)
    inflation = cpi.pct_change(12) * 100
    inflation.name = "INFLATION"

    df = pd.concat([unrate, inflation], axis=1).dropna()
    df.index.name = "date"
    df.columns = ["unemployment", "inflation"]
    df["decade"] = (df.index.year // 10 * 10).astype(str) + "s"
    return df
