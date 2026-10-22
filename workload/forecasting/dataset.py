"""Historical airline demand dataset loader, validator, and chronological splitter.

Handles empirical data from the U.S. Bureau of Transportation Statistics (BTS)
TranStats reporting carrier on-time performance and passenger traffic tables.
Strictly enforces chronological time-series splitting without shuffling to prevent
future-data leakage.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("baccp.workload.forecasting.dataset")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "workload" / "datasets" / "sample_data" / "bts_daily_passenger_demand.csv"


class DatasetValidationError(ValueError):
    """Raised when the airline demand dataset fails integrity or schema checks."""
    pass


def load_airline_demand_dataset(
    path: Optional[Union[str, Path]] = None,
    target_column: str = "enplaned_passengers",
    date_column: str = "date",
    fill_missing: bool = True,
) -> pd.DataFrame:
    """Load and validate historical airline passenger demand time-series data.
    
    Parameters:
        path: Path to CSV dataset. Defaults to bundled BTS sample dataset.
        target_column: Primary demand target variable.
        date_column: Date/timestamp column name.
        fill_missing: If True, interpolates missing calendar dates using forward fill.
        
    Returns:
        Validated DataFrame sorted chronologically with continuous DateTimeIndex.
    """
    target_path = Path(path) if path else DEFAULT_DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Airline demand dataset not found at: {target_path}")

    # 1. Ingestion
    df = pd.read_csv(target_path)
    if df.empty:
        raise DatasetValidationError(f"Dataset at {target_path} is empty.")

    # 2. Date column validation
    if date_column not in df.columns:
        raise DatasetValidationError(f"Required date column '{date_column}' not found. Available: {list(df.columns)}")

    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except Exception as exc:
        raise DatasetValidationError(f"Failed to parse dates in column '{date_column}': {exc}")

    # 3. Duplicate handling
    duplicates = df[df.duplicated(subset=[date_column], keep=False)]
    if not duplicates.empty:
        logger.warning(f"Detected {len(duplicates)} duplicate timestamps. Aggregating duplicates by mean.")
        df = df.groupby(date_column, as_index=False).mean()

    # Sort strictly chronologically
    df = df.sort_values(by=date_column).reset_index(drop=True)

    # 4. Target variable validation
    if target_column not in df.columns:
        raise DatasetValidationError(
            f"Target column '{target_column}' missing. Available: {list(df.columns)}"
        )

    # Convert target to numeric
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    if df[target_column].isna().all():
        raise DatasetValidationError(f"Target column '{target_column}' contains no valid numeric values.")

    if (df[target_column] < 0).any():
        raise DatasetValidationError(f"Target column '{target_column}' contains negative values, which is physically invalid for passenger demand.")

    # 5. Continuous time interval verification
    df = df.set_index(date_column)
    # Detect calendar date gaps
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    missing_dates = date_range.difference(df.index)
    
    if len(missing_dates) > 0:
        logger.warning(f"Identified {len(missing_dates)} missing calendar days in time series.")
        if fill_missing:
            df = df.reindex(date_range)
            # Forward-fill target then backward-fill edge
            df[target_column] = df[target_column].ffill().bfill()
            if "day_of_week" in df.columns:
                df["day_of_week"] = df.index.strftime("%A")
            if "day_of_week_num" in df.columns:
                df["day_of_week_num"] = df.index.weekday
            if "year" in df.columns:
                df["year"] = df.index.year
            if "month" in df.columns:
                df["month"] = df.index.month
            if "day" in df.columns:
                df["day"] = df.index.day

    df.index.name = date_column
    return df.reset_index()


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_column: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series strictly chronologically without random shuffling.
    
    Guarantees:
    - Train set precedes Validation set in time: max(train_date) < min(val_date)
    - Validation set precedes Test set in time: max(val_date) < min(test_date)
    - Zero future-data leakage
    """
    total_len = len(df)
    if total_len < 10:
        raise DatasetValidationError(f"Dataset too small ({total_len} rows) for train/val/test splitting.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    train_r = train_ratio / ratio_sum
    val_r = val_ratio / ratio_sum

    train_end = int(total_len * train_r)
    val_end = int(total_len * (train_r + val_r))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    # Verification assertions
    train_max_date = pd.to_datetime(train_df[date_column]).max()
    val_min_date = pd.to_datetime(val_df[date_column]).min()
    val_max_date = pd.to_datetime(val_df[date_column]).max()
    test_min_date = pd.to_datetime(test_df[date_column]).min()

    if train_max_date >= val_min_date:
        raise DatasetValidationError(f"Temporal leakage detected: train_max ({train_max_date}) >= val_min ({val_min_date})")
    if val_max_date >= test_min_date:
        raise DatasetValidationError(f"Temporal leakage detected: val_max ({val_max_date}) >= test_min ({test_min_date})")

    logger.info(
        f"Chronological split created: Train={len(train_df)} ({train_df[date_column].iloc[0]} to {train_max_date}), "
        f"Val={len(val_df)} ({val_min_date} to {val_max_date}), "
        f"Test={len(test_df)} ({test_min_date} to {test_df[date_column].iloc[-1]})"
    )

    return train_df, val_df, test_df
