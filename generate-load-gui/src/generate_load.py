"""Generate a synthetic 15-minute load profile for an entire year.

This module contains the logic for generating a synthetic load profile based on historical data and user-defined daily and monthly curves. It integrates with the GUI to allow user interaction for file input and curve adjustments.

The script consumes a partial history of quarter-hour load data and two sets of control points:

* A daily curve with a handful of time-of-day grab points. These points let
  analysts drag the load shape up or down at arbitrary times of the day. A
  smooth cubic Hermite spline connects the points to create a multiplier that is applied to the time-of-day profile that is derived from the historical data.
* A monthly curve with twelve grab points (one per month). The same spline
  approach is used to produce a smooth set of monthly multipliers that scale
  the daily profile.

The combined multiplier is applied to a baseline time-of-day profile that is
calculated from the provided data, yielding a full-year set of quarter-hour
values. The output is written to an Excel file with three sheets:

1. ``full_year_load`` – The completed 15 minute data set.
2. ``daily_curve`` – The daily multiplier evaluated for each 15 minute slot.
3. ``monthly_curve`` – The monthly multipliers for each month.

Example usage::

    from generate_load import generate_load_profile

    profile = generate_load_profile(
        historical_load_data='path/to/historical_load.xlsx',
        output_file='path/to/extrapolated_load.xlsx',
        start_date='2023-01-01',
        config_file='path/to/curve_points.json'
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyCurvePoint:
    minutes: int
    multiplier: float


@dataclass(frozen=True)
class MonthlyCurvePoint:
    month: int
    multiplier: float


def parse_time_string(time_string: str) -> int:
    hours, minutes = map(int, time_string.split(":"))
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        raise ValueError(f"Time string must be between 00:00 and 23:59, got {time_string!r}.")
    return hours * 60 + minutes


def ensure_quarter_hour_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if len(index) < 2:
        return index

    diffs = index.to_series().diff().dropna().dt.total_seconds()
    expected = 15 * 60
    if not np.allclose(diffs, expected):
        raise ValueError("Input timestamps must be spaced every 15 minutes.")
    return index


def load_input_data(path: Path, start_date: pd.Timestamp | None) -> pd.Series:
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Input file does not contain any data.")

    datetime_candidates = [
        col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or col.lower() in {"timestamp", "datetime", "date", "time"}
    ]

    if datetime_candidates:
        timestamp_col = datetime_candidates[0]
        timestamps = pd.to_datetime(df.pop(timestamp_col))
    else:
        if start_date is None:
            raise ValueError("The input data does not contain a timestamp column. Please provide --start-date.")
        timestamps = pd.date_range(start=start_date, periods=len(df), freq="15min")

    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_columns:
        raise ValueError("No numeric column found in the input data.")

    load_series = df[numeric_columns[0]].astype(float)
    load_series.index = pd.to_datetime(timestamps)
    load_series = load_series.sort_index()
    load_series.index = ensure_quarter_hour_index(load_series.index)
    return load_series


def load_input_data_robust(path: Path, start_date: pd.Timestamp | None = None) -> pd.Series:
    """Robust loader that accepts irregular or incomplete 15-minute data.

    This function tries harder to locate/parse timestamps and numeric columns,
    handles duplicates by averaging, resamples onto a strict 15-minute grid
    between the minimum and maximum timestamps, and interpolates/fills
    missing values so downstream code receives a clean quarter-hour series.
    """

    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Input file does not contain any data.")

    # Try to find or construct a datetime index.
    tscol = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            tscol = col
            break
    if tscol is None:
        for col in df.columns:
            if col.lower() in {"timestamp", "datetime", "date", "time"}:
                tscol = col
                break

    # If still not found, attempt coercion and pick the best candidate.
    if tscol is None:
        best = None
        best_count = 0
        for col in df.columns:
            coerced = pd.to_datetime(df[col], errors="coerce")
            nonnull = int(coerced.notna().sum())
            if nonnull > best_count:
                best = col
                best_count = nonnull
        if best is not None and best_count >= max(1, len(df) // 4):
            tscol = best

    if tscol is not None:
        df[tscol] = pd.to_datetime(df[tscol], errors="coerce")
        df = df.loc[~df[tscol].isna()].copy()
        df = df.set_index(tscol)
        if df.index.duplicated().any():
            df = df.groupby(df.index).mean()
    else:
        if start_date is None:
            start_date = pd.Timestamp(pd.Timestamp.now().year, 1, 1)
        df.index = pd.date_range(start=start_date, periods=len(df), freq="15min")

    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    chosen_col = None
    if numeric_columns:
        chosen_col = numeric_columns[0]
    else:
        coerced = df.apply(lambda c: pd.to_numeric(c, errors="coerce"))
        nonnull_counts = coerced.notna().sum()
        if nonnull_counts.max() > 0:
            chosen_col = nonnull_counts.idxmax()
            df[chosen_col] = coerced[chosen_col]

    if chosen_col is None:
        raise ValueError("No numeric column found in the input data.")

    series = df[chosen_col].astype(float).sort_index()

    start = series.index.min()
    end = series.index.max()
    full_index = pd.date_range(start=start, end=end, freq="15min")

    series = series.reindex(series.index.union(full_index))
    series = series.sort_index().interpolate(method="time")
    series = series.reindex(full_index)
    series = series.fillna(method="ffill").fillna(method="bfill")

    try:
        series.index = ensure_quarter_hour_index(series.index)
    except Exception:
        series.index = full_index

    return series


def infer_start_year(series: pd.Series) -> int:
    if len(series.index) == 0:
        raise ValueError("Cannot infer year from empty series.")
    return int(series.index[0].year)


def build_time_of_day_profile(series: pd.Series) -> pd.Series:
    grouped = series.groupby(series.index.time).mean()
    quarter_hours = pd.date_range("00:00", "23:45", freq="15min").time
    profile = grouped.reindex(quarter_hours, fill_value=np.nan)
    profile = profile.fillna(profile.mean())
    return profile


def evaluate_daily_curve(points: Sequence[DailyCurvePoint]) -> pd.Series:
    if not points:
        points = [DailyCurvePoint(0, 1.0), DailyCurvePoint(1440, 1.0)]

    sorted_points = sorted(points, key=lambda p: p.minutes)
    minutes = np.array([p.minutes for p in sorted_points], dtype=float)
    multipliers = np.array([p.multiplier for p in sorted_points], dtype=float)

    if minutes[0] != 0:
        minutes = np.insert(minutes, 0, 0.0)
        multipliers = np.insert(multipliers, 0, multipliers[0])
    if minutes[-1] != 1440:
        minutes = np.append(minutes, 1440.0)
        multipliers = np.append(multipliers, multipliers[-1])

    sample_minutes = np.arange(0, 1440, 15)
    sample_values = cubic_hermite_interpolate(minutes, multipliers, sample_minutes)

    times = pd.date_range("00:00", "23:45", freq="15min").time
    return pd.Series(sample_values, index=times)


def evaluate_monthly_curve(points: Sequence[MonthlyCurvePoint]) -> pd.Series:
    if not points:
        points = [MonthlyCurvePoint(1, 1.0), MonthlyCurvePoint(12, 1.0)]

    sorted_points = sorted(points, key=lambda p: p.month)
    months = np.array([p.month for p in sorted_points], dtype=float)
    multipliers = np.array([p.multiplier for p in sorted_points], dtype=float)

    if months[0] > 1:
        months = np.insert(months, 0, 1.0)
        multipliers = np.insert(multipliers, 0, multipliers[0])
    if months[-1] < 12:
        months = np.append(months, 12.0)
        multipliers = np.append(multipliers, multipliers[-1])

    sample_months = np.arange(1, 13)
    sample_values = cubic_hermite_interpolate(months, multipliers, sample_months)

    return pd.Series(sample_values, index=sample_months)


def cubic_hermite_interpolate(x: np.ndarray, y: np.ndarray, x_new: Iterable[float]) -> np.ndarray:
    if len(x) != len(y):
        raise ValueError("Control point arrays must be the same length.")
    if np.any(np.diff(x) < 0):
        raise ValueError("Control point x-values must be increasing.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(list(x_new), dtype=float)

    if len(x) == 2:
        slopes = (y[1] - y[0]) / (x[1] - x[0])
        return y[0] + slopes * (x_new - x[0])

    h = np.diff(x)
    delta = np.diff(y) / h

    slopes = np.zeros_like(y)
    slopes[0] = delta[0]
    slopes[-1] = delta[-1]

    for i in range(1, len(y) - 1):
        if delta[i - 1] * delta[i] <= 0:
            slopes[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    result = np.empty_like(x_new)
    for idx, value in enumerate(x_new):
        if value <= x[0]:
            i = 0
        elif value >= x[-1]:
            i = len(x) - 2
        else:
            i = np.searchsorted(x, value) - 1

        h_i = x[i + 1] - x[i]
        t = (value - x[i]) / h_i
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t**2 * (3 - 2 * t)
        h11 = t**2 * (t - 1)

        result[idx] = (
            h00 * y[i]
            + h10 * h_i * slopes[i]
            + h01 * y[i + 1]
            + h11 * h_i * slopes[i + 1]
        )
    return result


def compute_monthly_adjustment(series: pd.Series, base_profile: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(1.0, index=pd.RangeIndex(1, 13))

    expected = series.index.map(lambda ts: base_profile[ts.time()])
    ratio = series / expected
    monthly = ratio.groupby(series.index.month).median()
    monthly = monthly.reindex(range(1, 13), fill_value=np.nan)
    monthly = monthly.fillna(monthly.mean()).fillna(1.0)
    return monthly


def synthesize_full_year(base_profile: pd.Series, daily_curve: pd.Series, monthly_curve: pd.Series, monthly_adjustment: pd.Series, year: int) -> pd.Series:
    start = pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0)
    end = start + pd.offsets.YearEnd()
    index = pd.date_range(start=start, end=end + pd.Timedelta(minutes=15), freq="15min", inclusive="left")

    base_values = index.map(lambda ts: base_profile[ts.time()])
    daily_values = index.map(lambda ts: daily_curve[ts.time()])
    monthly_values = index.map(lambda ts: monthly_curve[ts.month])
    monthly_adj_values = index.map(lambda ts: monthly_adjustment.get(ts.month, 1.0))

    load = base_values * daily_values * monthly_values * monthly_adj_values
    return pd.Series(load, index=index)


def load_configuration(path: Path | None) -> tuple[List[DailyCurvePoint], List[MonthlyCurvePoint]]:
    if path is None:
        return [], []

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    daily_points = []
    for item in data.get("daily_curve_points", []):
        minutes = parse_time_string(item["time"])
        daily_points.append(DailyCurvePoint(minutes=minutes, multiplier=float(item["multiplier"])))

    monthly_points = []
    for item in data.get("monthly_curve_points", []):
        month = int(item["month"])
        if not 1 <= month <= 12:
            raise ValueError(f"Month must be between 1 and 12, got {month}.")
        monthly_points.append(MonthlyCurvePoint(month=month, multiplier=float(item["multiplier"])))

    return daily_points, monthly_points


def write_output(
    output_path: Path,
    full_year: pd.Series,
    daily_curve: pd.Series,
    monthly_curve: pd.Series,
) -> None:
    """Write the output Excel file with the synthesized data and multipliers."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        full_year.to_frame(name="load").to_excel(writer, sheet_name="full_year_load")
        daily_curve.rename("multiplier").to_frame().to_excel(writer, sheet_name="daily_curve")
        monthly_curve.rename("multiplier").to_frame().to_excel(writer, sheet_name="monthly_curve")
