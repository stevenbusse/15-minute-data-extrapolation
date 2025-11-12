"""Generate a synthetic 15-minute load profile for an entire year.

The script consumes a partial history of quarter-hour load data and two sets
of control points:

* A daily curve with a handful of time-of-day grab points.  These points let
  analysts drag the load shape up or down at arbitrary times of the day.  A
  smooth cubic Hermite spline connects the points to create a multiplier that
  is applied to the time-of-day profile that is derived from the historical
  data.
* A monthly curve with twelve grab points (one per month).  The same spline
  approach is used to produce a smooth set of monthly multipliers that scale
  the daily profile.

The combined multiplier is applied to a baseline time-of-day profile that is
calculated from the provided data, yielding a full-year set of quarter-hour
values.  The output is written to an Excel file with three sheets:

1. ``full_year_load`` – The completed 15 minute data set.
2. ``daily_curve`` – The daily multiplier evaluated for each 15 minute slot.
3. ``monthly_curve`` – The monthly multipliers for each month.

Example usage::

    python -m generate_load \
        --input historical_load.xlsx \
        --output extrapolated_load.xlsx \
        --start-date 2023-01-01 \
        --config curve_points.json

The configuration file is optional.  When omitted, neutral multipliers (1.0)
are used for both the daily and monthly curves.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyCurvePoint:
    """A grab point that adjusts the daily profile.

    Attributes
    ----------
    minutes: int
        Minutes since midnight for the control point.
    multiplier: float
        The desired multiplier at the control point.
    """

    minutes: int
    multiplier: float


@dataclass(frozen=True)
class MonthlyCurvePoint:
    """A grab point that adjusts the monthly profile."""

    month: int
    multiplier: float


def parse_time_string(time_string: str) -> int:
    """Convert ``HH:MM`` strings to minutes since midnight."""

    hours, minutes = map(int, time_string.split(":"))
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        msg = f"Time string must be between 00:00 and 23:59, got {time_string!r}."
        raise ValueError(msg)
    return hours * 60 + minutes


def ensure_quarter_hour_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Validate that the supplied index is spaced at 15 minute intervals."""

    if len(index) < 2:
        return index

    diffs = index.to_series().diff().dropna().dt.total_seconds()
    expected = 15 * 60
    if not np.allclose(diffs, expected):
        msg = (
            "Input timestamps must be spaced every 15 minutes. "
            "Found differences: "
            f"{sorted(set(int(d) for d in diffs.unique()))[:5]} seconds."
        )
        raise ValueError(msg)
    return index


def load_input_data(path: Path, start_date: pd.Timestamp | None) -> pd.Series:
    """Load a single column of 15-minute data from CSV or Excel files."""

    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Input file does not contain any data.")

    # Prefer an explicit timestamp column if one is present.
    datetime_candidates = [
        col
        for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
        or col.lower() in {"timestamp", "datetime", "date", "time"}
    ]

    if datetime_candidates:
        timestamp_col = datetime_candidates[0]
        timestamps = pd.to_datetime(df.pop(timestamp_col))
    else:
        if start_date is None:
            raise ValueError(
                "The input data does not contain a timestamp column. "
                "Please provide --start-date so timestamps can be inferred."
            )
        timestamps = pd.date_range(
            start=start_date,
            periods=len(df),
            freq="15min",
        )

    # Use the first numeric column as the load values.
    numeric_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_columns:
        raise ValueError("No numeric column found in the input data.")

    load_series = df[numeric_columns[0]].astype(float)
    load_series.index = pd.to_datetime(timestamps)
    load_series = load_series.sort_index()
    load_series.index = ensure_quarter_hour_index(load_series.index)
    return load_series


def infer_start_year(series: pd.Series) -> int:
    """Determine the year to synthesize.  Prefer the first year in the data."""

    if len(series.index) == 0:
        raise ValueError("Cannot infer year from empty series.")
    return int(series.index[0].year)


def build_time_of_day_profile(series: pd.Series) -> pd.Series:
    """Return the average load for each quarter-hour slot."""

    grouped = series.groupby(series.index.time).mean()
    # Ensure all 96 quarter-hour buckets exist.
    quarter_hours = pd.date_range("00:00", "23:45", freq="15min").time
    profile = grouped.reindex(quarter_hours, fill_value=np.nan)
    profile = profile.fillna(profile.mean())
    return profile


def evaluate_daily_curve(
    points: Sequence[DailyCurvePoint],
) -> pd.Series:
    """Evaluate the daily curve at 15 minute increments."""

    if not points:
        points = [DailyCurvePoint(0, 1.0), DailyCurvePoint(1440, 1.0)]

    sorted_points = sorted(points, key=lambda p: p.minutes)
    minutes = np.array([p.minutes for p in sorted_points], dtype=float)
    multipliers = np.array([p.multiplier for p in sorted_points], dtype=float)

    # Add wrap-around point if needed to ensure coverage of the full day.
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
    """Evaluate the monthly curve at integer months 1-12."""

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


def cubic_hermite_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    x_new: Iterable[float],
) -> np.ndarray:
    """Piecewise cubic Hermite interpolation preserving monotonic segments."""

    if len(x) != len(y):
        raise ValueError("Control point arrays must be the same length.")
    if np.any(np.diff(x) < 0):
        raise ValueError("Control point x-values must be increasing.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(list(x_new), dtype=float)

    if len(x) == 2:
        # Fall back to linear interpolation.
        slopes = (y[1] - y[0]) / (x[1] - x[0])
        return y[0] + slopes * (x_new - x[0])

    # Estimate slopes with the Fritsch-Carlson method which preserves shape.
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

    # Evaluate the piecewise Hermite polynomials.
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


def compute_monthly_adjustment(
    series: pd.Series,
    base_profile: pd.Series,
) -> pd.Series:
    """Derive month-specific adjustments from the provided data."""

    if series.empty:
        return pd.Series(1.0, index=pd.RangeIndex(1, 13))

    # Compare the observed load with the base profile for matching timestamps.
    expected = series.index.map(lambda ts: base_profile[ts.time()])
    ratio = series / expected
    monthly = ratio.groupby(series.index.month).median()
    monthly = monthly.reindex(range(1, 13), fill_value=np.nan)
    monthly = monthly.fillna(monthly.mean()).fillna(1.0)
    return monthly


def synthesize_full_year(
    base_profile: pd.Series,
    daily_curve: pd.Series,
    monthly_curve: pd.Series,
    monthly_adjustment: pd.Series,
    year: int,
) -> pd.Series:
    """Create the full-year 15 minute series."""

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
    """Load curve points from a JSON configuration file."""

    if path is None:
        return [], []

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    daily_points = []
    for item in data.get("daily_curve_points", []):
        minutes = parse_time_string(item["time"])
        daily_points.append(
            DailyCurvePoint(
                minutes=minutes,
                multiplier=float(item["multiplier"]),
            )
        )

    monthly_points = []
    for item in data.get("monthly_curve_points", []):
        month = int(item["month"])
        if not 1 <= month <= 12:
            raise ValueError(f"Month must be between 1 and 12, got {month}.")
        monthly_points.append(
            MonthlyCurvePoint(
                month=month,
                multiplier=float(item["multiplier"]),
            )
        )

    return daily_points, monthly_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input CSV or Excel file with 15 minute data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination Excel file for the full year series. Required unless"
            " --preview is supplied."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=pd.Timestamp,
        default=None,
        help="Starting timestamp for the input series when the file lacks explicit timestamps.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to synthesize. Defaults to the year of the first data point.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file with daily and monthly curve points.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Print a textual preview of the synthesized series. When provided"
            " the Excel workbook is only written if --output is also set."
        ),
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Number of rows to display when --preview is set.",
    )
    return parser


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


def preview_results(
    full_year: pd.Series,
    daily_curve: pd.Series,
    monthly_curve: pd.Series,
    rows: int,
) -> None:
    """Print a concise preview of the generated series to stdout."""

    rows = max(1, rows)
    print("=== Full year load preview ===")
    print(full_year.head(rows).to_frame(name="load"))
    print()
    print(
        "Total points:"
        f" {len(full_year):,} (approximately {len(full_year) / 96:.1f} days of data)"
    )
    print("Min/Max (MW):", round(float(full_year.min()), 3), "/", round(float(full_year.max()), 3))
    print()
    print("=== Daily curve multipliers (first 8 slots) ===")
    print(daily_curve.head(8).to_frame(name="multiplier"))
    print()
    print("=== Monthly curve multipliers ===")
    print(monthly_curve.to_frame(name="multiplier"))
    print()


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.preview and args.output is None:
        parser.error("--output is required unless --preview is specified.")

    daily_points, monthly_points = load_configuration(args.config)

    series = load_input_data(args.input, args.start_date)
    base_profile = build_time_of_day_profile(series)
    daily_curve = evaluate_daily_curve(daily_points)
    monthly_curve = evaluate_monthly_curve(monthly_points)
    monthly_adjustment = compute_monthly_adjustment(series, base_profile)

    year = args.year or infer_start_year(series)
    full_year = synthesize_full_year(
        base_profile=base_profile,
        daily_curve=daily_curve,
        monthly_curve=monthly_curve,
        monthly_adjustment=monthly_adjustment,
        year=year,
    )

    if args.preview:
        preview_results(full_year, daily_curve, monthly_curve, args.preview_rows)

    if args.output is not None:
        write_output(args.output, full_year, daily_curve, monthly_curve)
    elif not args.preview:
        # Should be unreachable thanks to earlier validation, but guards
        # against future changes to argument handling.
        parser.error("Output path missing.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
