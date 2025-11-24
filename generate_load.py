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
            # require at least 25% parseability to consider it a timestamp column
            tscol = best

    if tscol is not None:
        df[tscol] = pd.to_datetime(df[tscol], errors="coerce")
        df = df.loc[~df[tscol].isna()].copy()
        df = df.set_index(tscol)
        # If there are duplicate timestamps, average them
        if df.index.duplicated().any():
            df = df.groupby(df.index).mean()
    else:
        # No timestamp column; synthesize index from provided start_date or Jan 1
        if start_date is None:
            start_date = pd.Timestamp(pd.Timestamp.now().year, 1, 1)
        df.index = pd.date_range(start=start_date, periods=len(df), freq="15min")

    # Choose numeric column: prefer numeric dtypes, otherwise coerce and pick best
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

    # If the data has irregular spacing, resample from min to max on a strict 15T grid
    start = series.index.min()
    end = series.index.max()
    full_index = pd.date_range(start=start, end=end, freq="15min")

    # Align to the new index, interpolate and fill edges
    series = series.reindex(series.index.union(full_index))
    # interpolate in time where possible
    series = series.sort_index().interpolate(method="time")
    series = series.reindex(full_index)
    series = series.fillna(method="ffill").fillna(method="bfill")

    # final check: ensure index is strict 15T
    try:
        series.index = ensure_quarter_hour_index(series.index)
    except Exception:
        # if something still odd, coerce to the requested full_index
        series.index = full_index

    return series


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
    monthly_kwh_targets: dict[int, float] | None = None,
) -> pd.Series:
    """Create the full-year 15 minute series."""
    # Build an index that covers every 15-minute slot in the requested year.
    # Use an exclusive end at Jan 1 of the following year to avoid YearEnd edge cases.
    start = pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0)
    end_exclusive = pd.Timestamp(year=year + 1, month=1, day=1, hour=0, minute=0)
    # Some pandas versions do not support the ``closed`` keyword on
    # ``pd.date_range``. To remain compatible, subtract one 15-minute step
    # from the exclusive end and generate an inclusive range instead.
    end_adj = end_exclusive - pd.Timedelta(minutes=15)
    index = pd.date_range(start=start, end=end_adj, freq="15min")

    # Vectorized lookups for speed and robustness.  Reindexing a small Series
    # is cheaper than calling a lambda for every timestamp.
    times = index.time
    months = index.month

    base_values = base_profile.reindex(times).to_numpy()
    daily_values = daily_curve.reindex(times).to_numpy()
    monthly_values = monthly_curve.reindex(months).to_numpy()
    monthly_adj_values = (
        monthly_adjustment.reindex(months, fill_value=1.0).to_numpy()
    )

    if monthly_kwh_targets:
        # Precompute daily weights from the daily curve (96 slots sum to 1).
        daily_sum = float(daily_curve.sum())
        if daily_sum == 0 or np.isnan(daily_sum):
            normalized_daily = np.ones_like(daily_curve.values) / len(daily_curve)
        else:
            normalized_daily = daily_curve.values / daily_sum
        # Slight deterministic variation across days to avoid perfectly flat months.
        # 2% swing over the month.
        load = np.zeros_like(base_values)
        # build date index aligned to slots
        dates = pd.Series(index).dt.normalize().to_numpy()
        for month in range(1, 13):
            if month not in monthly_kwh_targets:
                continue
            try:
                target_val = float(monthly_kwh_targets[month])
            except Exception:
                continue
            mask = months == month
            if not np.any(mask):
                continue
            # all days in this month
            month_dates = pd.unique(dates[mask])
            day_count = len(month_dates)
            if day_count == 0:
                continue
            # day weights with slight ripple
            day_idx = np.arange(day_count, dtype=float)
            day_weights = 1.0 + 0.02 * np.sin(2 * np.pi * day_idx / max(day_count, 1))
            day_weights = day_weights / day_weights.sum()
            # energy per day
            day_targets = target_val * day_weights
            # fill slots for each day
            for d, day in enumerate(month_dates):
                day_mask = mask & (dates == day)
                slots = int(np.sum(day_mask))
                if slots == 0:
                    continue
                # repeat normalized daily profile across the day's slots (96 expected)
                weights = np.resize(normalized_daily, slots)
                weights_sum = weights.sum()
                if weights_sum == 0 or np.isnan(weights_sum):
                    weights = np.full(slots, 1.0 / slots)
                else:
                    weights = weights / weights_sum
                # distribute day's energy target across slots
                load[day_mask] = day_targets[d] * weights
        # Final correction per month to hit exact kWh (sum of slot energies = target)
        for month, target in monthly_kwh_targets.items():
            mask = months == month
            if not np.any(mask):
                continue
            current = float(np.sum(load[mask]))
            try:
                target_val = float(target)
            except Exception:
                continue
            if current and np.isfinite(current) and current != 0:
                load[mask] *= target_val / current
    else:
        base = base_values * daily_values * monthly_adj_values
        load = base * monthly_values

    return pd.Series(load, index=index)


def estimate_monthly_baseline_energy(
    base_profile: pd.Series,
    daily_curve: pd.Series,
    monthly_adjustment: pd.Series,
    year: int,
    use_proportional_daily: bool = False,
) -> pd.Series:
    """Estimate month energy (in input units * hours) without monthly multipliers."""
    start = pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0)
    end_exclusive = pd.Timestamp(year=year + 1, month=1, day=1, hour=0, minute=0)
    end_adj = end_exclusive - pd.Timedelta(minutes=15)
    index = pd.date_range(start=start, end=end_adj, freq="15min")
    times = index.time
    months = index.month
    base_values = base_profile.reindex(times).to_numpy()
    daily_values = daily_curve.reindex(times).to_numpy()
    if use_proportional_daily:
        daily_sum = float(daily_curve.sum())
        if daily_sum == 0 or np.isnan(daily_sum):
            daily_values = np.ones_like(daily_values)
        else:
            daily_values = daily_values / daily_sum * len(daily_curve)
    monthly_adj_values = (
        monthly_adjustment.reindex(months, fill_value=1.0).to_numpy()
    )
    base = base_values * daily_values * monthly_adj_values
    df = pd.DataFrame({"month": months, "base": base})
    month_energy = df.groupby("month")["base"].sum() * 0.25
    month_energy = month_energy.reindex(range(1, 13), fill_value=np.nan)
    return month_energy


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
