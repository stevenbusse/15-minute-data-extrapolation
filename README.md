# 15 Minute Data Extrapolation

Generate a full year of 15-minute load data from a partial history. The tool
uses curve-based multipliers to let analysts shape both the daily load curve
and the seasonal trend before synthesizing the missing values.

## Installation

The project targets Python 3.11+. Install the dependencies with pip:

```bash
pip install -r requirements.txt
```

Alternatively, use `pipx` or a virtual environment of your choice.

## Usage

1. Prepare a CSV or Excel file that contains a single column of 15-minute load
   data. If a timestamp column is not included, provide the `--start-date`
   argument so the script can infer the timestamps.
2. Optionally create a JSON configuration file describing the daily and
   monthly curve control points (see below for the format).
3. Run the generator:

```bash
python -m generate_load \
    --input path/to/partial_load.csv \
    --output path/to/full_year.xlsx \
    --start-date 2023-01-01 \
    --config curve_points.json
```

The resulting Excel file contains three sheets:

* `full_year_load` – the completed 15 minute series for the selected year.
* `daily_curve` – 96 rows showing the daily multipliers applied to each
  quarter-hour slot.
* `monthly_curve` – 12 rows showing the seasonal multipliers.

### Preview or smoke-test inside the repository

Use the `--preview` flag to inspect the synthesized series without writing an
Excel workbook. The repository ships with a 60-day sample input file that you
can use to experiment with the generator:

```bash
python -m generate_load --input examples/sample_input.csv --preview --preview-rows 8
```

When you are happy with the shape of the output, provide `--output` alongside
`--preview` (or drop `--preview` entirely) to produce the Excel workbook.

## Configuration file format

The JSON file uses two arrays: one for the daily curve (in minutes since
midnight) and one for the monthly curve. Example:

```json
{
  "daily_curve_points": [
    {"time": "06:00", "multiplier": 0.9},
    {"time": "09:00", "multiplier": 1.1},
    {"time": "18:00", "multiplier": 1.2}
  ],
  "monthly_curve_points": [
    {"month": 1, "multiplier": 0.8},
    {"month": 6, "multiplier": 1.05},
    {"month": 12, "multiplier": 0.95}
  ]
}
```

Daily control points can be as sparse or dense as necessary. A smooth cubic
Hermite spline is drawn through the points to compute a multiplier for each
15-minute slot. Monthly control points work the same way with one grab point
per month.

## Development

Run a basic syntax check before committing changes:

```bash
python -m compileall generate_load.py
```

Feel free to add unit tests under a `tests/` directory and execute them with
`pytest`.
