from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSplitter,
)
from PyQt5.QtCore import Qt
import sys
import os
import numpy as np
import json
import pandas as pd
from pathlib import Path

# Ensure repo root is importable so we can call functions directly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import generate_load as gl

import matplotlib
# enforce Qt5Agg backend early to avoid backend-selection/import errors in some venvs
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DraggableScatter:
    """Simple draggable vertical-only scatter.

    x are fixed; users may drag points up/down to change y values.
    A callback is invoked after each drag event.
    """

    def __init__(self, ax, x, y, callback=None, fmt="o", color="C0"):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.callback = callback

        # Draw initial scatter points
        self.scatter = ax.scatter(
            self.x,
            self.y,
            c=color,
            s=80,
            picker=5,
            zorder=10
        )

        # Internal drag state
        self.line = None
        self._ind = None

        # Event bindings
        self.cid_press = self.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

    def on_press(self, event):
        """Handle mouse press: find nearest point within click radius."""
        if event.inaxes != self.ax:
            return

        xy = np.column_stack([self.x, self.y])
        d = np.hypot(
            (xy[:, 0] - event.xdata),
            (xy[:, 1] - event.ydata)
        )
        idx = int(np.argmin(d))
        if d[idx] < 0.5:
            self._ind = idx

    def on_motion(self, event):
        """Drag point vertically, update scatter + callback."""
        if self._ind is None or event.inaxes != self.ax:
            return

        try:
            val = float(event.ydata)
        except Exception:
            return

        # Clamp range to reasonable multiplier bounds
        new_y = max(0.01, min(5.0, val))
        self.y[self._ind] = new_y

        # Update scatter graphics
        self.scatter.set_offsets(
            np.column_stack([self.x, self.y])
        )

        # Optional callback (e.g., to redraw a line)
        if self.callback:
            self.callback()

        self.canvas.draw_idle()

    def on_release(self, event):
        """End drag."""
        self._ind = None

    def update_y(self, y):
        """Update all Y values and redraw the scatter plot."""
        self.y = np.array(y, dtype=float)
        self.scatter.set_offsets(
            np.column_stack([self.x, self.y])
        )
        self.canvas.draw_idle()

    def get_points(self):
        """Return list of (x, y) tuples."""
        return list(zip(self.x.tolist(), self.y.tolist()))



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("15-minute Load Extrapolator — Curve Editor")
        self.resize(1100, 700)

        # Minimal UI: only input browse and Run button as requested
        main = QWidget()
        main_layout = QVBoxLayout(main)

        self.input_field = QLineEdit()
        btn_in = QPushButton("Browse input")
        btn_in.clicked.connect(self.browse_input)

        # preset buttons
        presets_layout = QHBoxLayout()
        btn_preset_home = QPushButton("Preset: Home")
        btn_preset_home.clicked.connect(lambda: self.load_preset("home"))
        btn_preset_office = QPushButton("Preset: Office")
        btn_preset_office.clicked.connect(lambda: self.load_preset("office"))
        btn_preset_multi = QPushButton("Preset: Multi")
        btn_preset_multi.clicked.connect(lambda: self.load_preset("multi"))
        presets_layout.addWidget(btn_preset_home)
        presets_layout.addWidget(btn_preset_office)
        presets_layout.addWidget(btn_preset_multi)

        btn_run = QPushButton("Run")
        btn_run.clicked.connect(self.run_and_export)

        main_layout.addWidget(QLabel("Input file:"))
        main_layout.addWidget(self.input_field)
        main_layout.addWidget(btn_in)
        main_layout.addLayout(presets_layout)

        # optional start date and config fields
        extra_row = QHBoxLayout()
        extra_row.addWidget(QLabel("Start date (YYYY-MM-DD, optional):"))
        self.start_field = QLineEdit()
        self.start_field.setPlaceholderText("e.g. 2024-01-01")
        extra_row.addWidget(self.start_field)

        extra_row.addWidget(QLabel("Config (optional):"))
        self.config_field = QLineEdit()
        extra_row.addWidget(self.config_field)
        btn_cfg = QPushButton("Browse config")
        def browse_cfg():
            p, _ = QFileDialog.getOpenFileName(self, "Select config JSON", "", "JSON (*.json);;All Files (*)")
            if p:
                self.config_field.setText(p)
        btn_cfg.clicked.connect(browse_cfg)
        extra_row.addWidget(btn_cfg)

        main_layout.addLayout(extra_row)
        main_layout.addStretch(1)
        main_layout.addWidget(btn_run)

        self.setCentralWidget(main)

        # hold loaded series (filled during run)
        self.loaded_series = None
        self.base_profile = None
        self.monthly_adjustment = None

        # add interactive editors below controls
        # create a small area with two plots
        fig1 = Figure(figsize=(6, 2.5))
        ax_daily = fig1.add_subplot(111)
        ax_daily.set_title("Daily multiplier (hours)")
        ax_daily.set_xlim(0, 24)
        ax_daily.set_ylim(0, 5.0)
        ax_daily.set_xlabel("Hour of day")
        ax_daily.set_ylabel("Multiplier")
        canvas1 = FigureCanvas(fig1)

        fig2 = Figure(figsize=(6, 2.5))
        ax_month = fig2.add_subplot(111)
        ax_month.set_title("Monthly multipliers")
        ax_month.set_xlim(1, 12)
        ax_month.set_ylim(0, 5.0)
        ax_month.set_xlabel("Month")
        ax_month.set_ylabel("Multiplier")
        canvas2 = FigureCanvas(fig2)

        main_layout.addWidget(canvas1)
        main_layout.addWidget(canvas2)

        # initial editor points
        daily_hours = np.array([0.0, 6.0, 12.0, 18.0, 23.75])
        daily_y = np.ones_like(daily_hours)
        month_x = np.arange(1, 13)
        month_y = np.ones_like(month_x, dtype=float)

        self.daily_editor = DraggableScatter(ax_daily, daily_hours, daily_y, callback=self.update_daily_curve, color="C1")
        self.monthly_editor = DraggableScatter(ax_month, month_x, month_y, callback=self.update_monthly_curve, color="C2")

        self.daily_curve_line, = ax_daily.plot([], [], lw=2, color="C0")
        self.monthly_curve_line, = ax_month.plot([], [], lw=2, color="C3")

        # draw initial curves
        self.update_daily_curve()
        self.update_monthly_curve()

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select input CSV/Excel", "", "CSV/Excel (*.csv *.xlsx *.xls);;All Files (*)")
        if path:
            self.input_field.setText(path)

    def update_daily_curve(self):
        """Redraw the daily curve line from the draggable editor points."""
        pts = self.daily_editor.get_points()
        if not pts:
            return
        # Build DailyCurvePoint list and evaluate a smooth daily curve via generator
        daily_pts = [gl.DailyCurvePoint(int(round(h * 60)), float(m)) for h, m in pts]
        series = gl.evaluate_daily_curve(daily_pts)
        # convert times to hours for plotting (0..24)
        hours = [t.hour + t.minute / 60.0 for t in series.index]
        self.daily_curve_line.set_data(hours, series.values)
        self.daily_curve_line.figure.canvas.draw_idle()

    def update_monthly_curve(self):
        """Redraw the monthly curve line from the draggable editor points."""
        pts = self.monthly_editor.get_points()
        if not pts:
            return
        # Use generator's cubic interpolation to build a smooth monthly curve
        # pts are (month, multiplier)
        sorted_pts = sorted(pts, key=lambda p: p[0])
        months, mults = zip(*sorted_pts)
        months = np.array(months, dtype=float)
        mults = np.array(mults, dtype=float)
        # ensure endpoints cover 1..12
        if months[0] > 1:
            months = np.insert(months, 0, 1.0)
            mults = np.insert(mults, 0, mults[0])
        if months[-1] < 12:
            months = np.append(months, 12.0)
            mults = np.append(mults, mults[-1])
        xs = np.linspace(1.0, 12.0, 240)
        ys = gl.cubic_hermite_interpolate(months, mults, xs)
        self.monthly_curve_line.set_data(xs, ys)
        self.monthly_curve_line.figure.canvas.draw_idle()


    def run_and_export(self):
        inp = self.input_field.text().strip()
        if not inp:
            QMessageBox.warning(self, "No input", "Choose an input file first")
            return

        # parse optional start date from UI
        start_date = None
        sd_text = self.start_field.text().strip() if hasattr(self, 'start_field') else ""
        if sd_text:
            try:
                start_date = pd.to_datetime(sd_text)
            except Exception:
                QMessageBox.warning(self, "Start date", "Start date not recognised — expected YYYY-MM-DD. Ignoring.")
                start_date = None

        try:
            series = gl.load_input_data(Path(inp), start_date)
        except Exception:
            try:
                series = gl.load_input_data_robust(Path(inp), start_date)
                QMessageBox.information(self, "Input cleaned",
                                        "Input timestamps were irregular or missing — the file was automatically cleaned and resampled to 15-minute spacing.")
            except Exception as e2:
                QMessageBox.critical(self, "Load error", f"Failed to load input file:\n{e2}")
                return

        try:
            base_profile = gl.build_time_of_day_profile(series)
            monthly_adjustment = gl.compute_monthly_adjustment(series, base_profile)
            # build curves from current editor points
            daily_pts = [gl.DailyCurvePoint(int(round(h * 60)), float(m)) for h, m in self.daily_editor.get_points()]
            monthly_pts = [gl.MonthlyCurvePoint(int(m), float(v)) for m, v in self.monthly_editor.get_points()]
            daily_curve = gl.evaluate_daily_curve(daily_pts)
            monthly_curve = gl.evaluate_monthly_curve(monthly_pts)
            year = gl.infer_start_year(series)

            full = gl.synthesize_full_year(base_profile, daily_curve, monthly_curve, monthly_adjustment, year)

            inp_path = Path(inp)
            out_path = inp_path.with_name(inp_path.stem + "_extrapolated.xlsx")
            if out_path.exists():
                resp = QMessageBox.question(self, "Overwrite?", f"{out_path}\n\nFile exists. Overwrite?", QMessageBox.Yes | QMessageBox.No)
                if resp != QMessageBox.Yes:
                    return

            gl.write_output(out_path, full, daily_curve, monthly_curve)
            QMessageBox.information(self, "Done", f"Wrote output to {out_path}\nPoints: {len(full)}\nMin: {full.min():.3f}\nMax: {full.max():.3f}")
        except Exception as e:
            QMessageBox.critical(self, "Generation error", str(e))

    def load_config_into_editors(self):
        cfg = self.config_field.text().strip()
        if not cfg:
            QMessageBox.warning(self, "No config", "Choose a config JSON first")
            return
        try:
            with open(cfg, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            QMessageBox.critical(self, "Config error", f"Failed to read JSON: {e}")
            return

        # daily points expected: list of {time: "HH:MM", multiplier: float}
        dpts = data.get("daily_curve_points", [])
        if dpts:
            mult = []
            for item in dpts[:5]:
                try:
                    minutes = gl.parse_time_string(item["time"])
                    mult.append(float(item.get("multiplier", 1.0)))
                except Exception:
                    continue
            if len(mult) >= 1:
                x = self.daily_editor.x
                ys = (mult + [1.0] * (len(x) - len(mult)))[:len(x)]
                self.daily_editor.update_y(ys)
                self.update_daily_curve()

        mpts = data.get("monthly_curve_points", [])
        if mpts:
            multm = []
            for item in mpts:
                try:
                    multm.append(float(item.get("multiplier", 1.0)))
                except Exception:
                    continue
            if len(multm) >= 1:
                ys = (multm + [1.0] * (12 - len(multm)))[:12]
                self.monthly_editor.update_y(ys)
                self.update_monthly_curve()

        QMessageBox.information(self, "Config loaded", "Config loaded into editors (if values present).")


    def save_current_config(self):
        cfg = self.config_field.text().strip()
        if not cfg:
            cfg, _ = QFileDialog.getSaveFileName(self, "Save config JSON", "", "JSON (*.json);;All Files (*)")
            if not cfg:
                return
            self.config_field.setText(cfg)

        daily = [
            {"time": f"{int(h):02d}:{int((h-int(h))*60):02d}", "multiplier": float(v)}
            for h, v in self.daily_editor.get_points()
        ]
        monthly = [
            {"month": int(m), "multiplier": float(v)}
            for m, v in self.monthly_editor.get_points()
        ]
        payload = {"daily_curve_points": daily, "monthly_curve_points": monthly}
        try:
            with open(cfg, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            QMessageBox.information(self, "Saved", f"Config saved to {cfg}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def load_preset(self, name: str):
        presets = {
            "home": {
                "daily": [(0, 0.9), (6, 0.7), (12, 1.0), (18, 1.3), (23.75, 1.0)],
                "monthly": [(i, 1.0) for i in range(1, 13)],
            },
            "office": {
                "daily": [(0, 0.5), (6, 0.6), (12, 1.2), (18, 0.6), (23.75, 0.5)],
                "monthly": [(i, 1.0) for i in range(1, 13)],
            },
            "multi": {
                "daily": [(0, 0.8), (6, 0.9), (12, 1.1), (18, 1.0), (23.75, 0.9)],
                "monthly": [(i, 1.0 + 0.05 * ((i%12)//3)) for i in range(1, 13)],
            },
        }
        p = presets.get(name)
        if not p:
            return
        dy = [y for _, y in p["daily"]]
        my = [m for _, m in p["monthly"]]
        # update editors and redraw smooth curves
        self.daily_editor.update_y(dy)
        self.update_daily_curve()
        self.monthly_editor.update_y(my)
        self.update_monthly_curve()
        QMessageBox.information(self, "Preset loaded", f"Loaded preset: {name}")


if __name__ == "__main__":
    from pathlib import Path

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
