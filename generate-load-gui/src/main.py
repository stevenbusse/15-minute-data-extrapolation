from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QMessageBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QDateEdit,
)
from PyQt5.QtCore import Qt, QDate, pyqtSignal
from PyQt5.QtGui import QFont
import sys
import os
import numpy as np
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


class FileDropWidget(QFrame):
    """Minimal drag-and-drop area that doubles as a browse button."""

    fileSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FileDrop")
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self._path = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignCenter)

        self.title = QLabel("Load profile")
        self.title.setAlignment(Qt.AlignCenter)
        self.hint = QLabel("Drop CSV/XLSX here\nor click to browse")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setObjectName("FileDropHint")
        self.path_label = QLabel("No file selected")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setObjectName("FileDropPath")

        layout.addWidget(self.title)
        layout.addWidget(self.hint)
        layout.addWidget(self.path_label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.set_path(url.toLocalFile())
                event.acceptProposedAction()
                return
        event.ignore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.open_file_dialog()
        super().mousePressEvent(event)

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input CSV/Excel",
            "",
            "CSV/Excel (*.csv *.xlsx *.xls);;All Files (*)",
        )
        if path:
            self.set_path(path)

    def set_path(self, path: str):
        self._path = path
        name = os.path.basename(path)
        self.path_label.setText(name if name else "No file selected")
        self.fileSelected.emit(path)

    def current_path(self) -> str:
        return self._path


class DraggableScatter:
    """Simple draggable vertical-only scatter.

    x are fixed; users may drag points up/down to change y values.
    A callback is invoked after each drag event.
    """

    def __init__(self, ax, x, y, callback=None, fmt="o", color="C0", y_min=0.01, y_max=5.0):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.callback = callback
        self.y_min = y_min
        self.y_max = y_max

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
        if event.inaxes != self.ax or event.x is None or event.y is None:
            return

        # measure distance in screen pixels for stable picking across scales
        xy_disp = self.ax.transData.transform(np.column_stack([self.x, self.y]))
        click = np.array([event.x, event.y])
        d = np.linalg.norm(xy_disp - click, axis=1)
        idx = int(np.argmin(d))
        if d[idx] <= 12.0:  # 12 px tolerance
            self._ind = idx

    def on_motion(self, event):
        """Drag point vertically, update scatter + callback."""
        if self._ind is None or event.inaxes != self.ax:
            return

        try:
            val = float(event.ydata)
        except Exception:
            return

        # Clamp range to reasonable bounds
        new_y = val
        if self.y_min is not None:
            new_y = max(self.y_min, new_y)
        if self.y_max is not None:
            new_y = min(self.y_max, new_y)
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
        arr = np.array(y, dtype=float)
        if self.y_min is not None:
            arr = np.maximum(arr, self.y_min)
        if self.y_max is not None:
            arr = np.minimum(arr, self.y_max)
        self.y = arr
        self.scatter.set_offsets(
            np.column_stack([self.x, self.y])
        )
        self.canvas.draw_idle()

    def set_limits(self, y_min=None, y_max=None):
        """Set drag bounds for y values."""
        self.y_min = y_min
        self.y_max = y_max

    def get_points(self):
        """Return list of (x, y) tuples."""
        return list(zip(self.x.tolist(), self.y.tolist()))



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("15-minute Load Extrapolator")
        self.resize(1200, 820)
        self.selected_input_path = ""

        base_font = QFont("Segoe UI", 10)
        self.setFont(base_font)
        self.setStyleSheet(
            """
            QWidget {
                font-family: 'Segoe UI';
                font-size: 11pt;
                color: #1f2430;
            }
            QFrame#Card {
                background: #f7f8fb;
                border: 1px solid #d9dce5;
                border-radius: 14px;
            }
            QFrame#FileDrop {
                border: 1px dashed #9ba5be;
                border-radius: 14px;
                background: #fff;
            }
            QLabel#FileDropHint {
                color: #6b7287;
                font-size: 10pt;
            }
            QLabel#FileDropPath {
                color: #3c3f51;
                font-weight: 600;
            }
            QPushButton {
                border-radius: 8px;
                padding: 10px 18px;
                background-color: #3c7ae5;
                color: white;
                border: none;
            }
            QPushButton#Secondary {
                background-color: #eef1f7;
                color: #1f2430;
            }
            QPushButton:disabled {
                background-color: #a5b4d4;
            }
            QComboBox, QDateEdit {
                border: 1px solid #c9ccdb;
                border-radius: 8px;
                padding: 6px 10px;
                background: white;
            }
            QListWidget {
                border: 1px solid #e1e2ea;
                border-radius: 10px;
                padding: 6px;
                background: white;
            }
            """
        )

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)
        self.setCentralWidget(main)

        # hold loaded series (filled during run)
        self.loaded_series = None
        self.base_profile = None
        self.monthly_adjustment = None

        # add interactive editors below controls
        # create a small area with two plots
        graphs_frame = QFrame()
        graphs_frame.setObjectName("Card")
        graphs_layout = QHBoxLayout(graphs_frame)
        graphs_layout.setContentsMargins(12, 12, 12, 12)
        graphs_layout.setSpacing(12)

        fig1 = Figure(figsize=(5.5, 2.8))
        ax_daily = fig1.add_subplot(111)
        ax_daily.set_title("Daily multiplier (hours)", loc="left")
        ax_daily.set_xlim(0, 24)
        ax_daily.set_ylim(0, 3.0)
        ax_daily.set_xlabel("Hour of day")
        ax_daily.set_ylabel("Multiplier")

        fig2 = Figure(figsize=(5.5, 2.8))
        ax_month = fig2.add_subplot(111)
        ax_month.set_title("Monthly multipliers", loc="left")
        ax_month.set_xlim(1, 12)
        ax_month.set_ylim(0, 3.0)
        ax_month.set_xlabel("Month")
        ax_month.set_ylabel("Multiplier")

        for ax in (ax_daily, ax_month):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="#e6e8f2", linewidth=0.6, linestyle="-", alpha=0.8)

        canvas1 = FigureCanvas(fig1)
        canvas2 = FigureCanvas(fig2)
        self.ax_daily = ax_daily
        self.ax_month = ax_month

        graphs_layout.addWidget(canvas1, 1)

        month_container = QWidget()
        month_layout = QVBoxLayout(month_container)
        month_layout.setContentsMargins(0, 0, 0, 0)
        month_layout.setSpacing(6)
        self.month_mode_combo = QComboBox()
        self.month_mode_combo.addItem("Multiplier", "multiplier")
        self.month_mode_combo.addItem("kWh target", "kwh")
        self.month_mode_combo.currentIndexChanged.connect(self.on_month_mode_changed)
        month_layout.addWidget(self.month_mode_combo, 0, Qt.AlignLeft)
        month_layout.addWidget(canvas2, 1)

        graphs_layout.addWidget(month_container, 1)
        main_layout.addWidget(graphs_frame, 2)

        controls = QFrame()
        controls.setObjectName("Card")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(12)

        file_row = QHBoxLayout()
        file_row.setSpacing(12)
        self.file_drop = FileDropWidget()
        self.file_drop.fileSelected.connect(self.handle_file_selected)
        file_row.addWidget(self.file_drop, 2)

        preset_col = QVBoxLayout()
        preset_col.setSpacing(6)
        preset_label = QLabel("Curve presets")
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Choose preset…", "")
        self.preset_combo.addItem("Home", "home")
        self.preset_combo.addItem("Office", "office")
        self.preset_combo.addItem("Multi-use", "multi")
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        preset_col.addWidget(preset_label)
        preset_col.addWidget(self.preset_combo)
        file_row.addLayout(preset_col, 1)

        controls_layout.addLayout(file_row)

        start_row = QHBoxLayout()
        start_row.setSpacing(12)
        start_label = QLabel("Start date")
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        start_year = QDate.currentDate().year()
        self.start_date_edit.setDate(QDate(start_year, 1, 1))
        self.start_date_edit.setDisplayFormat("MM/dd/yyyy")
        start_row.addWidget(start_label)
        start_row.addWidget(self.start_date_edit)
        start_row.addStretch(1)

        self.run_button = QPushButton("Generate extrapolated year")
        self.run_button.clicked.connect(self.run_and_export)
        start_row.addWidget(self.run_button)
        controls_layout.addLayout(start_row)

        log_label = QLabel("Recent files")
        self.file_log = QListWidget()
        self.file_log.setMaximumHeight(120)
        self.file_log.itemDoubleClicked.connect(self.on_recent_file_activated)
        controls_layout.addWidget(log_label)
        controls_layout.addWidget(self.file_log)

        main_layout.addWidget(controls, 1)

        # initial editor points
        daily_hours = np.array([0.0, 6.0, 12.0, 18.0, 23.75])
        daily_y = np.ones_like(daily_hours)
        month_x = np.arange(1, 13)
        month_y = np.ones_like(month_x, dtype=float)

        self.month_mode = "multiplier"
        self.month_multiplier_values = month_y.copy()
        # start kWh targets at 2,000 kWh/month unless we can infer later
        self.month_kwh_values = np.full_like(month_y, 2000.0, dtype=float)

        self.daily_baseline = ax_daily.axhline(1.0, color="#b6bccf", linestyle="--", linewidth=1.0, zorder=1)
        self.monthly_baseline = ax_month.axhline(1.0, color="#b6bccf", linestyle="--", linewidth=1.0, zorder=1)

        self.daily_editor = DraggableScatter(
            ax_daily, daily_hours, daily_y, callback=self.update_daily_curve, color="#ff8c42"
        )
        self.monthly_editor = DraggableScatter(
            ax_month, month_x, month_y, callback=self.update_monthly_curve, color="#5a9bd4", y_min=0.0, y_max=5.0
        )

        self.daily_curve_line, = ax_daily.plot([], [], lw=2.4, color="#1f4f8d")
        self.monthly_curve_line, = ax_month.plot([], [], lw=2.4, color="#8c4cd9")

        # draw initial curves
        self.update_daily_curve()
        self.update_monthly_curve()
        self.run_button.setEnabled(False)
        self.statusBar().showMessage("Drop a load profile to begin")

    def _update_run_enabled(self):
        """Enable run when a file is selected or when kWh mode is active."""
        self.run_button.setEnabled(bool(self.selected_input_path) or self.month_mode == "kwh")

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
        values = series.values
        self.daily_curve_line.set_data(hours, values)
        self._set_axis_limits(self.ax_daily, values)
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
        # persist current edits per mode
        if self.month_mode == "multiplier":
            self.month_multiplier_values = mults.copy()
        else:
            self.month_kwh_values = mults.copy()
        self.monthly_curve_line.set_data(xs, ys)
        if self.month_mode == "kwh":
            self._set_axis_limits(self.ax_month, ys, min_floor=4000, max_ceiling=12000)
        else:
            self._set_axis_limits(self.ax_month, ys)
        self.monthly_curve_line.figure.canvas.draw_idle()

    def _set_axis_limits(self, ax, values, min_floor=None, max_ceiling=None):
        if values is None or len(values) == 0:
            return
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size == 0:
            return
        peak = float(np.nanmax(finite_vals))
        top = peak * 1.2 if peak > 0 else (max_ceiling or 1.0)
        if min_floor is not None:
            top = max(min_floor, top)
        if max_ceiling is not None:
            top = min(max_ceiling, top)
        ax.set_ylim(0, top)

    def _get_month_values(self) -> np.ndarray:
        pts = sorted(self.monthly_editor.get_points(), key=lambda p: p[0])
        if not pts:
            return np.ones(12, dtype=float)
        return np.array([v for _, v in pts], dtype=float)

    def estimate_month_targets(self) -> np.ndarray:
        """Rough guess for kWh targets from loaded data."""
        if self.loaded_series is not None and len(self.loaded_series) > 0:
            # convert 15-min power values to energy by multiplying by 0.25h
            month_energy = self.loaded_series.resample("M").sum() * 0.25
            month_energy.index = month_energy.index.month
            values = np.full(12, float(month_energy.mean() or 1.0))
            for m in range(1, 13):
                if m in month_energy.index:
                    values[m - 1] = float(month_energy.loc[m])
            return values
        return np.full(12, 2000.0, dtype=float)

    def on_month_mode_changed(self, _index=None):
        # Preserve current edits before switching modes
        current_vals = self._get_month_values()
        if self.month_mode == "multiplier":
            self.month_multiplier_values = current_vals
        else:
            self.month_kwh_values = current_vals

        self.month_mode = self.month_mode_combo.currentData()
        if self.month_mode == "kwh":
            if self.month_kwh_values is None:
                self.month_kwh_values = self.estimate_month_targets()
            next_vals = self.month_kwh_values
            self.ax_month.set_ylabel("kWh target")
            self._set_axis_limits(self.ax_month, next_vals, min_floor=4000, max_ceiling=12000)
            # allow pulling beyond 10k; clamp drag at a higher bound but not too loose
            self.monthly_editor.set_limits(0.0, 12000.0)
            self.statusBar().showMessage("Editing monthly kWh targets")
        else:
            next_vals = self.month_multiplier_values
            self.ax_month.set_ylabel("Multiplier")
            self.ax_month.set_ylim(0, 5.0)
            self.monthly_editor.set_limits(0.0, 5.0)
            self.statusBar().showMessage("Editing monthly multipliers")

        self.monthly_editor.update_y(next_vals)
        self.update_monthly_curve()
        self._update_run_enabled()

    def handle_file_selected(self, path: str):
        path = path.strip()
        self.selected_input_path = path
        has_path = bool(path)
        self._update_run_enabled()
        if not has_path:
            self.statusBar().showMessage("Please select a load profile (or switch to kWh mode)")
            return

        display_name = Path(path).name
        # remove duplicates
        for i in range(self.file_log.count()):
            item = self.file_log.item(i)
            if item.data(Qt.UserRole) == path:
                self.file_log.takeItem(i)
                break

        item = QListWidgetItem(display_name)
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self.file_log.insertItem(0, item)
        self.file_log.setCurrentItem(item)
        self.statusBar().showMessage(f"Ready — {display_name}")

    def on_recent_file_activated(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self.file_drop.set_path(path)

    def on_preset_changed(self, index: int):
        key = self.preset_combo.itemData(index)
        if key:
            self.load_preset(str(key))

    def run_and_export(self):
        inp = self.selected_input_path.strip()
        start_date = pd.Timestamp(self.start_date_edit.date().toPyDate())

        # Load input if provided; allow kWh-only mode without a file.
        if inp:
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
        else:
            if self.month_mode != "kwh":
                QMessageBox.warning(self, "No input", "Choose an input file first (required in multiplier mode)")
                return
            # No file and kWh mode: fabricate a neutral series placeholder.
            series = None

        try:
            # build curves from current editor points
            daily_pts = [gl.DailyCurvePoint(int(round(h * 60)), float(m)) for h, m in self.daily_editor.get_points()]
            daily_curve = gl.evaluate_daily_curve(daily_pts)
            month_vals = self._get_month_values()
            if self.month_mode == "multiplier":
                monthly_pts = [
                    gl.MonthlyCurvePoint(int(idx + 1), float(val)) for idx, val in enumerate(month_vals)
                ]
                monthly_curve = gl.evaluate_monthly_curve(monthly_pts)
                monthly_kwh_targets = None
            else:
                monthly_curve = pd.Series(1.0, index=pd.Index(range(1, 13), name="month"))
                monthly_kwh_targets = {int(idx + 1): float(val) for idx, val in enumerate(month_vals)}

            if series is None:
                # Neutral base and adjustment when no input file in kWh mode.
                base_profile = pd.Series(1.0, index=daily_curve.index)
                monthly_adjustment = pd.Series(1.0, index=pd.RangeIndex(1, 13))
                year = start_date.year
            else:
                base_profile = gl.build_time_of_day_profile(series)
                monthly_adjustment = gl.compute_monthly_adjustment(series, base_profile)
                year = gl.infer_start_year(series)

            full = gl.synthesize_full_year(
                base_profile,
                daily_curve,
                monthly_curve,
                monthly_adjustment,
                year,
                monthly_kwh_targets=monthly_kwh_targets,
            )

            if self.month_mode == "kwh":
                baseline_energy = gl.estimate_monthly_baseline_energy(
                    base_profile=base_profile,
                    daily_curve=daily_curve,
                    monthly_adjustment=monthly_adjustment,
                    year=year,
                    use_proportional_daily=True,
                )
                implied = []
                for month_idx in range(1, 13):
                    target = monthly_kwh_targets.get(month_idx, 0.0)
                    base_val = float(baseline_energy.get(month_idx, np.nan))
                    if base_val and base_val != 0:
                        implied.append(target / base_val)
                    else:
                        implied.append(1.0)
                monthly_curve = pd.Series(implied, index=pd.Index(range(1, 13), name="month"))

            if inp:
                inp_path = Path(inp)
                out_path = inp_path.with_name(inp_path.stem + "_extrapolated.xlsx")
            else:
                out_dir = Path("out")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "kwh_extrapolated.xlsx"
            if out_path.exists():
                resp = QMessageBox.question(self, "Overwrite?", f"{out_path}\n\nFile exists. Overwrite?", QMessageBox.Yes | QMessageBox.No)
                if resp != QMessageBox.Yes:
                    return

            gl.write_output(out_path, full, daily_curve, monthly_curve)
            QMessageBox.information(self, "Done", f"Wrote output to {out_path}\nPoints: {len(full)}\nMin: {full.min():.3f}\nMax: {full.max():.3f}")
        except Exception as e:
            QMessageBox.critical(self, "Generation error", str(e))

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
        self.statusBar().showMessage(f"Preset '{name}' applied")


if __name__ == "__main__":
    from pathlib import Path

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
