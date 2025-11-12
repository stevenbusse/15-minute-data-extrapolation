from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QSlider,
    QHBoxLayout,
    QMessageBox,
)
from PyQt5.QtCore import Qt
import sys
import json
from pathlib import Path
from ..generate_load import main as generate_load_main  # Adjust import based on your structure

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Load Profile Generator")
        self.setGeometry(100, 100, 600, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.init_ui()

    def init_ui(self):
        self.input_file_label = QLabel("Input File:")
        self.layout.addWidget(self.input_file_label)

        self.input_file_line_edit = QLineEdit()
        self.layout.addWidget(self.input_file_line_edit)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        self.layout.addWidget(self.browse_button)

        self.start_date_label = QLabel("Start Date (YYYY-MM-DD):")
        self.layout.addWidget(self.start_date_label)

        self.start_date_line_edit = QLineEdit()
        self.layout.addWidget(self.start_date_line_edit)

        self.year_label = QLabel("Year to Synthesize:")
        self.layout.addWidget(self.year_label)

        self.year_slider = QSlider(Qt.Horizontal)
        self.year_slider.setMinimum(2000)
        self.year_slider.setMaximum(2100)
        self.year_slider.setValue(2023)
        self.year_slider.setTickPosition(QSlider.TicksBelow)
        self.year_slider.setTickInterval(1)
        self.year_slider.valueChanged.connect(self.update_year_label)
        self.layout.addWidget(self.year_slider)

        self.year_value_label = QLabel("2023")
        self.layout.addWidget(self.year_value_label)

        self.generate_button = QPushButton("Generate Load Profile")
        self.generate_button.clicked.connect(self.generate_load_profile)
        self.layout.addWidget(self.generate_button)

    def browse_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)", options=options)
        if file_name:
            self.input_file_line_edit.setText(file_name)

    def update_year_label(self, value):
        self.year_value_label.setText(str(value))

    def generate_load_profile(self):
        input_file = self.input_file_line_edit.text()
        start_date = self.start_date_line_edit.text()
        year = self.year_slider.value()

        if not Path(input_file).is_file():
            QMessageBox.warning(self, "Input Error", "Please select a valid input file.")
            return

        # Call the existing generate_load functionality
        try:
            generate_load_main(input_file=input_file, start_date=start_date, year=year)
            QMessageBox.information(self, "Success", "Load profile generated successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())