from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit
import sys
import json
from pathlib import Path
import os, sys, subprocess

def run_cli(input_path, output_path, year=None, start_date=None, config_path=None, preview=True):
    # Resolve the path to the CLI script we want to call
    # Here we call the *root* CLI (recommended). If you prefer the GUI copy, point at src/generate_load.py instead.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cli_script = os.path.join(repo_root, "generate_load.py")  # root CLI that you already verified works

    cmd = [sys.executable, cli_script, "--input", input_path, "--output", output_path]
    if year:
        cmd += ["--year", str(year)]
    if start_date:
        cmd += ["--start-date", start_date]
    if config_path:
        cmd += ["--config", config_path]
    if preview:
        cmd += ["--preview"]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


class LoadProfileApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Load Profile Generator")
        self.setGeometry(100, 100, 600, 400)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.input_file_label = QLabel("Input File:")
        self.input_file_field = QLineEdit()
        self.browse_input_button = QPushButton("Browse")
        self.browse_input_button.clicked.connect(self.browse_input_file)

        self.output_file_label = QLabel("Output File:")
        self.output_file_field = QLineEdit()
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self.browse_output_file)

        self.start_button = QPushButton("Generate Load Profile")
        self.start_button.clicked.connect(self.generate_load_profile)

        layout.addWidget(self.input_file_label)
        layout.addWidget(self.input_file_field)
        layout.addWidget(self.browse_input_button)
        layout.addWidget(self.output_file_label)
        layout.addWidget(self.output_file_field)
        layout.addWidget(self.browse_output_button)
        layout.addWidget(self.start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def browse_input_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)", options=options)
        if file_name:
            self.input_file_field.setText(file_name)

    def browse_output_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "Excel Files (*.xlsx)", options=options)
        if file_name:
            self.output_file_field.setText(file_name)

    def generate_load_profile(self):
        input_file = self.input_file_field.text()
        output_file = self.output_file_field.text()

        if not input_file or not output_file:
            # Handle error: both fields must be filled
            return

        # Call the main function from generate_load.py with the appropriate arguments
        main(input_file, output_file)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoadProfileApp()
    window.show()
    sys.exit(app.exec_())