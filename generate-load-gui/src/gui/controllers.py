from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QMessageBox
from PyQt5.QtCore import pyqtSlot
import json
from pathlib import Path
from ..generate_load import main as generate_load_main  # Adjust import based on your structure

class LoadProfileController:
    def __init__(self, main_window):
        self.main_window = main_window
        self.setup_ui()

    def setup_ui(self):
        self.main_window.setWindowTitle("Load Profile Generator")
        layout = QVBoxLayout()

        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select input file...")
        layout.addWidget(self.file_input)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        layout.addWidget(self.browse_button)

        self.start_button = QPushButton("Generate Load Profile")
        self.start_button.clicked.connect(self.generate_load_profile)
        layout.addWidget(self.start_button)

        self.main_window.setLayout(layout)

    @pyqtSlot()
    def browse_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self.main_window, "Select Input File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)", options=options)
        if file_name:
            self.file_input.setText(file_name)

    @pyqtSlot()
    def generate_load_profile(self):
        input_file = self.file_input.text()
        if not input_file:
            QMessageBox.warning(self.main_window, "Input Error", "Please select an input file.")
            return

        try:
            # Call the main function from generate_load.py with appropriate arguments
            generate_load_main(input_file)
            QMessageBox.information(self.main_window, "Success", "Load profile generated successfully.")
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"An error occurred: {str(e)}")