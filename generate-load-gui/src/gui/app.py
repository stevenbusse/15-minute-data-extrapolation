from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit
import sys
import json
from pathlib import Path
from generate_load import main  # Assuming main is the function to run the load generation logic

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