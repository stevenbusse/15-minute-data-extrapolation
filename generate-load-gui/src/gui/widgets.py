from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QFileDialog, QFormLayout

class FileInputWidget(QWidget):
    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        self.layout = QFormLayout(self)
        
        self.label = QLabel(label_text)
        self.file_input = QLineEdit(self)
        self.browse_button = QPushButton("Browse", self)
        
        self.browse_button.clicked.connect(self.browse_file)
        
        self.layout.addRow(self.label, self.file_input)
        self.layout.addRow(self.browse_button)

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_name:
            self.file_input.setText(file_name)


class CurveAdjustmentWidget(QWidget):
    def __init__(self, label_text: str, min_value: int, max_value: int, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        self.label = QLabel(label_text)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)