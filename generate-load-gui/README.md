# Generate Load GUI

This project provides a user-friendly graphical interface for generating synthetic 15-minute load profiles based on historical load data. The application allows users to easily input their data, adjust daily and monthly load curves, and generate a full year's worth of load data.

## Features

- Load historical quarter-hour load data from CSV or Excel files.
- Adjust daily load profiles using customizable multipliers.
- Modify monthly load profiles with user-defined multipliers.
- Generate a complete 15-minute load profile for an entire year.
- Export the generated load data to an Excel file.
- Preview the generated data directly within the application.

## Project Structure

- `src/main.py`: Entry point for the application, initializes the GUI.
- `src/generate_load.py`: Contains the logic for generating synthetic load profiles.
- `src/gui/`: Directory containing all GUI-related files.
  - `app.py`: Initializes the main application window.
  - `main_window.py`: Defines the layout and components of the main window.
  - `controllers.py`: Handles user interactions and input validation.
  - `widgets.py`: Defines reusable custom widgets.
- `src/config/example_curve_points.json`: Example configuration for daily and monthly curve points.
- `src/utils/file_handlers.py`: Utility functions for file input and output.
- `src/tests/test_integration.py`: Integration tests for the application.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd generate-load-gui
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

Follow the on-screen instructions to load your data, adjust the curves, and generate the load profile.

## Building a standalone executable (Windows)

See `BUILD_EXE.md` for a short guide using PyInstaller. In short:

- Install the GUI requirements (use a virtualenv).
- Install PyInstaller and run `pyinstaller pyinstaller.spec` from the `generate-load-gui` folder.
- Test the resulting `dist/generate-load-gui` folder on a clean Windows machine and include Qt platform plugins if needed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.   