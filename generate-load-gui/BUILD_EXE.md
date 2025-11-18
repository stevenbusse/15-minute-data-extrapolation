Building a Windows executable (PyInstaller)

Prerequisites:
- Install the GUI requirements into a clean virtual environment.
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -r requirements.txt
  python -m pip install pyinstaller

Basic build command (from the `generate-load-gui` folder):

  pyinstaller pyinstaller.spec

Notes:
- The spec is a minimal template. You may need to add hidden imports for PyQt5 or matplotlib plugins.
- Test the exe on a clean Windows machine; include DLLs if needed (matplotlib backends, QT platform plugins).
- If you want a single-file exe, set `onefile=True` in the EXE() call and be aware of a longer startup time.

Troubleshooting:
- Missing Qt platform plugin: ensure `--add-data` includes the `PyQt5\Qt\plugins` folder or use the `--add-binary` option.
- If matplotlib backend issues arise, try setting the backend to `Qt5Agg` explicitly before imports or include matplotlib data files via `--add-data`.
