Building a Windows executable with PyInstaller
===============================================

The GUI is already configured with a `pyinstaller.spec` file that pulls in the
PyQt5/Matplotlib dependencies that PyInstaller sometimes misses (matplotlib
data files and plotting backends). Follow the steps below on a Windows machine.

1. Create a clean virtual environment (PowerShell shown)

   ```powershell
   cd .\generate-load-gui
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   `requirements.txt` already includes PyInstaller so there is no extra step.

2. Build the executable

   ```powershell
   pyinstaller pyinstaller.spec
   ```

   - If you need to rebuild from scratch, delete the `build/` and `dist/`
     folders first (`Remove-Item build,dist -Recurse -Force`).
   - The spec uses the Matplotlib + pandas helper functions from
     `PyInstaller.utils.hooks` so you should no longer see missing backend /
     font errors.

3. Locate the generated app

   - PyInstaller creates a folder app by default (faster startup than single
     file). After a successful build you will have:
     `generate-load-gui/dist/generate-load-gui/generate-load-gui.exe`
   - Zip that folder if you want to send it to colleagues. Everything inside
     the folder (Qt plugins, matplotlib data, DLLs) must stay together.

4. Test locally

   - Still inside your virtual environment, run the exe directly to make sure
     it launches:
     `.\dist\generate-load-gui\generate-load-gui.exe`
   - Verify that drag-and-drop works, presets load, and the generated Excel
     file ends up next to your sample input. PyInstaller bundles Python so it
     should run even if you deactivate the venv—which mimics a non-technical
     user's machine.

5. Test on a clean machine (recommended)

   - Copy the entire `generate-load-gui` folder from `dist/` to a Windows VM or
     another PC that does not have Python installed.
   - Run the exe from there; verify no missing DLL dialogs appear.

Optional single-file build
--------------------------
- If you prefer a single `.exe`, change the `EXE()` call in `pyinstaller.spec`
  to `onefile=True`. Startup becomes slower because PyInstaller has to unpack
  files to a temporary directory.
- Single-file builds sometimes need the environment variable
  `QT_QPA_PLATFORM_PLUGIN_PATH` set manually. For most users, the folder-based
  (default) build is more reliable.

Troubleshooting
---------------
- **Missing Qt platform plugin**: make sure you are running the exe from within
  the generated folder so the `platforms/` plugins remain next to it.
- **Matplotlib backend issues**: the spec already list matplotlib data files
  and we set the backend to `Qt5Agg` in `src/main.py`. If you customized the
  code later, keep the backend line before Matplotlib is imported.
