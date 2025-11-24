# PyInstaller spec for building the GUI executable.
# Run from the generate-load-gui directory:
#   pyinstaller pyinstaller.spec

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

project_dir = Path(__file__).resolve().parent
entry_point = project_dir / "src" / "main.py"

matplotlib_datas = collect_data_files("matplotlib", include_py_files=True)
hidden_imports = collect_submodules("matplotlib") + collect_submodules("pandas")

a = Analysis(
    [str(entry_point)],
    pathex=[str(project_dir)],
    binaries=[],
    datas=matplotlib_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="generate-load-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="generate-load-gui",
)
