from pathlib import Path
import pandas as pd
import json

def read_excel_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    return df

def write_excel_file(file_path: Path, data: pd.DataFrame) -> None:
    data.to_excel(file_path, index=False)

def read_json_file(file_path: Path) -> dict:
    with file_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json_file(file_path: Path, data: dict) -> None:
    with file_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)