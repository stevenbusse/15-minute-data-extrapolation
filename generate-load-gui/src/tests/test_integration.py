import pytest
from src.gui.app import create_app
from src.generate_load import load_input_data, synthesize_full_year

@pytest.fixture
def app():
    app = create_app()
    yield app

def test_load_input_data(app):
    test_file_path = "path/to/test/historical_load.xlsx"
    start_date = "2023-01-01"
    series = load_input_data(test_file_path, start_date)
    assert not series.empty
    assert series.index.freq == '15T'

def test_synthesize_full_year(app):
    base_profile = pd.Series([1.0] * 96, index=pd.date_range("00:00", "23:45", freq="15min").time)
    daily_curve = pd.Series([1.0] * 96, index=pd.date_range("00:00", "23:45", freq="15min").time)
    monthly_curve = pd.Series([1.0] * 12, index=range(1, 13))
    monthly_adjustment = pd.Series([1.0] * 12, index=range(1, 13))
    year = 2023

    full_year = synthesize_full_year(base_profile, daily_curve, monthly_curve, monthly_adjustment, year)
    assert len(full_year) == 35040  # 365 days * 96 quarter-hours
    assert full_year.min() >= 0  # Ensure no negative values
    assert full_year.max() > 0  # Ensure there are positive values