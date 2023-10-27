
import pandas as pd
from pathlib import Path

def load_economic_data(root_dir: Path) -> pd.DataFrame:
    economic_data_path = root_dir / "data/dataland/dataland_economic_data_1984_2023.csv"
    economic_data = pd.read_csv(economic_data_path)
    economic_data['date'] = pd.to_datetime(economic_data['date'])
    return economic_data

def load_electoral_results(root_dir: Path) -> pd.DataFrame:
    electoral_results_path = root_dir / "data/dataland/dataland_election_results_1984_2023.csv"
    electoral_results = pd.read_csv(electoral_results_path)
    return electoral_results

def load_electoral_calendar(root_dir: Path) -> pd.DataFrame:
    electoral_calendar_path = root_dir / "data/dataland/dataland_electoral_calendar.csv"
    electoral_calendar = pd.read_csv(electoral_calendar_path)
    electoral_calendar['election_day'] = pd.to_datetime(electoral_calendar['election_day'])
    return electoral_calendar
