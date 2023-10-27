#%%
import pandas as pd

import numpy as np
from ElectionForecasting.src.data.polls.poll_trends import main as poll_trends
from ElectionForecasting.src.data.polls.poll_forecasts import main as poll_forecasts
from ElectionForecasting.src.data.processing.process_fundamentals_data import main as process_fundamentals_data
from ElectionForecasting.src.data.processing.process_partisan_leans import main as process_partisan_leans
from ElectionForecasting.src.root import ROOT_DIR
from pathlib import Path

def process(
            interim_dir, sub_dir, polls_file, save=False,
            upper_range=2025, poll_weightings_config_file=None,
            forecasting_method='particle_filter'
        ):

    poll_trends(
            f"{interim_dir}/{sub_dir}/{polls_file}",
            f"{interim_dir}/{sub_dir}", save_outputs=save,
            configuration_file=poll_weightings_config_file
        )
    poll_forecasts(
            sub_dir, ROOT_DIR, save=save, upper_range=upper_range,
            forecasting_method=forecasting_method
        )
    process_fundamentals_data(
            f'{ROOT_DIR}/data/dataland/',
            f'{interim_dir}/{sub_dir}',
            save_output=save
        )
    process_partisan_leans(
            f'{interim_dir}/{sub_dir}/state_results/',
            f'{interim_dir}/{sub_dir}/pleans_new/',
            save_output=save
        )

if __name__ == '__main__':
    print(f'PROCESSING TRAINING DATA')
    process(f"{ROOT_DIR}/data/interim/", "", "../dataland/dataland_polls_1984_2023.csv", save=True, upper_range=2024)
    print(f'PROCESSED TRAINING DATA')
    for scenario in ['A', 'B', 'C', 'D', 'E']:
        print(f'PROCESSING SCENARIO {scenario}')
        process(f"{ROOT_DIR}/data/interim/", f"scenarios/{scenario}/", "dataland_polls_1984_2024_scenario.csv", save=True)