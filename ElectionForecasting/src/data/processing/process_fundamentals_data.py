# Refactored Data Processing Script

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import json
from ElectionForecasting.src.root import ROOT_DIR

def generate_config(raw_data_path, output_path):
    econ_data = Path(f'{raw_data_path}/dataland_economic_data_1984_2023.csv')
    if 'scenarios' in output_path:
        econ_data =  Path(f'{output_path}/dataland_economic_data_1984_2024_scenario.csv')
    print(econ_data)
    return {
        'economic_data_path': econ_data,
        'electoral_results_path': Path(f'{raw_data_path}/dataland_election_results_1984_2023.csv'),
        'electoral_calendar_path': Path(f'{raw_data_path}/dataland_electoral_calendar.csv'),
    }

def load_data(economic_data_path, electoral_results_path, electoral_calendar_path):
    economic_data = pd.read_csv(economic_data_path)
    economic_data['date'] = pd.to_datetime(economic_data['date'])
    electoral_results = pd.read_csv(electoral_results_path)
    next_cycle = electoral_results[electoral_results.year == np.max(electoral_results.year)].copy()
    next_cycle.iloc[:,4:] = np.nan
    next_cycle.year += 1
    electoral_results = pd.concat([electoral_results, next_cycle]).reset_index(drop=True)
    electoral_calendar = pd.read_csv(electoral_calendar_path)
    electoral_calendar['election_day'] = pd.to_datetime(electoral_calendar['election_day'])
    return economic_data, electoral_results, electoral_calendar


def process_data(
        economic_data, electoral_results,
        electoral_calendar, 
        output_path, save_output
    ):
    fundamentals = pd.merge(
        electoral_calendar, electoral_results, left_on='election_cycle', right_on='year'
    )
    fundamentals = pd.merge_asof(economic_data.sort_values('date'),
                                    fundamentals.sort_values('election_day'),
                                    left_on='date', right_on='election_day', direction='forward')
    # Create lagged variables for economic indicators
    lag_cols = []
    quarterly_cols=['year_on_year_gdp_pct_change', 'unemployment_rate',
                    'year_on_year_inflation', 'year_on_year_stock_mkt_pct_change',
                    'party_in_power', 'national_pop_vote_winner']
    for lag in [0, 1, 2, 3]:
        for col in quarterly_cols:
            lag_col=f'{col}_lag_{lag}q'
            lag_cols.append(lag_col)
            fundamentals[lag_col] = fundamentals[col].shift(lag)
        lag_date = f'date_lag_{lag}q'
        lag_cols.append(lag_date)
        fundamentals[lag_date] = fundamentals['date'].shift(lag)
    fundamentals = fundamentals.groupby('election_day').nth(-1)
    if save_output:
        time_for_change_path = Path(f'{output_path}/time_for_change/inputs.csv')
        print(f'Writing fundamentals data to {time_for_change_path}')
        time_for_change_path.parent.mkdir(exist_ok=True, parents=True)
        fundamentals.to_csv(time_for_change_path)
    return fundamentals


def process_state_results(electoral_results, output_path, save_output):
    state_results = {k: v for k, v in electoral_results.groupby('year')}
    cols = [c for c in electoral_results.columns if '_share' in c]
    if save_output:
        state_results_path = Path(f'{output_path}/state_results/')
        state_results_path.mkdir(exist_ok=True, parents=True)
        # Save processed data to CSV
        print(f'Writing state_results to {state_results_path}')
        for year, df in state_results.items():
            df.set_index('province')[cols].to_csv(state_results_path / f'{year}.csv')
    return state_results

def main(raw_data_path, output_path, save_output=False):
    config = generate_config(raw_data_path, output_path)
    economic_data, electoral_results, electoral_calendar = load_data(
        config['economic_data_path'],
        config['electoral_results_path'],
        config['electoral_calendar_path']
    )
    fundamentals_data = process_data(
        economic_data, electoral_results, electoral_calendar,
        output_path, save_output
    )
    state_results = process_state_results(
        electoral_results, output_path, save_output
    )
    return fundamentals_data, state_results

debug=False
if __name__ == '__main__':
    if debug:
        main(f'{ROOT_DIR}/data/dataland/', '{ROOT_DIR}/data/interim/', False)
        raise Warning
    parser = argparse.ArgumentParser(description='Data Processing Script')
    parser.add_argument('--raw_data_path', required=True, help='Path to the configuration file')
    parser.add_argument('--output_path', required=True, help='Path to the configuration file')
    parser.add_argument('--save_output', action='store_true', help='Flag to save output, omit for debugging')
    args = parser.parse_args()
    main(args.raw_data_path, args.output_path, args.save_output)