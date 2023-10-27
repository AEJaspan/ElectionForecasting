#%%
import pandas as pd
import argparse
import pickle
from pathlib import Path
import logging
from ElectionForecasting.src.utils.general import configure_logging
from ElectionForecasting.src.root import ROOT_DIR
from ElectionForecasting.src.data.polls.forecasting_methods import (
    particle_filter, pymc_gam, statsmodels_gam
)
configure_logging()
electoral_calendar = pd.read_csv(f'{ROOT_DIR}/data/dataland/dataland_electoral_calendar.csv')
electoral_calendar['election_day'] = pd.to_datetime(electoral_calendar['election_day'])

forecasting_methods = dict(
    particle_filter={
            'function': particle_filter, 'type': 'PF', 'args': {}
        },
    pymc_gam = {
            'function': pymc_gam, 'type': 'PyMCGAM',
            'args': {'DoF': 15, 'degree': 3}
        },
    statsmodels_gam = {
            'function': statsmodels_gam, 'type': 'GAM',
            'args': {'DoF': 15, 'degree': 3}
        },
)

predictor_vars = [
    'cc_poll_share_lag',
    'ssp_poll_share_lag',
    'dgm_poll_share_lag',
    'undecided_poll_share_lag',
    'days_until_election'
]
def add_lags(data, election_day):
    data['date'] = pd.to_datetime(data['date'])
    data['election_day'] = election_day.item()
    data['days_until_election'] = data.election_day - data.date
    data['days_until_election'] = data['days_until_election'].dt.days
    target_vars = [
        'cc_poll_share',
        'ssp_poll_share',
        'dgm_poll_share',
        'pdal_poll_share',
        'undecided_poll_share',
    ]
    lags = data.copy()
    lag_vars = lags[target_vars].shift(-1)
    lag_vars.columns += '_lag'
    data[lag_vars.columns] = lag_vars
    data.dropna(inplace=True)
    # data.fillna(0, inplace=True)
    return data

def cap_data(election_year_data):
    # Initialize a dictionary to store capped election year data
    capped_election_year_data = {}

    # Loop for each election year to cap the test data based on the training data's feature ranges
    for test_year in election_year_data.keys():
        # Prepare training and test sets
        train_data = pd.concat([data for year, data in election_year_data.items() if year != test_year])
        test_data = election_year_data[test_year].copy()  # Make a copy to avoid modifying the original data
        
        # Skip if test data is empty
        if test_data.empty:
            continue
        
        for col in predictor_vars:
            # Find the minimum and maximum values for each feature in the training set
            min_val = train_data[col].min()
            max_val = train_data[col].max()
            
            # Cap the test data based on the training set's feature ranges
            test_data[col] = test_data[col].clip(lower=min_val, upper=max_val)
        
        # Store the capped test data
        capped_election_year_data[test_year] = test_data

    # Check if the data has been capped successfully
    list(capped_election_year_data.keys())
    return capped_election_year_data

def forecasts(capped_election_year_data, forecast_function, forecast_args={}):
    # Initialize a dictionary to store evaluation metrics for each party and year after capping the test data
    predictions_vs_reality = {}
    i=0
    # Loop over each election year and each party to build and evaluate models using capped data
    for year, data in capped_election_year_data.items():
        logging.debug(f'Producing Poll Forecasts for year {year} using '
                      f'the {forecast_function} function, with arguments '
                      f'{forecast_args}')
        # Skip if data is empty
        if data.empty:
            continue
        predictions = forecast_function(data, **forecast_args)
        predictions_vs_reality[year] = predictions
    return predictions_vs_reality


def main(scenario, root_dir, data_store='data/interim/', save=False, upper_range=2025, forecasting_method='statsmodels_gam'):
    if forecasting_method == 'pymc_gam':
        logging.warning('WARNING!! PROBABILISTIC FORECASTING FUNCTION '
                        'REQUESTED - VERY COMPUTATIONALLY EXPENSIVE')
    forecasting_function = forecasting_methods[forecasting_method]['function']
    forecast_type = forecasting_methods[forecasting_method]['type']
    forecast_args = forecasting_methods[forecasting_method]['args']
    interim_data_folder = Path(f"{root_dir}/{data_store}/{scenario}/")
    processed_data_folder = Path(f"{interim_data_folder}/trends")
    pickle_path = f'{interim_data_folder}/pickles/{forecast_type}_forecasts.pkl'
    
    election_year_data = {}
    for year in list(map(str, range(1984, upper_range))):
        try:
            data = pd.read_csv(Path(f"{processed_data_folder}/{year}.csv"), index_col=0)
            election_day = electoral_calendar.loc[electoral_calendar.election_cycle == int(year)].election_day
            election_year_data[year] = add_lags(data, election_day)
        except FileNotFoundError:
            continue
    
    
    capped_election_year_data = cap_data(election_year_data)
    predictions_vs_reality = forecasts(capped_election_year_data, forecasting_function, forecast_args)
    if save:
        pickle_path = Path(pickle_path)
        pickle_path.parent.mkdir(exist_ok=True, parents=True)
        print(f'Writing pickle file to {pickle_path}')
        with open(pickle_path, 'wb') as f:
            pickle.dump(predictions_vs_reality, f)
    return predictions_vs_reality
#%%
debugging = False
if __name__ == "__main__":
    if debugging:
        scenario='B'
        # main(f"scenarios/{scenario}", ROOT_DIR, save=True, forecasting_method='particle_filter')
        # main(f"scenarios/{scenario}", ROOT_DIR, save=True, forecasting_method='pymc_gam')
        main(f"scenarios/{scenario}", ROOT_DIR, save=True, forecasting_method='statsmodels_gam')
        # main("", ROOT_DIR, save=True)
        raise Warning
    parser = argparse.ArgumentParser(description='Calculate forecasts based on trends.')
    parser.add_argument('--scenario', required=True, help='Scenario folder name')
    parser.add_argument('--save', required=True, help='Save outputs')
    
    args = parser.parse_args()
    main(args.scenario, ROOT_DIR, save=args.save)

# %%
