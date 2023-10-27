from pathlib import Path
import pandas as pd
import json
from pollscraper.trends import PollTrend, Weighting
import argparse
from ElectionForecasting.src.root import ROOT_DIR
import numpy as np

def read_data(input_folder, file_name):
    return pd.read_csv(Path(f"{input_folder}/{file_name}"))
 
electoral_calendar = read_data(f'{ROOT_DIR}/data/dataland/', 'dataland_electoral_calendar.csv')


def calculate_trends(df, weights_col, output_folder, year, save_outputs):
    pt = PollTrend()
    df.reset_index(inplace=True)
    trends, outliers_avg, outliers_poll = pt.calculate_trends(df, weights_col=weights_col, start_date=None)
    if save_outputs:
        results_path = Path(f"{output_folder}/trends/{year}.csv")
        results_path.parent.mkdir(exist_ok=True, parents=True)
        print(f'Writing results to {results_path}')
        trends.to_csv(results_path)
    return trends

def main(input_file, output_folder,
         configuration_file=None,
         save_outputs=False):
    
    config_file = f'{ROOT_DIR}/configuration_files/poll_weightings.json'
    if type(configuration_file) == str:
        if str(configuration_file).lower().endswith('.json'):
            config_file = configuration_file
    # Read poll weightings from a JSON config file
    with open(config_file, 'r') as f:
        loaded_data = json.load(f)

    # Access the data
    modality_factor_weights = loaded_data['modality_factor_weights']
    population_factor_weights = loaded_data['population_factor_weights']
    pollster_factor_weights = loaded_data['pollster_factor_weights']


    df = pd.read_csv(input_file)
    w=Weighting()
    w.modality_factor_weights = modality_factor_weights
    w.population_factor_weights = population_factor_weights
    w.pollster_factor_weights = pollster_factor_weights
    weightings_function=w.weighting_scheme_538
    cols = [c for c in df.columns if 'poll_share' in c]
    df_a = df[df['geography']=='National'][['date_conducted','pollster', 'sample_size']+cols].rename(columns={'sample_size':'n', 'date_conducted':'date'})
    df_a['date'] = pd.to_datetime(df_a['date'], format='%Y-%m-%d')#.dt.round('1d')
    df_a['weights'] = weightings_function(
                    df['sample_size'],
                    pollster_col=df['pollster'],
                    modality_col=df['mode'],
                    population_col=df['population_surveyed']
                )
    elections = pd.to_datetime(electoral_calendar['election_day'], format='%Y-%m-%d').dt.round('1d')
    elections = elections.sort_values(ascending=False)
    df_a = df_a.sort_values(by='date', ascending=False)
    end = df_a.set_index('date').first_valid_index()
    start = df_a.set_index('date').last_valid_index()
    elections = elections.loc[(elections.values <= end) & (elections.values >= start)]
    elections = elections.to_list()
    elections = [end] + elections + [start]
    election_windows = zip(elections, elections[1:])
    v = list(election_windows)[0]
    x = df_a[(df_a.date >= (v[1])) & (df_a.date <= (v[0]))]
    year = v[0].year
    all_trends = {}
    election_windows = zip(elections, elections[1:])
    for v in list(election_windows):
        x = df_a[(df_a.date >= (v[1])) & (df_a.date <= (v[0]))]
        year = v[0].year
        trends = calculate_trends(x, 'weights', output_folder, year, save_outputs=save_outputs)
        all_trends[year] = trends
    return all_trends

debugging = False
if __name__ == "__main__":
    if debugging:
        scenario='A'
        # main("data/dataland/dataland_polls_1984_2023.csv", "data/interim/", save_outputs=True)
        main(f"data/interim/scenarios/{scenario}/dataland_polls_1984_2024_scenario.csv", f"data/interim/scenarios/{scenario}")
        raise Warning
    parser = argparse.ArgumentParser(description='Calculate weighted averages using PollTrends library.')
    parser.add_argument('--input_file', required=True, help='Input file path')
    parser.add_argument('--output_folder', required=True, help='Output folder path')
    
    args = parser.parse_args()
    main(args.input_file, args.output_folder)
