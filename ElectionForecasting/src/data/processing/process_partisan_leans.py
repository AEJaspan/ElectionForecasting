import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from ElectionForecasting.src.root import ROOT_DIR
from ElectionForecasting.src.config import party_order, province_order


def load_election_results(election_results_folder):
    election_results = []
    election_result_paths = Path(election_results_folder).glob('*.csv')
    for p in election_result_paths:
        election_result = pd.read_csv(p.as_posix(), index_col=0).fillna(0)
        year = p.stem
        election_results.append((str(year), election_result))
    election_results.sort(key=lambda x: x[0], reverse=False)
    return election_results

def normalized_sequential_exponential_weighted_sum(partisan_leans, alpha=0.9):
    weighted_sums = []
    current_sum = pd.DataFrame(np.zeros(partisan_leans[0][1].shape), columns=partisan_leans[0][1].columns, index=partisan_leans[0][1].index)
    normalization_factor = 0
    # Normalised weighted sum:
    # weight = \frac{\alpha^i}{\sum_{j=0}^{i} \alpha^j}
    for i, (year, lean) in enumerate(partisan_leans):
        weight = (alpha ** i) / sum(alpha ** j for j in range(i + 1))
        current_sum = weight * lean + (1 - weight) * current_sum
        weighted_sums.append((year, current_sum.copy()))
        
    return weighted_sums


def process_election_results(election_results, output_folder, save_output):
    partisan_leans = [(str(int(year)), df.sub(df.mean(axis=0), axis=1)) for year, df in election_results]
    weighted_sums = normalized_sequential_exponential_weighted_sum(partisan_leans)
    # weighted_sums = partisan_leans
    if save_output:
        output_folder = Path(output_folder) / 'pleans'
        output_folder.mkdir(exist_ok=True, parents=True)
        # Save processed data to CSV
        print(f'Writing pleans to {output_folder}')
        for y, s in weighted_sums:
            s.to_csv(f'{output_folder}/{y}.csv')
    return weighted_sums

def main(election_results_folder, output_folder, save_output=False):
    election_results = load_election_results(election_results_folder)
    process_election_results(election_results, output_folder, save_output)

debug=True
if __name__ == '__main__':
    if debug:
        main(f'{ROOT_DIR}/data/interim/state_results/',
            f'{ROOT_DIR}/data/interim/pleans_new/',
            save_output=True)
        raise Warning
    parser = argparse.ArgumentParser(description='Election Results Processing Script')
    parser.add_argument('--election_results_folder', required=True, help='Path to the election_results')
    parser.add_argument('--output_folder', required=True, help='Path to store outputs')
    parser.add_argument('--save_output', action='store_true', help='Flag to save output, omit for debugging')
    args = parser.parse_args()
    main(args.election_results_folder, args.output_folder, args.save_output)