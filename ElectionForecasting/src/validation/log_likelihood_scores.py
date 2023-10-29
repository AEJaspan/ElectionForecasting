#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from ElectionForecasting.src.config import party_order, DATE_TODAY
from ElectionForecasting.src.utils.backtesting_utilities import calculate_log_likelihoods, dirichlet_log_likelihood
from ElectionForecasting.src.root import ROOT_DIR
from ElectionForecasting.src.utils.plotting_utils import plot_log_likelihood
from ElectionForecasting.src.utils.general import configure_logging

configure_logging()
# DATE_TODAY = datetime.today().strftime('%Y-%m-%d')

def get_scores(election_results_df, predicted_national_results, predicted_provincial_results, year):
    share_columns = [c+'_share' for c in party_order]
    selected_election_results = election_results_df[election_results_df.year == int(year)]
    selected_election_results = selected_election_results[['national_winner', 'province'] + share_columns].reset_index(drop=True)
    x = selected_election_results[share_columns].mean(axis=0)
    alpha = predicted_national_results.iloc[0]
    log_likelihoods = {}
    log_likelihoods['national'] = [calculate_log_likelihoods(x, alpha)]
    selected_election_results[selected_election_results.province == 'Amperville'][share_columns].T[0]
    predicted_provincial_results.sort_index(level=[0, 1], inplace=True, ascending=False)
    predicted_provincial_results = predicted_provincial_results.loc[predicted_provincial_results.index.levels[0][0]]
    predicted_provincial_results.columns = predicted_provincial_results.columns.str.replace('_mean_vote_share','')
    predicted_provincial_results = predicted_provincial_results[party_order]
    for p in predicted_provincial_results.index:
        x = selected_election_results[selected_election_results.province == p][share_columns].T.iloc[:,0].values
        alpha = predicted_provincial_results.loc[p].values
        log_likelihoods[p] = [dirichlet_log_likelihood(x, alpha)]
    return pd.DataFrame(log_likelihoods, index=[f'log_likelihood_{year}']).T

def evaluate(
        election_results_df, daily_national_results_df,
        daily_provincial_results_df, y, save=False,
        show=False, test_name=''
    ):
    log_likelihoods = get_scores(
        election_results_df, daily_national_results_df,
        daily_provincial_results_df, y
    )
    logging.info('Equal weighted sum of the log-likelihood scores: '
                 f'{log_likelihoods.mean().item():.2f}')
    plot_log_likelihood(log_likelihoods.iloc[:,0])
    sum_ll_str = f"{log_likelihoods.sum().item():.2f}".replace('-', 'm_')
    headline = f'LL_mean_score_{sum_ll_str}'
    if show:
        plt.show()
    if save:
        results_dir = Path(f'{ROOT_DIR}/results/evaluation/{DATE_TODAY}/{test_name}/{y}/{headline}/')
        logging.info(f'Saving experiment results to:\n{results_dir}')
        results_dir.mkdir(exist_ok=True, parents=True)
        log_likelihoods.iloc[:,0].sort_values().to_csv(results_dir/'log_likelihood_scores.csv')
        plt.savefig(results_dir / f'log_likelihoods.png')
    return log_likelihoods

def main(latest_results_dir, save=False):
    election_results_df = pd.read_csv(f'{ROOT_DIR}/data/dataland/dataland_election_results_1984_2023.csv')
    for results_dir in latest_results_dir.glob('*/'):
        year = results_dir.stem
        print(f'Calculating log likelihood scores for year {year} from dir {results_dir}')
        predicted_national_results = pd.read_csv(f"{results_dir}/national_forecast_{year}.csv", index_col=0)
        predicted_provincial_results = pd.read_csv(f"{results_dir}/provincial_forecast_{year}.csv", index_col=[0,1])
        print(f'Found predicted results in file: {results_dir}')
        log_likelihoods = evaluate(election_results_df, predicted_national_results, predicted_provincial_results, year)

if __name__ == '__main__':
    results_paths = Path(f'{ROOT_DIR}/results/backtesting_results/').glob('*/')
    latest_results_dir = max([f for f in results_paths], key=lambda item: item.stat ().st_ctime)
    main(latest_results_dir, save=True)
