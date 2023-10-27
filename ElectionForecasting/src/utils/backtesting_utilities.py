import numpy as np
import pandas as pd

from itertools import product

from ElectionForecasting.src.modelling.Driver import Driver
from ElectionForecasting.src.config import party_order
from ElectionForecasting.src.root import ROOT_DIR
import numpy as np
from scipy.special import gammaln


# model_utils.py
def initialize_driver_and_data(year, data_loader):
    driver = Driver(data_loader, year)
    data_loader.load_data()
    driver.set_seed()
    driver.build()
    return driver

def fit_and_predict(driver, sample_kwargs={}):
    driver.fit(sample_kwargs=sample_kwargs)
    return driver.predict()

def dirichlet_log_likelihood(alpha, x):
    """
    Compute the log-likelihood of Dirichlet distribution.
    
    Parameters:
        alpha (array-like): Parameter vector of the Dirichlet distribution
        x (array-like): Data vector (should be on the same simplex as the distribution)
        
    Returns:
        float: The log-likelihood value
    """
    alpha = alpha.astype('float')+1e-15
    x = x.astype('float')+1e-15
    # Compute the constant term
    constant_term = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))
    
    # Compute the summation term
    summation_term = np.sum((alpha - 1) * np.log(x))
    
    log_likelihood = constant_term + summation_term
    if not np.isfinite(log_likelihood):
        raise ValueError(f'Non finite log likelihood! {log_likelihood}'
                         f'Inputs were: \nalpha: {alpha}\nx: {x}')
    return log_likelihood

def calculate_log_likelihoods(election_result: pd.Series, final_day_prediction: pd.Series):
    x = election_result 
    alpha = final_day_prediction 
    alpha = alpha[alpha.index.str.contains('_mean_vote_share')]
    alpha.index = alpha.index.str.replace('_mean_vote_share', '')
    alpha.sort_index(inplace=True)
    alpha = alpha.values
    x = x[x.index.str.contains('_share')]
    x.index = x.index.str.replace('_share', '')
    x = x.values
    log_likelihood = dirichlet_log_likelihood(alpha, x)
    print("Dirichlet log-likelihood:", log_likelihood)
    return log_likelihood

    
def transform_obs_series(series):
    series.index = series.index.str.replace('national_','')
    series.index = series.index.str.replace('_share','')
    series.index = series.index.str.upper()
    return series.to_frame().T

# 160 EC votes
EC = pd.read_csv(f'{ROOT_DIR}/data/dataland/dataland_demographics.csv', index_col=0)['electoral_college_votes']
def count_winners(simulatons, province_order):
    electoral_college_votes = EC.loc[province_order].values
    # Find the indices of the winning categories
    winning_categories_indices = np.argmax(simulatons, axis=3)  # Shape: (1000, 73, 12)

    # Create a mask identifying the winning category for each observation
    winning_mask = np.arange(4) == winning_categories_indices[..., np.newaxis]  # Shape: (1000, 73, 12, 4)

    # Broadcast point_values to match the dimensions of data
    broadcasted_point_values = electoral_college_votes[np.newaxis, np.newaxis, :, np.newaxis]  # Shape: (12, 1, 1, 1)

    # Use the mask to allocate the points to the winning categories
    points_allocated = winning_mask * broadcasted_point_values  # Shape: (1000, 73, 12, 4)

    # Sum the points for each category in each group on each day in each event
    tallied_points = np.sum(points_allocated, axis=2)  # Shape: (1000, 73, 4)

    # Now you can proceed to find the category with the most points for each day in each event
    resulting_winning_categories = np.argmax(tallied_points, axis=2)  # Shape: (1000, 73)
    return resulting_winning_categories


def daily_national_win_percentages(resulting_winning_categories):
    num_events, num_days = resulting_winning_categories.shape[:2]
    # Number of categories
    num_categories = 4

    # Step 1: Create a 2D array to hold the count of wins for each category on each day
    win_counts = np.zeros((num_days, num_categories), dtype=int)

    # Step 2: Increment the count of wins for the winning category on each day
    np.add.at(win_counts, (np.arange(num_days)[None, :].repeat(num_events, axis=0), resulting_winning_categories), 1)

    # Step 3: Divide the count of wins by the total number of events to obtain the win percentage
    win_percentages = (win_counts / num_events)

    return win_percentages

def daily_provincial_win_percentages(provincial_winners, n_polling_days=73):
    # Number of events
    num_events = provincial_winners.shape[0]

    # Step 1: Create an empty array to store the count of wins
    win_counts = np.zeros((n_polling_days, 12, 4), dtype=int)

    # Step 2: Increment the count of wins
    np.add.at(win_counts, (np.arange(n_polling_days)[None, :, None], 
                        np.arange(12)[None, None, :], 
                        provincial_winners), 1)

    # Step 3: Calculate the percentage of wins
    win_percentages = (win_counts / num_events)

    # Now, win_percentages is a 3D array of shape (73, 12, 4) with the percentage of wins for each category in each group on each day
    return win_percentages


def calculate_national_results(samples, poll_dates, updated_province_order, p_order):
    """Calculate and return the national results DataFrame."""
    mean_vote_shares = samples.mean(axis=2).mean(axis=0)
    resulting_winning_categories = count_winners(samples, updated_province_order)
    national_win_percentages = daily_national_win_percentages(resulting_winning_categories)
    
    # Construct the DataFrame
    daily_national_vote_shares_df = pd.DataFrame(mean_vote_shares, index=poll_dates, columns=[f"{c}_mean_vote_share" for c in party_order])
    daily_national_win_probability_df = pd.DataFrame(national_win_percentages, columns=[f"{c}_win_probability" for c in p_order], index=poll_dates)
    
    return daily_national_vote_shares_df.join(daily_national_win_probability_df)


def calculate_provincial_results(samples, poll_dates, new_order, party_order, p_order):
    """Calculate and return the provincial results DataFrame."""
    provincial_winners = np.argmax(samples, axis=3)
    daily_provincial_wins = daily_provincial_win_percentages(provincial_winners, n_polling_days=provincial_winners.shape[1])
    
    # Reshape and create MultiIndex
    daily_provincial_wins = daily_provincial_wins.reshape(-1, 4)
    multi_index_tuples = list(product(poll_dates, new_order))
    multi_index = pd.MultiIndex.from_tuples(multi_index_tuples, names=['polling_date', 'province'])
    
    # Construct the DataFrame
    daily_provincial_vote_shares_df = pd.DataFrame(samples.mean(axis=0).reshape(-1, 4), index=multi_index, columns=[f"{c}_mean_vote_share" for c in party_order])
    daily_provincial_win_probability_df = pd.DataFrame(daily_provincial_wins, columns=[f"{c}_win_probability" for c in p_order], index=multi_index)
    
    return daily_provincial_vote_shares_df.join(daily_provincial_win_probability_df)

def save_results(national_df, provincial_df, dir, y, save):
    """Save the calculated results to CSV files if 'save' is True."""
    if save:
        national_df.to_csv(dir.parent / f'national_forecast_{y}.csv')
        provincial_df.to_csv(dir.parent / f'provincial_forecast_{y}.csv')

def format_provincial_wins_columns(daily_provincial_results_df):
    formatted_provincial_wins_df = daily_provincial_results_df.copy()
    formatted_provincial_wins_df['winner'] = formatted_provincial_wins_df.winner.str.replace('_win_probability', '')
    formatted_provincial_wins_df['winner'] = formatted_provincial_wins_df.winner.str.replace('_mean_vote_share', '')
    formatted_provincial_wins_df = formatted_provincial_wins_df.filter(like='win')
    formatted_provincial_wins_df.columns = formatted_provincial_wins_df.columns.str.replace('_win_probability', '')
    formatted_provincial_wins_df.columns = formatted_provincial_wins_df.columns.str.replace('_mean_vote_share', '')
    return formatted_provincial_wins_df

def get_final_day_provincial_results(daily_provincial_results_df, new_order):
    final_provincial_results = daily_provincial_results_df.sort_index(level=('polling_date', 'province'))
    final_provincial_results = final_provincial_results.iloc[-12:]
    final_provincial_results.reset_index(inplace=True)
    final_provincial_results.drop(columns='polling_date', inplace=True)
    final_provincial_results.set_index('province', inplace=True)
    final_provincial_results = final_provincial_results.loc[new_order]
    final_provincial_results['winner'] = final_provincial_results.columns[np.argmax(final_provincial_results, axis=1)]
    return format_provincial_wins_columns(final_provincial_results)
