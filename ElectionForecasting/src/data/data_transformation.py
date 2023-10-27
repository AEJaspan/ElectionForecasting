
import pandas as pd

def transform_electoral_results(electoral_results: pd.DataFrame) -> pd.DataFrame:
    # Group by year and aggregate
    electoral_results = electoral_results[['year','party_in_power',
       'national_winner', 'national_pop_vote_winner', 'cc_share', 'dgm_share',
       'pdal_share', 'ssp_share']].groupby('year').agg({
           'party_in_power': 'max',
           'national_winner': 'max',
           'national_pop_vote_winner': 'max',
           'cc_share': 'mean',
           'dgm_share': 'mean',
           'pdal_share': 'mean',
           'ssp_share': 'mean',
       })
    
    electoral_results.rename(columns={
           'cc_share': 'national_cc_share',
           'dgm_share': 'national_dgm_share',
           'pdal_share': 'national_pdal_share',
           'ssp_share': 'national_ssp_share',
           }, inplace=True)
    electoral_results.reset_index(inplace=True)
    return electoral_results

def merge_electoral_data(electoral_calendar: pd.DataFrame, electoral_results: pd.DataFrame) -> pd.DataFrame:
    # Merge electoral_calendar with electoral_results
    election_results = pd.merge(
        electoral_calendar, electoral_results, left_on='election_cycle', right_on='year'
    )
    return election_results

def generate_lagged_variables(economic_data: pd.DataFrame, election_results: pd.DataFrame) -> pd.DataFrame:
    # Generate lagged variables for economic indicators
    election_results = pd.merge_asof(economic_data.sort_values('date'),
                                     election_results.sort_values('election_day'),
                                     left_on='date', right_on='election_day', direction='forward')
    
    lag_cols = []
    quarterly_cols=['year_on_year_gdp_pct_change', 'unemployment_rate',
                    'year_on_year_inflation', 'year_on_year_stock_mkt_pct_change',
                    'party_in_power', 'national_pop_vote_winner']
    
    for lag in [0, 1, 2, 3]:
        for col in quarterly_cols:
            lag_col=f'{col}_lag_{lag}q'
            lag_cols.append(lag_col)
            election_results[lag_col] = election_results[col].shift(lag)
        lag_date = f'date_lag_{lag}q'
        lag_cols.append(lag_date)
        election_results[lag_date] = election_results['date'].shift(lag)
        
    return election_results
