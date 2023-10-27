"""
This module contains the DataLoader class which is responsible for loading
the model inputs from static .csv and .pkl files. These are interim data
files produced in the processing stage.
"""

import logging
# Standard Library Imports
from pathlib import Path
import pickle

# Third-Party Imports
import pandas as pd

# Local/Custom Imports
from ElectionForecasting.src.root import ROOT_DIR
from ElectionForecasting.src.config import party_order, province_order
from ElectionForecasting.src.config import unrestricted_provinces, restricted_provinces

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Utility function for sorting by filename stem
sort_by_stem = lambda x: x.stem

class DataLoader:
    interim_data_dir = f'{ROOT_DIR}/data/interim/'
    def __init__(self, interim_data_path=None) -> None:
        if type(interim_data_path) == str:
            self.interim_data_dir = interim_data_path

    def load_model_inputs(self):
        """Load model inputs from CSV file.

        Returns:
            pd.DataFrame: DataFrame containing model inputs.
        """
        logging.info("Loading model inputs.")
        return pd.read_csv(f'{self.interim_data_dir}/time_for_change/inputs.csv', index_col=0)

    def load_gam(self):
        """Load GAM forecasts from a pickle file.

        Returns:
            dict: Dictionary containing GAM forecasts.
        """
        logging.info("Loading GAM forecasts.")
        pickle_path = f'{self.interim_data_dir}/pickles/GAM_forecasts.pkl'
        try:
            with open(pickle_path, 'rb') as f:
                gam_forecasts = pickle.load(f)
        except FileNotFoundError:
            logging.error(f"File not found: {pickle_path}")
        return gam_forecasts

    def get_gam_forecast(self, year):
        """Retrieve GAM forecast for a specific year.

        Parameters:
            year (int or str): The year for which to retrieve the forecast.

        Returns:
            pd.DataFrame: DataFrame containing the GAM forecast.
        """
        logging.info(f"Retrieving GAM forecast for {year}.")
        pred = {}
        obs = []
        for prty in party_order:#+['undecided']:
            party_col = f'{prty}_poll_share'
            pred[f'{prty}_pred'] = self.gam_forecasts[str(year)][party_col]['y_pred']
            obs.append(self.gam_forecasts[str(year)][party_col]['y'])
        date = self.gam_forecasts[str(year)]['cc_poll_share']['date']
        # Y_pred = pd.DataFrame(pred, index=date)
        Y_pred = pd.DataFrame.from_dict(pred)
        Y_pred.index = date
        return Y_pred

    def load_state_results(self):
        """Load state results from CSV files.

        Returns:
            dict: Dictionary containing state votes by year.
        """
        logging.info("Loading state results.")
        state_results = Path(f'{self.interim_data_dir}/state_results/').glob('*.csv')
        x = list(state_results)
        s = sorted(x, key=sort_by_stem)
        state_votes = {}
        for p in s:
            sv = pd.read_csv(p.as_posix(), index_col=0)
            sv = sv[[p+'_share' for p in party_order]].loc[province_order]
            year = str(p.stem)
            state_votes[year] = sv
        return state_votes

    def load_pleans(self):
        """Load pleans data from CSV files.

        Returns:
            dict: Dictionary containing pleans by year.
        """
        logging.info("Loading pleans.")
        # pleans_paths = Path(f'{self.interim_data_dir}/pleans_no_weighting/pleans/').glob('*.csv')
        pleans_paths = Path(f'{self.interim_data_dir}/pleans_new/pleans/').glob('*.csv')
        pleans_paths = sorted(pleans_paths, key=sort_by_stem)
        pleans = {}
        for p in pleans_paths:
            plean = pd.read_csv(p.as_posix(), index_col=0)#.reset_index(drop=True)
            plean = plean[[p+'_share' for p in party_order]].loc[province_order]
            year = p.stem
            pleans[str(year)] = plean
        return pleans

    def load_corr_matrix(self):
        logging.info("Loading pleans correlations.")
        path = f'{ROOT_DIR}/data/interim/correlations/significant_province_correlation_matrices.csv'
        corr_full = pd.read_csv(path, index_col=0).fillna(0)
        ssp_cols = [c for x in unrestricted_provinces for c in corr_full.columns if x in c]
        ssp_free_cols = [c for x in restricted_provinces for c in corr_full.columns if (x in c) and ('ssp' not in c)]
        corr_48x48 = corr_full.copy()
        corr_27x27 = corr_full.copy().loc[ssp_free_cols][ssp_free_cols]
        corr_12x12 = corr_full.copy().loc[ssp_cols][ssp_cols]
        return corr_48x48, corr_27x27, corr_12x12


    def load_data(self):
        """Convenience method to load all data at once."""
        self.pleans = self.load_pleans()
        self.state_votes = self.load_state_results()
        self.inputs = self.load_model_inputs()
        self.gam_forecasts = self.load_gam()
        corrs = self.load_corr_matrix()
        self.corr_48x48 = corrs[0]
        self.corr_27x27 = corrs[1]
        self.corr_12x12 = corrs[2]
