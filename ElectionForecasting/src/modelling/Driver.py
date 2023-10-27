import logging
# Standard Library Imports
from functools import partial

# Third-Party Imports
import cloudpickle
import numpy as np
import pandas as pd
import pymc as pm
from pytensor.tensor.random.utils import RandomStream

# Local/Custom Imports
from ElectionForecasting.src.root import ROOT_DIR
# from ElectionForecasting.src.modelling.model import time_for_change_model
from ElectionForecasting.src.modelling.DatalandElectionModel import time_for_change_model
from ElectionForecasting.src.modelling.DataLoader import DataLoader
from ElectionForecasting.src.config import party_order, province_order
from ElectionForecasting.src.config import (unrestricted_parties, unrestricted_provinces,
                                                    restricted_parties, restricted_provinces)

# Initialize logging
logging.basicConfig(level=logging.INFO)

SEED=123

def get_iteration_train_test(inputs, all_state_votes, testing_year):
    """
    Splits the data into training and testing sets based on the testing year.

    Parameters:
        inputs (pd.DataFrame): The input DataFrame containing all data.
        all_state_votes (np.ndarray): Numpy array containing state votes.
        testing_year (int or str): The year for which testing needs to be done.

    Returns:
        tuple: Contains training and testing data along with state votes for training and testing.
    """
    train = inputs.loc[inputs.election_cycle < int(testing_year)]
    test = inputs.loc[inputs.election_cycle >= int(testing_year)]
    frac = train.shape[0]
    state_train = all_state_votes[:frac, :]
    state_test = all_state_votes[frac:, :]
    return train, test, state_train, state_test

def categorical_transformations(inputs, categorical_columns):
    """
    Transforms categorical columns based on the provided party order.

    Parameters:
        inputs (pd.DataFrame): Input data.
        categorical_columns (list): List of categorical columns to transform.

    Returns:
        pd.DataFrame: Transformed DataFrame with categorical columns encoded.
    """
    logging.info("Performing categorical transformations.")
    transformed = inputs.copy()
    for c in categorical_columns:
        cat_trf = transformed[c].astype('category').cat.set_categories(party_order)
        codes = cat_trf.cat.codes
        transformed[c] = codes.astype('int32')
    return transformed


def calculate_log_loss(observed, predictions, axis):
    """
    Calculates the log loss between observed and predicted values.

    Parameters:
        observed (pd.DataFrame): Observed values.
        predictions (np.ndarray): Predicted values.
        axis (int): Axis along which to normalize and compute log loss.

    Returns:
        pd.Series: Mean log loss for each event, sorted in ascending order.
    """
    # Ensure the predictions are properly normalized to sum to 1 across classes
    predictions /= predictions.sum(axis=axis, keepdims=True)

    # Compute log loss for each sample and each event
    log_loss_samples = -np.sum(observed.values[np.newaxis, ...] * np.log(predictions + 1e-15), axis=(axis))  # Adding small value to avoid log(0)
    # Adding a small value to avoid log(0) while calculating log loss
    mask = np.isfinite(np.log(predictions + 1e-15))

    if np.any(mask):
        # Select the values from pred where the mask is False (i.e., where log(pred) is infinite)
        infinite_log_indices = np.argwhere(~mask)
        # Now iterate over these indices to inspect the values from pred at these locations
        for idx in infinite_log_indices:
            idx_tuple = tuple(idx)
            print(f'Predicted Value: {predictions[idx_tuple]} (idx {idx}) has non finite logarithm')

    # Average log loss across all samples for each event
    mean_log_loss = pd.Series(log_loss_samples.mean(axis=0), index=observed.index)
    mean_log_loss = mean_log_loss.sort_values()
    return mean_log_loss

def get_update_data(test_df, polls_df, pleans, state_votes, columns, index, model, testing_year, all_years, corr_matrix):
    """
    Prepares the data needed for Bayesian updating.

    Parameters:
        test_df (pd.DataFrame): Testing data.
        polls_df (pd.DataFrame): Polling data.
        pleans (pd.DataFrame): Priors for the model.
        state_votes (pd.DataFrame): State voting data.
        columns (list): List of columns to be considered.
        index (list): List of index values.
        model (pm.Model): PyMC model object.
        testing_year (str): Year for testing.
        all_years (list): List of all years.
        corr_matrix (np.ndarray): Correlation matrix.

    Returns:
        tuple: Updated data and coordinates for Bayesian updating.
    """
    data = test_df.to_dict('list')
    share_cols = [c+'_share' for c in columns]
    data['y_data'] = np.ones((1,len(columns)))
    previous_year = int(testing_year)-1
    data['previous_year_state_results'] = state_votes[str(previous_year)].loc[province_order][share_cols].values
    data['previous_year_pleans'] = pleans[str(previous_year)].loc[province_order][share_cols].values
    restricted_forecast = polls_df.values
    restricted_forecast /= restricted_forecast.sum(axis=1, keepdims=True)
    data['gam_forecasts'] = restricted_forecast
    coords = {
        "year": [int(testing_year)], 
        "election_cycle": len([int(testing_year)]), 
        "polling_date": polls_df.index,
        "parties": columns,
        "provinces": index,
    }
    data_ = {}
    data_containers = list(dir(model))
    coords_containers = list(model.coords.keys())
    for c in data.keys():
        if c in data_containers:
            data_[c] = data[c]
        if c in coords_containers:
            coords[c] = data[c]
    return data_, coords

class Driver:
    """
    Main class for steering the Bayesian modeling workflow.
    Manages the entire lifecycle of the Bayesian model from data preparation to prediction.
    """
    def __init__(self, data_loader: DataLoader, testing_year: str):
        """Initialize the Driver object.

        Parameters:
            data_loader (DataLoader): Instance of DataLoader to fetch data.
            testing_year (str): The year to test the model.
        """
        logging.info("Initializing Driver object.")
        self.set_years(testing_year)
        self.dl = data_loader
        self.train = None
        self.test = None
        self.state_train = None
        self.state_test = None
        self.model = None
        self.trace = None
        self.post_pred = None

    def set_years(self, year: str):
        """
        Sets the years for training and testing.

        Parameters:
            year (str): The year to set for testing.
        """
        valid_years = [str(y) for y in range(1984, 2025)]
        if year not in valid_years:
            raise ValueError(f"Year {year} is invalid.")
        self.testing_year = year
        self.years = [y for y in valid_years if y < year]

    def set_seed(self, seed=SEED):
        """Set the random seed for reproducibility.

        Parameters:
            seed (int, optional): The random seed. Defaults to 123.
        """
        logging.info(f"Setting random seed to {seed}.")
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)
        #seed=sum(map(ord, "dimensionality")))
        draw = partial(pm.draw, random_seed=self.rng)
        srng = RandomStream(seed=seed)

    def get_bayesian_update_data(self, training_year, testing_year):
        """Get data for Bayesian updating.

        Parameters:
            training_year (int or str): The training year.
            testing_year (int or str): The testing year.

        Returns:
            tuple: A tuple containing updated unrestricted and restricted data and coordinates.
        """
        logging.info("Getting Bayesian update data.")
        self.updated_poll_data = self.dl.get_gam_forecast(str(testing_year))
        previous_state_votes = self.dl.state_votes#[str(training_year)]
        # all_state_votes = np.stack([v.values for v in self.dl.state_votes.values()])
        pleans = self.dl.pleans#[str(training_year)]
        self.updated_poll_data.columns = self.updated_poll_data.columns.str.replace('_pred','_share')
        update_data, update_coords = get_update_data(
            self.test[:1],
            self.updated_poll_data,
            pleans,
            previous_state_votes,
            unrestricted_parties,
            unrestricted_provinces,
            self.model,
            testing_year,
            self.years,
            self.dl.corr_12x12
        )
        return update_data, update_coords

    def predict(self, sampling_kwargs={}):
        """Run the prediction step.

        Returns:
            np.ndarray: XData array containing predictive samples.
        """
        logging.info("Running prediction step.")
        data, coords = self.get_bayesian_update_data(self.training_year, self.testing_year)
        self.update_data = data
        self.update_coords = coords
        with self.model:
            logging.info('Predictive sampling for all regions')
            pm.set_data(new_data=data, coords=coords)
            ppc = pm.sample_posterior_predictive(trace=self.trace, random_seed=self.rng, **sampling_kwargs)
        self.updated_province_order = list(unrestricted_provinces) + list(restricted_provinces)
        self.predictions = ppc
        return self.predictions

    def transform_inputs(self):
        """
        Transforms the input data for model building.
        This includes splitting the data into training and testing sets and encoding categorical variables.
        """
        logging.info("Transforming input data.")
        all_state_votes = np.stack([v.values for v in self.dl.state_votes.values()])
        # the itteration is the year that you are testing
        train, test, state_train, state_test = get_iteration_train_test(self.dl.inputs, all_state_votes, self.testing_year)
        self.training_year = train.iloc[-1].year
        # assert int(self.testing_year) == int(test.iloc[0].year), f"{int(self.testing_year)} != {int(test.iloc[0])}"
        assert int(self.training_year) == int(self.testing_year) - 1, f"{int(self.training_year)} != {int(self.testing_year) - 1}"
        # self.testing_year = testing_year
        self.gam_forecast = self.dl.get_gam_forecast(self.testing_year)
        self.previous_plean = self.dl.pleans[str(self.training_year)]
        self.previous_state_votes = self.dl.state_votes[str(self.training_year)]
        cat_columns = ['year_on_year_gdp_pct_change', 'unemployment_rate',
                       'year_on_year_inflation', 'year_on_year_stock_mkt_pct_change',
                       'party_in_power', 'national_pop_vote_winner']
        self.target_state_votes = self.dl.state_votes[str(self.testing_year)]        
        input_columns = [c for c in self.dl.inputs.columns if any(x+'_' in c for x in cat_columns)]
        input_columns += ['election_cycle']
        categorical_columns = [c for c in self.dl.inputs.columns if 'national_pop_vote_winner'+'_' in c]
        categorical_columns += [c for c in self.dl.inputs.columns if 'party_in_power'+'_' in c]
        transformed_training = categorical_transformations(
            train[input_columns],
            categorical_columns=categorical_columns,
        )
        transformed_testing = categorical_transformations(
            test[input_columns], 
            categorical_columns=categorical_columns,
        )
        self.train = transformed_training
        self.test = transformed_testing
        self.state_train = state_train
        self.state_test = state_test

    def build_model(self):
        """
        Builds the Bayesian model based on the transformed input data.
        """
        share_cols = [c+'_share' for c in party_order]
        previous_year_pleans = self.dl.pleans[str(self.training_year)]
        self.previous_year_pleans = previous_year_pleans.loc[province_order][share_cols].values
        self.model = time_for_change_model(gam_forecasts=self.gam_forecast,
                                           train_df=self.train,
                                           vote_shares=self.state_train,
                                           previous_year_pleans=self.previous_year_pleans)

    def sample_posterior(self, sampling_kwargs={'target_accept': .9}):
        with self.model:
            self.trace = pm.sample(**sampling_kwargs)  

    def sample_posterior_predictive(self, sample_posterior_predictive_kwargs: dict):
        """Sample from the posterior predictive distribution.

        Returns:
            dict: Samples from the posterior predictive distribution.
        """
        with self.model:
            post_pred = pm.sample_posterior_predictive(self.trace, **sample_posterior_predictive_kwargs)
        return post_pred

    def build(self):
        """Transform inputs and build the model."""
        logging.info("Building build_modelthe model.")
        self.transform_inputs()
        self.build_model()

    def fit(self, sample_kwargs={}):
        """
        Fits the Bayesian model by sampling from its posterior distribution.

        Parameters:
            sample_kwargs (dict, optional): Additional keyword arguments for sampling.
        """
        logging.info("Fitting the model.")
        self.sample_posterior(sample_kwargs)

    def check(self, sample_posterior_predictive_kwargs={}):
        """Perform posterior predictive checks.

        Returns:
            dict: Posterior predictive samples.
        """
        return self.sample_posterior_predictive(sample_posterior_predictive_kwargs)

    def evaluate(self, pred=None, axis=2):
        """
        Fits the Bayesian model by sampling from its posterior distribution.

        Parameters:
            sample_kwargs (dict, optional): Additional keyword arguments for sampling.
        """
        obs = self.target_state_votes.loc[self.updated_province_order]
        if type(pred) != np.ndarray:
            pred = self.predictions[:,0,:]
        self.mean_log_loss = calculate_log_loss(obs, pred, axis)
        self.mean_log_loss.name = 'mean_log_loss'
        return self.mean_log_loss

    def save(self, path=''):
        """
        Saves the model to disk.

        Parameters:
            path (str, optional): The path where the model should be saved. If empty, uses a default path.
        """
        logging.info(f"Saving model for testing year {self.testing_year}.")
        Model = cloudpickle.dumps({'Driver': self})
        if '.pkl' not in path:
            path = f'{ROOT_DIR}/models/production_models/{self.testing_year}.pkl'
        with open(path, 'wb') as file:
            file.write(Model)

if __name__ == '__main__':
    # Initialize configuration parameters (this could be loaded from a file or environment variables)
    config = {}  
    logging.info("Starting Bayesian model workflow.")

    data_loader = DataLoader()
    # Initialize and build the model
    driver = Driver(data_loader)
    data_loader.load_data()
    driver.set_seed()
    driver.build()

    # Fit the model
    driver.fit()

    # Perform posterior predictive checks
    post_pred = driver.check()

    # Generate predictive samples
    predictive_samples = driver.predict()
