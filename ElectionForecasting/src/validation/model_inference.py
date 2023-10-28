import numpy as np
import pandas as pd
from datetime import datetime
import logging

from ElectionForecasting.src.modelling.DataLoader import DataLoader
from ElectionForecasting.src.utils.backtesting_utilities import (
    transform_obs_series, calculate_national_results,
    calculate_provincial_results,
    save_results, get_final_day_provincial_results
)
from ElectionForecasting.src.utils.plotting_utils import (
    plot_log_loss, violin_plot, plot_prob_band,
    close_figure, combined_scatter_and_traces
)
from ElectionForecasting.src.visualisations.electoral_map import \
    plot_election_map
from ElectionForecasting.src.config import (
    party_order, province_order, DATE_TODAY
)
from ElectionForecasting.src.root import ROOT_DIR

from ElectionForecasting.src.utils.general import (
    configure_logging, create_directory
)
from ElectionForecasting.src.utils.backtesting_utilities import (
    initialize_driver_and_data, fit_and_predict
)
from matplotlib.pyplot import savefig
from plotly.io import write_html, write_image

# Configure Logging
configure_logging()

# Initialize variables
# Moved to constants and more descriptive names.
P_ORDER = np.array(party_order)
# DATE_TODAY = datetime.today().strftime('%Y-%m-%d')
# logging.info(f'SETTING DATE TO {DATE_TODAY}')
    
def inference_on_historical_data(
        y, data_loader, sample_kwargs={}, save=False, show=False
):
    """
    The main function for backtesting election forecasting models.
    
    Parameters:
    - y (str): Year of interest for the backtest.
    - data_loader (DataLoader object): An object of the DataLoader class that
                                       handles data loading.
    - save (bool, optional): Whether to save the results and plots. Defaults
                             to False.
    - show (bool, optional): Whether to display the plots. Defaults to False.
    
    Returns:
    - None: Outputs are saved as files if 'save' is True.
    
    """
    logging.info(f'Performing inference on the year {y}.\n\n')
    logging.debug(f'FLAGS: save: {save}, show: {show}.')
    # Directory creation moved to a utility function
    results_dir = create_directory(
        f'{ROOT_DIR}/results/backtesting_results/{DATE_TODAY}/{y}/plots/'
    )
    logging.info(f'Results will be saved to:\n{results_dir}')
    raw_poll_data = pd.read_csv(
        f'{ROOT_DIR}/data/dataland/dataland_polls_1984_2023.csv'
    )
    raw_poll_data['date_conducted'] = pd.to_datetime(
        raw_poll_data['date_conducted'], errors='coerce'
    )
    # Initialize driver and data
    driver = initialize_driver_and_data(y, data_loader)
    
    # Fit and predict
    predictive_samples = fit_and_predict(driver, sample_kwargs)
    poll_dates = driver.model.coords['polling_date']
    new_order = driver.updated_province_order

    # Process predictive samples
    predictive_samples = predictive_samples.posterior_predictive.\
        state_vote_share.values
    predictive_samples = predictive_samples.reshape(
        ([-1] + list(predictive_samples.shape[-3:]))
    )

    # Evaluation
    driver.evaluate(axis=2, pred=predictive_samples[:, 0, :])

    # Plotting
    plot_log_loss(driver.mean_log_loss)
    close_figure(
        None, save_function=savefig, directory=results_dir,
        filename='log_loss', save=save, show=show
    )

    # Prepare observed data for national-level violin plot
    logging.info("Preparing observed data for national-level violin plot...")
    observations = pd.DataFrame(
        [driver.state_test[0].mean(axis=0)],
        columns=[p.upper() for p in party_order]
    )
    observations.columns = observations.columns.str.upper()

    # Generate and save the violin plot at the national level
    logging.info("Generating national-level violin plot...")
    violin_plot(observations, predictive_samples[:, 0, :].mean(axis=1))
    close_figure(
        None, save_function=savefig, directory=results_dir,
        filename='violin_national', save=save, show=show
    )

    # Loop through the provinces and create violin plots
    for i, province_name in enumerate(driver.updated_province_order):
        logging.info(f"Processing province {province_name}...")
        
        # Prepare observed data for provincial-level violin plot
        observations = transform_obs_series(
            pd.Series(driver.target_state_votes
                      .loc[driver.updated_province_order].values[i],
                      index=[p.upper() for p in P_ORDER])
        )

        # Generate and save the violin plot at the provincial level
        violin_plot(observations, predictive_samples[:, 0, :][:, i, :],
                    title=f'{province_name}')
        close_figure(None, save_function=savefig, directory=results_dir,
                     filename=f'violin_{province_name}', save=save, show=show)

    logging.info("Violin plots have been generated and saved.")

    posterior_samples = predictive_samples.mean(axis=2)
    observations = pd.DataFrame(
        [driver.state_test[0].mean(axis=0)], columns=party_order
    ).T
    plot_prob_band(
        predictions=posterior_samples, observed=observations,
        time_index=pd.to_datetime(
            driver.updated_poll_data.index)[::-1]
    )
    close_figure(None, save_function=savefig, directory=results_dir,
                 filename='daily_obs_vote_share', save=save, show=show)

    # Calculate national and provincial results
    daily_national_results_df = calculate_national_results(
        predictive_samples, poll_dates, new_order, P_ORDER
    )
    daily_provincial_results_df = calculate_provincial_results(
        predictive_samples, poll_dates, new_order, party_order, P_ORDER
    )

    final_provincial_results = get_final_day_provincial_results(
        daily_provincial_results_df, new_order
    )

    poll_columns = [c+'_poll_share' for c in P_ORDER]

    for region in ['National'] + driver.updated_province_order[:3]:
        daily_wins = daily_provincial_results_df.xs(
            key=region,
            level='province'
        ) if region != 'National' else daily_national_results_df
        fig = combined_scatter_and_traces(raw_poll_data, region, poll_columns,
                                          poll_dates, daily_wins, observed=observations.T)
        close_figure(
            fig, save_function=write_html, directory=results_dir,
            filename=f'{region}_predictions_with_win_probs', save=save,
            show=show
        )
        close_figure(
            fig, save_function=write_image, directory=results_dir,
            filename=f'{region}_predictions_with_win_probs', save=save,
            show=show
        )
    observations = pd.DataFrame(
        driver.state_test[0],
        columns=[p.upper() for p in party_order],
        index=province_order
    )
    election_map = plot_election_map(
        final_provincial_results,
        title_addition=' - Percentage of simulated wins',
        observed=observations,
    )
    close_figure(
        election_map, save_function=write_html, directory=results_dir,
        filename='simulation_results_map', save=save, show=show
    )
    close_figure(
        election_map, save_function=write_image, directory=results_dir,
        filename='simulation_results_map', save=save, show=show
    )

    # Save results
    save_results(
        daily_national_results_df, daily_provincial_results_df,
        results_dir, y, save
    )
    logging.info(f'Completed inference on the year {y}.\n\n')
    return daily_national_results_df, daily_provincial_results_df

def inference_on_holdout_data(
        y, scenario, sample_kwargs={}, save=False, show=False
):
    """
    The main function for testing the model on holdout data.
    
    Parameters:
    - test_data (DataFrame): Holdout dataset for testing.
    - model (Model object): The trained model for prediction.
    - parameters (dict): Dictionary containing additional parameters
                         required for testing.
    - save (bool, optional): Whether to save the results and plots.
                             Defaults to False.
    - show (bool, optional): Whether to display the plots. Defaults to False.
    
    Returns:
    - None: Outputs are saved as files if 'save' is True.
    
    """
    logging.info(f'Performing inference on holdout data, for the year {y},'
                 f'and the scenario {scenario}.\n\n')
    logging.debug(f'FLAGS: save: {save}, show: {show}.')
    # Directory creation
    results_dir = create_directory(
        f'{ROOT_DIR}/results/holdout_results/'
        f'{DATE_TODAY}/{y}/{scenario}/plots/'
    )
    logging.info(f'Results will be saved to:\n{results_dir}')
    scenario_dir = f'{ROOT_DIR}/data/interim/scenarios/{scenario}/'
    logging.info(f'Interim data will be loaded from:\n{scenario_dir}')
    data_loader = DataLoader(scenario_dir)

    # Initialize driver and data
    driver = initialize_driver_and_data(y, data_loader)
    raw_poll_data = pd.read_csv(
        f'{scenario_dir}/dataland_polls_1984_2024_scenario.csv'
    )
    raw_poll_data['date_conducted'] = pd.to_datetime(
        raw_poll_data['date_conducted'], errors='coerce'
    )

    # Fit and predict
    predictive_samples = fit_and_predict(driver, sample_kwargs)
    poll_dates = driver.model.coords['polling_date']
    new_order = driver.updated_province_order

    # Process predictive samples
    predictive_samples = predictive_samples.posterior_predictive.\
        state_vote_share.values
    predictive_samples = predictive_samples.reshape(
        ([-1] + list(predictive_samples.shape[-3:]))
    )

    # Calculate national and provincial results
    daily_national_results_df = calculate_national_results(
        predictive_samples,
        poll_dates, new_order, P_ORDER
    )
    daily_provincial_results_df = calculate_provincial_results(
        predictive_samples,
        poll_dates, new_order, party_order, P_ORDER
    )
    final_provincial_results = get_final_day_provincial_results(
        daily_provincial_results_df,
        new_order
    )
    poll_columns = [c+'_poll_share' for c in P_ORDER]
    
    for region in ['National'] + driver.updated_province_order[:3]:
        daily_wins = daily_provincial_results_df.xs(
            key=region,
            level='province'
        ) if region != 'National' else daily_national_results_df
        fig = combined_scatter_and_traces(raw_poll_data, region, poll_columns,
                                          poll_dates, daily_wins) # =daily_wins
        close_figure(
            fig, save_function=write_html, directory=results_dir,
            filename=f'{region}_predictions_with_win_probs', save=save,
            show=show
        )
        close_figure(
            fig, save_function=write_image, directory=results_dir,
            filename=f'{region}_predictions_with_win_probs', save=save,
            show=show
        )
        
    election_map = plot_election_map(
        final_provincial_results,
        title_addition=' - Percentage of simulated wins'

    )
    close_figure(election_map, save_function=write_html, directory=results_dir,
                 filename='simulation_results_map', save=save, show=show)
    close_figure(election_map, save_function=write_image, directory=results_dir,
                 filename='simulation_results_map', save=save, show=show)

    # Create empty observations df just for the plot
    observations = pd.DataFrame(
        np.zeros((1, 4)), columns=[p.upper() for p in P_ORDER]
    )

    # Generate and save the violin plot at the national level
    logging.info("Generating national-level violin plot...")
    violin_plot(observations, predictive_samples[:, 0, :].mean(axis=1))
    close_figure(None, save_function=savefig, directory=results_dir,
                 filename='violin_national', save=save, show=show)

    # Loop through the provinces and create violin plots
    for i, province_name in enumerate(driver.updated_province_order):
        logging.info(f"Processing province {province_name}...")

        # Generate and save the violin plot at the provincial level
        violin_plot(observations, predictive_samples[:, 0, :][:, i, :],
                    title=f'{province_name}')
        close_figure(
            None, save_function=savefig, directory=results_dir,
            filename=f'violin_{province_name}', save=save, show=show)

    logging.info("Violin plots have been generated and saved.")

    posterior_samples = predictive_samples.mean(axis=2)

    observations.columns = observations.columns.str.lower()
    plot_prob_band(
        predictions=posterior_samples, observed=observations.T,
        time_index=pd.to_datetime(driver.updated_poll_data.index)[::-1]
    )
    close_figure(
        None, save_function=savefig, directory=results_dir,
        filename='daily_obs_vote_share', save=save, show=show
    )

    save_results(
        daily_national_results_df, daily_provincial_results_df,
        results_dir, scenario, save=save
    )
    logging.info(f'Completed inference on holdout data, for the year {y},'
                 f' and the scenario {scenario}.\n\n')
    return daily_national_results_df, daily_provincial_results_df


if __name__ == '__main__':
    sampling_kwargs = {'target_accept': 1e-5}

    # scenario = 'B'
    # inference_on_holdout_data(
    #     '2024', scenario, sample_kwargs=sampling_kwargs,
    #     save=False, show=False
    # )

    data_loader = DataLoader()
    inference_on_historical_data(
       '2018', data_loader, sample_kwargs= sampling_kwargs,
       save=False, show=False
    )
