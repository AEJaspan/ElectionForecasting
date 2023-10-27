#!/usr/bin/env python
# supressing an irritating future warning internal to the Plotly package.
import warnings
import click
import logging
import pandas as pd
from pathlib import Path

from ElectionForecasting.src.validation.model_inference import (
    inference_on_historical_data, inference_on_holdout_data
)
from ElectionForecasting.src.validation.log_likelihood_scores import\
    evaluate
from ElectionForecasting.src.modelling.DataLoader import DataLoader
from ElectionForecasting.src.data.processing import process_inputs
from ElectionForecasting.src.root import ROOT_DIR
from ElectionForecasting.src.config import DATE_TODAY

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter("ignore", category=FutureWarning)


@click.command()
@click.option('--save', is_flag=True, help='Flag to save the results.')
@click.option('--show', is_flag=True, help='Flag to show the results.')
@click.option(
    '--years_min', type=int, required=False,
    help='Minimum year for backtesting.'
)
@click.option(
    '--years_max', type=int, required=False,
    help='Maximum year for backtesting.'
)
@click.option(
    '--test_name', type=str, required=False,
    help=('Commit name of the test, used when saving model comparison '
          'on historical data.')
)
@click.option(
    '--poll_weighting_config_file', type=str, required=False,
    default='configuration_files/poll_weightings.json',
    help=('Path to a JSON configuration file for the poll '
          'weightings. Defaults to .')
)
@click.option(
    '--forecasting_method',
    type=click.Choice(
        ['particle_filter', 'pymc_gam', 'statsmodels_gam']
    ),
    default='statsmodels_gam',
    required=True)

@click.option(
    '--run_backtesting', is_flag=True,
    help='Flag to run backtesting.'
)
@click.option(
    '--run_on_holdout', is_flag=True,
    help='Flag to run testing on holdout data.'
)
@click.option(
    '--process_scenarios', is_flag=True,
    help='Flag to process scenarios.'
)
@click.option(
    '--process_historical', is_flag=True,
    help='Flag to process historical data.'
)
@click.option(
    '--scenario_list', type=str, required=False,
    help='Comma seperated list of scenarios to process, i.e. A,B,C.'
)
def main(
        save, show, years_min, years_max, test_name, poll_weighting_config_file,
        forecasting_method, run_backtesting, run_on_holdout, process_scenarios,
        process_historical, scenario_list
    ):
    """
    Main function to orchestrate the execution based on CLI flags.
    """
    logging.basicConfig(level=logging.INFO)
    
    if run_backtesting and (years_min is None or years_max is None):
        logging.error("Both --years_min and --years_max "
                      "are required for backtesting.")
        return
    
    if (run_on_holdout or process_scenarios) and not scenario_list:
        logging.error("--scenario_list is required for holdout or scenarios.")
        return

    if scenario_list:
        scenarios = scenario_list.split(',')

    # sampling_kwargs = {'target_accept': .95, 'draws': 5000, 'cores': 4}
    sampling_kwargs = {'target_accept': .95}  # , 'draws': 5000, 'cores': 4}
    election_results_df = pd.read_csv(
        f'{ROOT_DIR}/data/dataland/dataland_election_results_1984_2023.csv'
    )
    # poll_weighting_config_file = Path(f"{ROOT_DIR}/{poll_weighting_config_file}")
    # print(poll_weighting_config_file)
    # return
    # Order of these method calls is important

    if process_historical:
        logging.info("Processing historical data.")
        process_inputs.process(
            f"{ROOT_DIR}/data/interim/", "",
            "../dataland/dataland_polls_1984_2023.csv", save=True,
            upper_range=2024,
            forecasting_method=forecasting_method
        )
    
    if process_scenarios:
        logging.info("Processing scenarios.")
        for scenario in scenarios:
            logging.info(f'PROCESSING SCENARIO {scenario}')
            process_inputs.process(
                f"{ROOT_DIR}/data/interim/", f"scenarios/{scenario}/",
                "dataland_polls_1984_2024_scenario.csv", save=True,
                forecasting_method=forecasting_method
            )

    if run_backtesting:
        data_loader = DataLoader()
        logging.info("Running backtesting.")
        years = [str(y) for y in range(years_min, years_max)]
        for y in years:
            daily_national_results_df, daily_provincial_results_df =\
                inference_on_historical_data(
                    y, data_loader, save=save, show=show,
                    sample_kwargs=sampling_kwargs
                )
            _ = evaluate(
                election_results_df, daily_national_results_df,
                daily_provincial_results_df, y, save=save,
                show=show, test_name=test_name
            )


    if run_on_holdout:
        logging.info("Running on holdout data.")
        for scenario in scenarios:
            _, _ = inference_on_holdout_data(
                '2024', scenario, save=save, show=show,
                sample_kwargs=sampling_kwargs
            )


if __name__ == '__main__':
    cmd = ""
    # cmd += "--run_backtesting --years_min 2018 --years_max 2024 "
    cmd += "--run_on_holdout --scenario_list A,B,C,D,E "
    # cmd += "--process_scenarios "
    # cmd += "--process_historical "
    cmd += "--save --test_name FinalModel"
    main(cmd.split())