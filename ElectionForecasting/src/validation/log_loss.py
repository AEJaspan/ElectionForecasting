# #%%
# from pathlib import Path
# import pandas as pd

# from ElectionForecasting.src.config import party_order
# from ElectionForecasting.src.modelling.modelling_class import calculate_log_loss
# from ElectionForecasting.src.root import ROOT_DIR
# election_results_df = pd.read_csv(f'{ROOT_DIR}/data/dataland/dataland_election_results_1984_2023.csv')
# results_paths = Path(f'{ROOT_DIR}/results/backtesting_results/').glob('*/*/')
# latest_results_dir = max([f for f in results_paths], key=lambda item: item.stat ().st_ctime)
# year = latest_results_dir.stem
# selected_election_results = election_results_df[election_results_df.year == int(year)]
# selected_election_results = selected_election_results[['national_winner'] + [c+'_share' for c in party_order]]
# predicted_national_results = pd.read_csv(f"{latest_results_dir}/national_forecast_2022.csv", index_col=0)
# # predicted_national_results.iloc[0]
# selected_election_results.national_winner.iloc[0]
# predicted_national_results
# #%%
# pd.read_csv(f"{latest_results_dir}/provincial_forecast_2022.csv", index_col=[0,1])
# #%%
