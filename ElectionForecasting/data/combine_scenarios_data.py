#%%
import numpy as np
import pandas as pd
from ElectionForecasting.src.root import ROOT_DIR
from pathlib import Path
# Usage
#%%

economic_data_scenarios = pd.read_csv(f'{ROOT_DIR}/data/dataland/dataland_economic_data_2024_scenarios.csv')
economic_data_1984_2023 = pd.read_csv(f'{ROOT_DIR}/data/dataland/dataland_economic_data_1984_2023.csv')
#%%
polls_data_scenarios = pd.read_csv(f'{ROOT_DIR}/data/dataland/dataland_polls_2024_scenarios.csv')
polls_data_1984_2023 = pd.read_csv(f'{ROOT_DIR}/data/dataland/dataland_polls_1984_2023.csv')
#%%
set(economic_data_scenarios[economic_data_scenarios['scenario'] == 'A'].drop(columns='scenario').columns) - set(economic_data_1984_2023.columns)

#%%
set(polls_data_scenarios[polls_data_scenarios['scenario'] == 'A'].drop(columns='scenario').columns) - set(polls_data_1984_2023.columns)
polls_data_scenarios[polls_data_scenarios['scenario'] == 'A'].drop(columns='scenario')
#%%
for s in np.unique(polls_data_scenarios.scenario.values):
    output_path = Path(f'{ROOT_DIR}/data/interim/scenarios/{s}/')
    output_path.mkdir(exist_ok=True, parents=True)
    pd.concat([economic_data_1984_2023, economic_data_scenarios[economic_data_scenarios['scenario'] == f'{s}']
               .drop(columns='scenario')]).reset_index(drop=True).to_csv(f'{output_path}/dataland_economic_data_1984_2024_scenario.csv')
    pd.concat([polls_data_1984_2023, polls_data_scenarios[polls_data_scenarios['scenario'] == f'{s}']
               .drop(columns='scenario')]).reset_index(drop=True).to_csv(f'{output_path}/dataland_polls_1984_2024_scenario.csv')
