import pandas as pd
import numpy as np
from ElectionForecasting.src.config import party_order
def get_election_results_df(path_list):
    lst_results_dfs = []
    dict_results_dfs = {}
    for i, (year, file) in enumerate(path_list):
        results = pd.DataFrame()
        national_vote = ""
        election = pd.read_csv(file.as_posix(), index_col=0)
        election.columns = election.columns.str.replace('_share','')
        results = election.copy()
        for c in election.columns:
            national_vote += f"{c}: {election[c].fillna(0).mean().round(2)*100:.1f}% "
        results['winner'] = election.idxmax(axis=1)
        results['vote_share'] = election.max(axis=1)
        lst_results_dfs.append(results)
        results.reset_index(inplace=True)
        dict_results_dfs[year] = [results, national_vote]
    return dict_results_dfs


def get_results_df_from_simulation(simulations, province_order, key='Simulation'):
    lst_results_dfs = []
    dict_results_dfs = {}
    n_errors = 0
    for i, simulation in enumerate(simulations):
        national_vote = ""
        election = pd.DataFrame(simulation, index=province_order, columns=party_order)
        results = election.copy()
        election.index.name = 'province'
        for c in election.columns:
            national_vote += f"{c}: {election[c].fillna(0).mean().round(2)*100:.1f}% "
        results['winner'] = election.idxmax(axis=1)
        results['vote_share'] = election.max(axis=1)
        results.index.name = 'province'
        lst_results_dfs.append(results)
        results.reset_index(inplace=True)
        dict_results_dfs[f"{key}: {i}"] = [results, national_vote]
    return dict_results_dfs
