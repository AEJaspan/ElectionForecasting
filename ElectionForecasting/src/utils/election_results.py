"""
Variable Dictionary:

parties: political parties participating in the election.

territories: States providing 1 electoral collage vote each.

restricted_party: The political party that is restricted to competing
                  in a subset of the territories.

restricted_territories: The territories in which said `restricted_party`
                        can compete in.

unrestricted_territories: The remaining territories in with `restricted_party`
                          does not compete.

state_wins: The average number of times each party wins in each state.

president_wins: The average number of times each party wins the presidency (wins
                in the most states).

nvs: List of national vote shares for each party.

p_lean / plean: The partisan leanings of each state (how much each state deviates
                away from the national vote).
"""


import pandas as pd
import numpy as np
from ElectionForecasting.src.config import parties as cols
from ElectionForecasting.src.config import territories as index_vals
from ElectionForecasting.src.config import restricted_territories
from ElectionForecasting.src.config import restricted_party

unrestricted_territories = list(set(index_vals) - set(restricted_territories))
    
def calculate_average_wins(dict_results_dfs):
    state_wins = pd.Series(np.array([0,0,0,0]), index=cols)
    multi_index = pd.MultiIndex.from_product([index_vals, cols], names=["Territory", "Party"])
    wins_per_state = pd.Series(np.array([0]*len(multi_index)), index=multi_index)
    president_wins = pd.Series(np.array([0,0,0,0]), index=cols)
    n_simulations = len(dict_results_dfs.items())
    for result in dict_results_dfs.values():
        itteration_state_wins = result[0].winner.value_counts()
        presidential_winner = itteration_state_wins.idxmax()
        president_wins.loc[presidential_winner] += 1
        state_wins.loc[itteration_state_wins.index.values] += itteration_state_wins
        a=result[0].set_index('State')['winner']
        wins_per_state[[*list(zip(a.index.values, a.values))]]+=1
    state_wins/=n_simulations
    president_wins/=n_simulations
    wins_per_state/=n_simulations
    average_result = wins_per_state.unstack(level=1)[cols].loc[index_vals]
    return state_wins, president_wins, average_result

def average_simulation_votes(simulations):
    assert len(simulations.shape) == 3
    average_state_results = pd.DataFrame(
        simulations.mean(axis=0), index=index_vals, columns=cols
    )
    results = average_state_results.copy()
    results['winner'] = average_state_results.idxmax(axis=1)
    results['vote_share'] = average_state_results.max(axis=1)
    results.index.name = 'State'
    return results

def average_data_votes(path_list):
    a = []
    for p in path_list:
        a.append(pd.read_csv(p[1].as_posix(), index_col=0).values)
    a = np.stack(a)
    return average_simulation_votes(a)

def get_election_results_df(path_list):
    lst_results_dfs = []
    dict_results_dfs = {}
    for i, (year, file) in enumerate(path_list):
        results = pd.DataFrame()
        national_vote = ""
        election = pd.read_csv(file.as_posix(), index_col=0)
        results = election.copy()
        for c in election.columns:
            national_vote += f"{c}: {election[c].fillna().mean().round(2)*100:.1f}% "
        results['winner'] = election.idxmax(axis=1)
        results['vote_share'] = election.max(axis=1)
        lst_results_dfs.append(results)
        results.reset_index(inplace=True)
        dict_results_dfs[year] = [results, national_vote]
    return dict_results_dfs


def get_results_df_from_simulation(simulations, key='Simulation'):
    lst_results_dfs = []
    dict_results_dfs = {}
    n_errors = 0
    for i, simulation in enumerate(simulations):
        national_vote = ""
        election = pd.DataFrame(simulation, index=index_vals, columns=cols)
        # election[restricted_party].loc[unrestricted_territories] = np.nan
        results = election.copy()
        election.index.name = 'State'
        for c in election.columns:
            national_vote += f"{c}: {election[c].fillna(0).mean().round(2)*100:.1f}% "
        results['winner'] = election.idxmax(axis=1)
        results['vote_share'] = election.max(axis=1)
        results.index.name = 'State'
        lst_results_dfs.append(results)
        results.reset_index(inplace=True)
        dict_results_dfs[f"{key}: {i}"] = [results, national_vote, election]
    return dict_results_dfs
