import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pyt

def get_lagged_mu_contribution(data, lag):
    """
    Time for change component of the model - creating a Dirichlet
    prior based on fundamentals based data. This is done by calculating
    the mu contribution with a given lag. Can combine additional lags.
    (data should contain up to 4 lagged quarters of fundamentals data)
        
    Parameters:
    - data: pd.DataFrame | Data for various economic indicators.
    - lag: int | The number of quarters to lag.
    
    Returns:
    - mu_contribution: PyMC3 variable | Calculated mu contribution.
    """
    ...
    lag_str = f'_lag_{lag}q'
    # print(data)
    gdp_growth = pm.MutableData('year_on_year_gdp_pct_change'+lag_str, data['year_on_year_gdp_pct_change'+lag_str])
    unemployment = pm.MutableData('unemployment_rate'+lag_str, data['unemployment_rate'+lag_str])
    inflation = pm.MutableData('year_on_year_inflation'+lag_str, data['year_on_year_inflation'+lag_str])
    stock_mkt = pm.MutableData('year_on_year_stock_mkt_pct_change'+lag_str, data['year_on_year_stock_mkt_pct_change'+lag_str])
    pop_vote_winner_labels = pm.MutableData('national_pop_vote_winner'+lag_str, data['national_pop_vote_winner'+lag_str])
    party_labels = pm.MutableData('party_in_power'+lag_str, data['party_in_power'+lag_str])
    number_of_parties = 4
    alpha_party = pm.Dirichlet('alpha_party'+lag_str, a=np.ones(number_of_parties))
    alpha_pop_vote = pm.Dirichlet('alpha_pop_vote'+lag_str, a=np.ones(number_of_parties))
    party_effect = pm.Categorical('party_effect'+lag_str, p=alpha_party, observed=party_labels)
    pop_vote = pm.Categorical('pop_vote'+lag_str, p=alpha_pop_vote, observed=pop_vote_winner_labels)

    # sigmas = .01
    transformed_inflation = pyt.log(pyt.sqrt(pyt.abs(inflation)))
    # betas_gdp = pm.Normal('betas_gdp'+lag_str, mu=0.05, sigma=sigmas)
    # mu_hyperparam  = pm.Normal('mu_hyperparam'+lag_str, mu=0., sigma=.05)
    # sig_hyperparam  = pm.Normal('sig_hyperparam'+lag_str, mu=.1, sigma=.05)
    # betas_inflation = pm.Normal('betas_inflation'+lag_str, mu=mu_hyperparam, sigma=sig_hyperparam)
    # betas_stk_mkt = pm.Normal('betas_stk_mkt'+lag_str, mu=0.05, sigma=sigmas)
    # betas_unemployment = pm.HalfNormal('betas_unemployment'+lag_str, sigma=sigmas)
    betas_gdp = pm.Normal('betas_gdp'+lag_str, mu=0.027, sigma=.02)
    # mu_hyperparam  = pm.Normal('mu_hyperparam'+lag_str, mu=0., sigma=.02)
    # sig_hyperparam  = pm.Normal('sig_hyperparam'+lag_str, mu=.1, sigma=.02)
    betas_inflation = pm.Normal('betas_inflation'+lag_str, mu=-1.798, sigma=.522)
    betas_stk_mkt = pm.Normal('betas_stk_mkt'+lag_str, mu=0.1, sigma=.1)
    betas_unemployment = pm.HalfNormal('betas_unemployment'+lag_str, sigma=.019)
    mu_contribution = betas_gdp * gdp_growth[:, None] + \
                      betas_unemployment * np.abs(unemployment)[:, None] + \
                      betas_stk_mkt * stock_mkt[:, None] + \
                      betas_inflation * transformed_inflation[:, None] + \
                      party_effect[:, None] + \
                      pop_vote[:, None]

    return mu_contribution

def sigmoid_combination_of_vote_share(t, n_time_points, gam_forecasts, observed_vote_share, switchpoint_min=.25, switchpoint_max=.75):
    switchpoint_diff = switchpoint_max - switchpoint_min
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    sig=pyt.sqrt(n_time_points).round()/25
    mu=(1*(n_time_points/4)).round()
    beta = pm.Normal('beta', mu=mu, sigma=sig)
    # Compute weight
    w_t = pm.Deterministic('w_t', (1 / (1 + pyt.exp(-alpha * (t - beta)))))
    # change the bounds of the sigmoid function
    w_t = pm.Deterministic('w_t_rescaled', switchpoint_min + switchpoint_diff * w_t)
    # Compute combined estimates
    a=(w_t[None,:,None] * gam_forecasts)
    b=((1 - w_t)[None,:,None] * observed_vote_share)
    return a+b

# def linear_combination_of_vote_share(
#         t, n_time_points, gam_forecasts, observed_vote_share,
#         switchpoint_min=.25, switchpoint_max=.75
#     ):
#     switchpoint_diff = switchpoint_max - switchpoint_min
#     # change the bounds of the linear function
#     norm_t = switchpoint_min + switchpoint_diff * (t/n_time_points)
#     a = gam_forecasts * norm_t[None,:,None]
#     b = observed_vote_share * (1-norm_t)[None,:,None]
#     return a+b

def linear_combination_of_vote_share(
        t, n_time_points, gam_forecasts, observed_vote_share,
        switchpoint_min, switchpoint_max
    ):
    switchpoint_diff = switchpoint_max - switchpoint_min
    # change the bounds of the linear function
    norm_t = switchpoint_min + switchpoint_diff * (t/n_time_points)
    a = gam_forecasts* norm_t[:, None, None]
    b = observed_vote_share[None,:] * (1-norm_t)[:,None, None]
    return a+b

def add_data(
            gam_forecasts,
            vote_shares,
            previous_year_pleans
        ):
    # Adding mutable data to the model
    # These can be updated at inference level to allow for inference on new data
    pm.MutableData('y_data', vote_shares.mean(axis=1), dims=('year', 'all_parties'))
    pm.MutableData('previous_year_pleans', previous_year_pleans)
    pm.MutableData('gam_forecasts', gam_forecasts, dims=('polling_date', 'all_parties'))
    does_compete = np.ones((12, 4))
    does_compete[3:, 3:] = 0
    # Adding constant data to the model
    # This is a immutable component of the Dataland electoral system
    pm.ConstantData('does_compete', does_compete)

def add_coordinates(model, train_df, gam_forecasts):
    model.add_coord('year', train_df.election_cycle.values, mutable=True)
    model.add_coord("polling_date", gam_forecasts.index, mutable=True)
    for c in train_df.columns:
        model.add_coord(c, train_df[c].values, mutable=True)

def normalise(tensor, axis=-1):
    """Normalizing array along the specified axis"""
    return tensor / tensor.sum(axis=axis, keepdims=True)

def min_max(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min())

def proportion(tensor, axis=-1):
    min_maxed = min_max(tensor)
    normed = normalise(min_maxed, axis=axis)
    return normed

def combine_with_pleans(national_prior, partisan_leans):
    vote_share_per_state = national_prior[0,:,None,:] + partisan_leans[None,:,:]
    return proportion(vote_share_per_state)

def time_for_change_model(
        gam_forecasts,
        train_df: pd.DataFrame,
        vote_shares,
        previous_year_pleans
    ) -> pm.Model:
    """
    Fit a Bayesian hierarchical model combining fundamentals and polls based predictors
    of the Dataland presidential election.
    
    Parameters:
    - gam_forecasts: Poll forecasts from a generalized additive model.
    - train_df: Fundamentals training data.
    - full_state_votes: Historical data on state specific electoral results.
    - vote_shares: Previous year's state specific vote shares, to be used in alpha prior. This could be omitted.
    - previous_year_pleans: Partisan leans from the previous year.
    
    Returns:
    - PyMC3 Model object
    """
    with pm.Model(coords={'all_days_till_the_election': gam_forecasts.index[::-1]}) as model:
        ### Welcome to the world of probabilistic programming.
        ### please leave your coats, shoes, and singularly defined
        ### variables at the door.

        # Prepare the PyMC model context
        add_coordinates(model, train_df, gam_forecasts)
        add_data(
            gam_forecasts.iloc[:,:4].values,
            vote_shares,
            previous_year_pleans
        )
        # A binary mask to distinguish between parties that do and do not contest in each province
        does_compete = model['does_compete']

        # Time for change section of the model
        alpha_prior_mean = np.full((4, ), .25)
        sigma_prior_mean = np.full((4, ), .25)
        alphas = pm.Normal('alphas', mu=alpha_prior_mean,  sigma=sigma_prior_mean)
        lagged_contributions = get_lagged_mu_contribution(train_df, lag=3)
        mus = pm.Deterministic('mus', pm.math.exp(alphas + lagged_contributions))
        dirichlet_alphas = pm.Deterministic('dirichlet_alphas', mus)

        # Fundamentals based national vote share prediction
        observed_vote_share = pm.Dirichlet('observed_vote_share', a=dirichlet_alphas, observed=model['y_data']) 

        # Combination of fundamentals based prediction with polling data
        poll_forecasts = model['gam_forecasts']
        time_for_change_prior = observed_vote_share[-1, None, :]
        n_days_of_polling = model.dim_lengths['polling_date']
        n_time_points = model.dim_lengths['all_days_till_the_election']
        t = pyt.arange(1, n_time_points + 1)
        t = t[-n_days_of_polling:]
        # Define the max and min for alpha switching
        # These can be replaced by RVs for a more Bayesian approach
        alpha_switch_max = 0.85
        alpha_switch_min = 0.35

        # I weight the parties national vote share since the SSP's nationally
        # observed polling results should be proportionally increased in the
        # states in which they do compete. This is a flawed assumption, and 
        # warrants more attention.
        does_compete_national = does_compete.mean(axis=0)
        poll_forecasts_state_wise = (
                poll_forecasts[:, None, :] *
                (does_compete_national*does_compete)[None, :]
        )

        # using a linear function to combine these predictors. Can adapt to
        # more complex functions.
        daily_obs_vote_share_state_wise = pm.Deterministic(
            'daily_obs_vote_share',
                linear_combination_of_vote_share(
                    t, n_time_points, poll_forecasts_state_wise,
                    time_for_change_prior,
                    switchpoint_min=alpha_switch_min,
                    switchpoint_max=alpha_switch_max
            )
        )

        # # using a linear function to combine these predictors. Can adapt to
        # # more complex functions.
        # daily_obs_vote_share = pm.Deterministic(
        #     'daily_obs_vote_share',
        #         linear_combination_of_vote_share(
        #             t, n_time_points, poll_forecasts,
        #             time_for_change_prior,
        #             switchpoint_min=alpha_switch_min,
        #             switchpoint_max=alpha_switch_max
        #     )
        # )

        # # I weight the parties national vote share since the SSP's nationally
        # # observed polling results should be proportionally increased in the
        # # states in which they do compete. This is a flawed assumption, and 
        # # warrants more attention.
        # does_compete_national = does_compete.mean(axis=0)
        # daily_obs_vote_share_state_wise = (
        #         daily_obs_vote_share[0][:, None, :] *
        #         (does_compete_national*does_compete)[None, :]
        # )

        # The state specific vote shares is then taken as the sum of the previous
        # year's partisan leans and the observed national vote share projections.
        # Incorporating the covariance martix into this calculation induced significant
        # model instability, and adversely affected model results, so was omitted, however
        # this is worth more attention.
        partisan_leans = model['previous_year_pleans'] 
        pm.Deterministic('state_vote_share', 
                            proportion(
                                    daily_obs_vote_share_state_wise +
                                    (
                                        partisan_leans[None, :, :] *
                                        does_compete[None, :, :]
                                    )
                                )*does_compete[None, :, :])
    return model
