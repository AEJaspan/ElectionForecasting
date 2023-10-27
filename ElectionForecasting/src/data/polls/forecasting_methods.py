from patsy import dmatrix
import pymc as pm
import numpy as np
import pytensor.tensor as tt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

target_columns = ['cc_poll_share', 'ssp_poll_share', 'pdal_poll_share', 'undecided_poll_share', 'dgm_poll_share']
predictor_columns = ['cc_poll_share_lag', 'ssp_poll_share_lag', 
                    'dgm_poll_share_lag', 'pdal_poll_share_lag', 
                    'undecided_poll_share_lag', 'days_until_election']

def statsmodels_gam(data, DoF=5, degree=3):
    # N knots: DoF - degree - 1
    prediction_vs_reality = {}
    for party in target_columns:
        
        # Prepare the data
        # X = data[predictor_vars]
        y = data[party]
        date = data.date
        # Skip if target variable has missing values
        if y.isnull().any():
            continue
        
        X = data[predictor_columns]
        
        # Skip if predictor variables have missing values
        if X.isnull().any().any():
            continue

        # Fit the GAM model
        try:
            gam_bs = BSplines(X, df=[30] * len(predictor_columns), degree=[5] * len(predictor_columns))
            gam_model = GLMGam(y, sm.add_constant(X), smoother=gam_bs).fit()
        except Exception as e:
            if type(e)==sm.tools.sm_exceptions.PerfectSeparationError:
                try:
                    gam_bs = BSplines(X[:-1], df=[DoF] * len(predictor_columns), degree=[degree] * len(predictor_columns))
                    gam_model = GLMGam(y, sm.add_constant(X[:-1]), alpha=1.0, smoother=gam_bs).fit()
                except Exception as e:
                    prediction_vs_reality[party] = {'y': y, 'y_pred': y.shift(-1).ffill(), 'date': date}
                    continue
            prediction_vs_reality[party] = {'y': y, 'y_pred': y.shift(-1).ffill(), 'date': date}
            continue
        
        # Generate predictions and evaluate the model
        y_pred = gam_model.predict(exog=sm.add_constant(X), exog_smooth=X)
        # mae = mean_absolute_error(y, y_pred)
        # rmse = sqrt(mean_squared_error(y, y_pred))
        prediction_vs_reality[party] = {'y': y, 'y_pred': y_pred, 'date': date}
        # Store the evaluation metrics
    return prediction_vs_reality



def pymc_gam(data, DoF=5, degree=3, percentile=97.5):
    ##### NOTE ######
    # - THIS IS FOR FUTURE WORK - A POINT ESTIMATE IS 
    # MORE COMPUTATIONALLY EFFICIENT IF JUST TAKING THE 
    # MEAN VALUE - IN FUTURE THESE PROBABILITIES COULD 
    # BE INCORPORATED INTO THE MODEL DIRECTLY, GIVING A 
    # BETTER UNCERTAINTY ESTIMATION IN THE MODEL

    # Load your data
    df = data.copy()
    prediction_vs_reality = {}
    # Prepare predictor and target variables
    predictors = df[predictor_columns].values
    target = df[target_columns].values


    # Generate spline basis functions for 'days_until_election'
    design_matrix = dmatrix(f"bs(days_until_election, df={DoF}, degree={degree}) - 1", data=df, return_type='dataframe')

    # Update predictors to include spline terms
    predictors_spline = df[predictor_columns].copy()
    predictors_spline = pd.concat([predictors_spline, design_matrix], axis=1)
    predictors_spline = predictors_spline.values

    # Define the Bayesian model with splines
    with pm.Model() as model_spline:
        
        # Priors for fixed effects
        alpha = pm.Normal('alpha', mu=0, sigma=10, shape=len(target_columns))
        beta = pm.Normal('beta', mu=0, sigma=10, shape=(predictors_spline.shape[1], len(target_columns)))
        
        # Linear model equation with spline terms
        mu = alpha + pm.math.dot(predictors_spline, beta)
        
        # Dirichlet likelihood
        likelihood = pm.Dirichlet('likelihood', a=tt.exp(mu), observed=target)

    # Perform inference
    with model_spline:
        trace_spline = pm.sample(1000, tune=1000)

    # Summary of the model parameters
    pm.summary(trace_spline)
    # Generate predictions based on the posterior samples
    posterior_samples = pm.sample_posterior_predictive(trace_spline, model=model_spline)
    posterior_samples = posterior_samples.posterior_predictive.likelihood.values
    posterior_samples = posterior_samples.reshape([-1]+list(posterior_samples.shape[-2:]))
    # Calculate the median and 95% prediction intervals
    median_pred = np.median(posterior_samples, axis=0)
    # lower_pred = np.percentile(posterior_samples, 100-percentile, axis=0)
    # upper_pred = np.percentile(posterior_samples, percentile, axis=0)
    for i, target_col in enumerate(target_columns):
        prediction_vs_reality[target_col] = {'y': target[:,i], 'y_pred': median_pred[:,i], 'date': data.date}
    return prediction_vs_reality


def particle_filter(data, N=10000):
    prediction_vs_reality = {}
    df=data.copy()
    # Observed vote shares
    observations = df[target_columns].values

    # Number of parties (based on the columns in the dataframe)
    K = len(target_columns)

    # Initialize particles and weights
    particles = np.random.dirichlet(np.ones(K), N)
    weights = np.ones(N) / N

    # Placeholder for forecasts and uncertainty estimates
    forecasts = []
    forecast_variances = []

    # Loop over observations
    for y in observations:
        # Prediction step: Evolve particles
        particles = 0.5 * particles + 0.5 * y + np.random.normal(0, 0.01, (N, K))
        particles = np.clip(particles, 0, 1)
        particles /= np.sum(particles, axis=1)[:, np.newaxis]
        
        # Update step: Update weights based on the Dirichlet likelihood
        likelihood = np.exp(np.sum((y - 1) * np.log(particles + 1e-10), axis=1))
        weights *= likelihood
        weights += 1e-10
        weights /= np.sum(weights)
        
        # Resampling step
        indices = np.random.choice(np.arange(N), size=N, p=weights)
        particles = particles[indices]
        weights = np.ones(N) / N
        
        # State estimation: Weighted mean of particles
        estimate = np.average(particles, axis=0, weights=weights)
        forecasts.append(estimate)
        
        # Uncertainty estimation: Weighted variance of particles
        variance = np.average((particles - estimate)**2, axis=0, weights=weights)
        forecast_variances.append(variance)

    # Convert forecasts and variances to NumPy arrays
    forecasts = np.array(forecasts)
    forecast_variances = np.array(forecast_variances)

    for i, target_col in enumerate(target_columns):
        prediction_vs_reality[target_col] = {'y': observations[:,i], 'y_pred': forecasts[:,i], 'date': data.date}

    return prediction_vs_reality


