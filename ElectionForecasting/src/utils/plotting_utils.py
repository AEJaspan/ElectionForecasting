import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objs as go
from matplotlib import colors as mcolors
from ElectionForecasting.src.config import party_order, province_order
from matplotlib.pyplot import savefig
from plotly.io import write_html, write_image
from ElectionForecasting.src.root import ROOT_DIR
from ElectionForecasting.src.visualisations.scatter_plots import plot_single_region_plotly
from ElectionForecasting.src.visualisations.scatter_plots import add_new_y_to_plotly_with_traces

colors = list(mcolors.CSS4_COLORS)
def violin_plot(observations, posterior_samples, title='', ):
    df_posterior = pd.DataFrame(posterior_samples, columns=[p.upper() for p in party_order])

    # Melt the DataFrame for easier plotting
    df_melted = df_posterior.melt(var_name='Party', value_name='Posterior Vote Share')

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Party', y='Posterior Vote Share', data=df_melted, inner='quartile', palette='muted', hue='Party')

    # Overlay the actual data points
    for j, party in enumerate([p.upper() for p in party_order]):
        plt.scatter(party, observations[party].item(), color='red', s=100, zorder=5, label='Observed' if j == 0 else "")
    # Add title and labels
    plt.title(title)#'Posterior Predicted Vote Shares with Actual Data Points')
    plt.xlabel('Party')
    plt.ylabel('Vote Share')
    plt.legend()
    # Show the plot
    # plt.show()
    errors = df_melted.groupby('Party').mean().T.reset_index(drop=True) - observations.reset_index(drop=True)
    return errors


def plot_log_loss(mean_log_loss_sorted):
    event_labels_sorted = mean_log_loss_sorted.loc[province_order].index
    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mean_log_loss_sorted, y=event_labels_sorted, palette="viridis", hue=event_labels_sorted)
    # Add vertical lines for log loss values
    log_loss_4_classes = np.log(4)  # Approx 1.3863
    log_loss_3_classes = np.log(3)  # Approx 1.0986
    plt.axvline(x=log_loss_4_classes, color='r', linestyle='--', label=f'Random guess in 4 (balanced) Classes: {log_loss_4_classes:.2f}')
    plt.axvline(x=log_loss_3_classes, color='g', linestyle='--', label=f'Random guess in 3 (balanced) Classes: {log_loss_3_classes:.2f}')
    plt.xlabel('Mean Log Loss')
    plt.title('Mean Log Loss for Each Event (Sorted)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # plt.show()


def plot_log_likelihood(df):
    df = df.sort_values()
    event_labels_sorted = df.index
    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df, y=event_labels_sorted, palette="viridis", hue=event_labels_sorted)
    plt.xlabel('Log likelihood')
    plt.title('Log likelihood scores by geography')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()


def plot_prob_band(predictions, observed, time_index=None):
    """plot the expected result on each day

    Args:
        predictions (np.ndarray): idx 0: time_axis, idx 1: mc samples, idx 3: vote share
        observed (np.ndarray, optional): observed vote share. Defaults to None.
        time_index (datetime64, optional): time axis - will be replaced by incremental if not a datetime64 array.
    """
    observed = observed.loc[party_order]
    mean_values = np.mean(predictions, axis=0)  # Mean along the samples axis
    # lower_bound = np.percentile(predictions, 2.5, axis=0)  # 2.5th percentile
    # upper_bound = np.percentile(predictions, 97.5, axis=0)  # 97.5th percentile
    l = 2.5 # 32.5
    lower_bound = np.percentile(predictions, l, axis=0)  # 2.5th percentile
    upper_bound = np.percentile(predictions, 100-l, axis=0)  # 97.5th percentile
    if type(time_index) != pd.core.indexes.datetimes.DatetimeIndex:
        time_index = np.arange(predictions.shape[1])[::-1]
    # Plotting
    plt.figure(figsize=(14, 10))
    # Loop over each parameter
    for i in range(4):    
        plt.fill_between(time_index, lower_bound[:, i], upper_bound[:, i], color='gray', alpha=0.5)
        plt.plot(time_index, mean_values[:, i], label=f'predicted {observed.index[i]}')
        
        # plt.(f'Time Series for {parameter_names[i]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
    # plt.gca().hlines(observations.values, min(time_index), max(time_index))
    for i, (l, o) in enumerate(observed.iloc[:,0].to_dict().items()):
        plt.gca().hlines(pd.Series(o), min(time_index), max(time_index), linestyle='--', colors=colors[i*10], label=f"observed {l}")
    plt.legend()

    plt.tight_layout()

def create_plotly_traces(df, yaxis='y1', trace_name='Win Probability'):
    """
    Creates a list of Plotly Scatter traces from a Pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        yaxis (str): The ID of the y-axis to plot against (default is 'y1').
        
    Returns:
        list: A list of Plotly Scatter traces.
    """
    traces = []  # Initialize an empty list to hold the Scatter traces
    
    # Loop through each column specified
    for column in df.columns:
        # Create a Plotly Scatter trace for each column
        trace = go.Scatter(
            x=df.index,  # Use DataFrame's index as the x-values
            y=df[column],  # Use the current column's values as the y-values
            name=f"{column} {trace_name}",  # Set the trace name
            hovertext=column,  # Set the hovertext
            yaxis=yaxis  # Associate this trace with the specified y-axis
        )
        
        traces.append(trace)  # Add the created trace to the list of traces

    return traces

def combined_scatter_and_traces(raw_poll_data, region, poll_columns, poll_dates, daily_wins, observed=None):
    fig = plot_single_region_plotly(raw_poll_data, poll_dates[-1], poll_dates[0], region, poll_columns)
    vote_shares = daily_wins.loc[:, daily_wins.columns.str.contains('_mean_vote_share')]
    traces = create_plotly_traces(vote_shares, yaxis='y1')
    win_probs = daily_wins.loc[:, daily_wins.columns.str.contains('_win_probability')]
    # traces += create_plotly_traces(win_probs, yaxis='y2')
    if type(observed) == pd.DataFrame:
        if observed.shape[0] == 1:
            observed = pd.concat([observed, observed])
            observed.index = [poll_dates[0], poll_dates[-1]]
        traces += create_plotly_traces(observed, yaxis='y1', trace_name='Observed Result')
    fig = add_new_y_to_plotly_with_traces(fig, traces)
    return fig

def close_figure(figure, save_function=savefig, directory=ROOT_DIR, filename='', save=False, show=False) -> None:
    if show:
        print('Showing plot...')
        plt.show()
    if save:
        if save_function==write_html:
            file_path = directory / f'{filename}.html'
            print(f'Saving plot to {file_path}...')
            figure.write_html(file_path)
        elif save_function == write_image:
            file_path = directory / f'{filename}.png'
            print(f'Saving plot to {file_path}...')
            figure.write_image(file_path)
        elif save_function==savefig:
            file_path = directory / f'{filename}.png'
            print(f'Saving plot to {file_path}...')
            plt.savefig(file_path)
        else:
            raise ValueError('Unknown save function')
    if save_function==savefig:
        plt.close()