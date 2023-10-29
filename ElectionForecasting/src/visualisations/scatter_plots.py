
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.express as px
import plotly.graph_objects as go
from ElectionForecasting.src.root import ROOT_DIR
from ElectionForecasting.src.config import PLOTLY_TEMPLATE
import logging
logging.basicConfig(level=logging.INFO)


def plot_with_rolling_avg(ax, data, x_col, y_col, weight_col, label, color, window=3):
    """Plot scatter plots with rolling average trend lines."""
    data = data.sort_values(by=[x_col])
    sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax, label=label, alpha=0.6)
    # Calculate weighted rolling average
    data['weighted_avg'] = (data[y_col] * data[weight_col]).rolling(window=window).sum() / data[weight_col].rolling(window=window).sum()
    # Interpolate to fill missing values
    data['weighted_avg'].interpolate(inplace=True)
    ax.plot(data[x_col], data['weighted_avg'], f"{color}--")
    # Format x-axis labels
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis='x', rotation=45)


def plot_single_region(df, start_date, end_date, region, poll_share_columns, color_dict, window=3):
    """Plot the poll share for a single region and specified date range."""
    """
    Plot the poll share for specified date range and regions.
    
    Parameters:
    - df: DataFrame containing the poll data
    - start_date: Start date for the date range
    - end_date: End date for the date range
    - regions: List of regions to filter the data
    - poll_share_columns: List of poll share columns to plot
    - window: Window size for rolling average
    """
    # Filter the data based on the specified date range and regions
    filtered_df = df[(df['date_conducted'] >= start_date) & (df['date_conducted'] <= end_date) & (df['geography'] == region)]
    
    # Create a single plot for the specified region
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if not filtered_df.empty:
        for col in poll_share_columns:
            plot_with_rolling_avg(ax, filtered_df, 'date_conducted', col, 'sample_size', col, color_dict.get(col, 'k'), window)
        ax.set_title(f'{region} Poll Share for {start_date} to {end_date} with Rolling Average Trend Lines')
        ax.set_xlabel('Date Conducted')
        ax.set_ylabel('Poll Share')
        ax.legend()
            
    plt.tight_layout()
    plt.show()

def plot_single_region_plotly(df, start_date, end_date, region, poll_share_columns, window=3):
    """
    Plot the poll share for a single region and specified date range using Plotly.
    """
    """
    Plot the poll share for a single region and specified date range using Plotly.
    Parameters:
        df: DataFrame containing the data
        start_date: Start date for the plot
        end_date: End date for the plot
        region: The region to plot
        poll_share_columns: Columns containing the poll share data
        group_legend: Boolean indicating whether to group poll shares and trend lines in the legend
        window: Window size for rolling average
    """    
    filtered_df = df.copy()
    filtered_df.set_index('date_conducted', inplace=True)
    filtered_df.sort_index(inplace=True)
    fig = go.Figure()

    filtered_df = filtered_df[
        (filtered_df.index >= start_date) &
        (filtered_df.index <= end_date) &
        (filtered_df['geography'] == region)
    ]
    
    if not filtered_df.empty:
        # Create hover text
        hover_text = []
        for index, row in filtered_df.iterrows():
            hover_text.append(
                f"Population Surveyed: {row['population_surveyed']}<br>" +
                f"Geography: {row['geography']}<br>" +
                f"Sample Size: {row['sample_size']}<br>" +
                f"Pollster: {row['pollster']}<br>" +
                f"Sponsor: {row['sponsor']}"
            )
        filtered_df.loc[:, 'hover_text'] = hover_text
        
        # Initialize figure
        
        # Add scatter plots and trend lines for each poll share column
        for col in poll_share_columns:
            legend_group = col
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=filtered_df.index, 
                    y=filtered_df[col],
                    mode='markers',
                    name=col,
                    legendgroup=legend_group,
                    hovertext=filtered_df['hover_text']
                )
            )
            
            # Calculate weighted rolling average
            filtered_df.loc[:, 'weighted_avg'] = (filtered_df[col] * filtered_df['sample_size']).rolling(window=window).sum() / filtered_df['sample_size'].rolling(window=window).sum()
            # Interpolate to fill missing values
            filtered_df.loc[:,'weighted_avg'].interpolate(inplace=True)
            
            # Add trend line
            fig.add_trace(
                go.Scatter(
                    x=filtered_df.index,
                    y=filtered_df['weighted_avg'],
                    mode='lines',
                    hovertext=f"{col} (Trend)",
                    name=f"{col} (Trend)",
                    legendgroup=legend_group
                )
            )
        
        # Update layout
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"{region} Poll Share for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            xaxis_title='Date Conducted',
            yaxis_title='Poll Share'
        )
        return fig
        fig.show()
    logging.warning('Plot failed. The filtered DataFrame was empty.'
                  'Perhaps there are no polls for the selected'
                  f'region ({region}) and date range?'
                  f"({start_date.strftime('%Y-%m-%d')} to"
                  f"{end_date.strftime('%Y-%m-%d')})")
    return fig

def add_new_y_to_plotly_with_traces(fig, traces, title='Title'):
    # Calculate the minimum and maximum values for the primary y-axis dynamically
    y_values = [trace.y for trace in fig.data if hasattr(trace, 'y') and trace.y is not None]
    [fig.add_trace(t) for t in traces]
    if not y_values:
        return fig
    min_val = min(min(y) for y in y_values)
    max_val = max(max(y) for y in y_values)
    primary_yaxis_range = [min_val, max_val]

    # Update the layout to include the new y-axis
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis=dict(
            range=primary_yaxis_range
        ),
        yaxis2=dict(
            title=title,
            overlaying='y',
            side='right',
            # range=primary_yaxis_range
            domain=[0, 0.9]  # Adjust this to add padding to the right of the secondary y-axis
        ),
        legend=dict(
            x=1,  # Adjust this to move the legend horizontally
            xanchor='left'  # 'auto', 'left', 'center', or 'right'
        )
    )
    return fig
