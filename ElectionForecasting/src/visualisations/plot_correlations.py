import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ElectionForecasting.src.config import party_order, province_order
from ElectionForecasting.src.root import ROOT_DIR
from scipy.stats import pearsonr
import plotly.graph_objects as go
from ElectionForecasting.src.config import PLOTLY_TEMPLATE

# Modify the function to make the figure square and adjust the orientation of the diagonal
def plot_significant_correlations_square(corr_matrix, p_values_matrix):
    """
    Plots a heatmap of the correlation matrix without annotations, overlaying p-values where appropriate
    on hover. Only shows significant correlations (based on the p_values_matrix). The figure is square
    and the diagonal runs from top-left to bottom-right.
    
    Parameters:
    - corr_matrix: DataFrame containing correlations
    - p_values_matrix: DataFrame containing p-values
    
    Returns: Plotly Figure
    """
    
    # Convert DataFrames to numpy for easier element-wise operations
    corr_values = corr_matrix.to_numpy()
    p_values = p_values_matrix.to_numpy()
    
    # Create hovertext matrix: show correlation and p-value for each cell
    hovertext = []
    for i in range(corr_values.shape[0]):
        hovertext_row = []
        for j in range(corr_values.shape[1]):
            text = f"{corr_matrix.columns[i].replace('_', ' ').replace(' share', '')} - {corr_matrix.index[j].replace('_', ' ').replace(' share', '')}<br>Correlation: {corr_values[i, j]:.4f}<br>P-Value: {p_values[i, j]:.4f}"
            hovertext_row.append(text)
        hovertext.append(hovertext_row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
                    z=np.flipud(corr_values),  # Flip the matrix upside down for correct orientation
                    x=corr_matrix.columns.str.replace('_', ' ').str.replace(' share', ''),
                    y=corr_matrix.index[::-1].str.replace('_', ' ').str.replace(' share', ''),  # Invert for correct orientation
                    hoverinfo='text',
                    text=np.flipud(hovertext),  # Flip the hovertext as well
                    reversescale=True,
                    colorscale='RdBu')) # Agsunset
    
    # Add titles and labels
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='Pairwise Correlation Matrix with P-Values',
        xaxis=dict(title='Party & State', tickfont=dict(size=7)),
        yaxis=dict(title='Party & State', tickfont=dict(size=7)),
        autosize=False,
        width=800,
        height=800  # Square figure
    )
    
    return fig

# Function to calculate the correlation and p-value
def calculate_correlation_and_pvalue(df, column1, column2):
    corr, p_value = pearsonr(df[column1], df[column2])
    return corr if p_value < 0.05 else 0

# Convert the multi-index to a single concatenated index
def concatenate_multi_index(multi_index):
    return [f"{state}_{party}" for state, party in multi_index]

def plot_heatmap(df, title='Pairwise Correlations Across All States and Parties'):
    # Plot the large correlation matrix as a heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(df.fillna(0), annot=False, cmap='coolwarm', cbar=True, square=True)
    plt.title(title)
    plt.show()

