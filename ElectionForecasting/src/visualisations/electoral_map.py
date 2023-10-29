import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly
from ElectionForecasting.src.root import ROOT_DIR
from ElectionForecasting.src.config import party_order as parties
from ElectionForecasting.src.config import PLOTLY_TEMPLATE

def plot_election_map(df, title_addition='', observed=None):
    have_observations = (type(observed) == pd.DataFrame)
    if df.index.name == 'province':
        df.reset_index(inplace=True)
    default_colours = plotly.colors.DEFAULT_PLOTLY_COLORS
    unique_states = df['province'].values
    colour_map = {
        parties[0]: default_colours[0],
        parties[1]: default_colours[1],
        parties[2]: default_colours[2],
        parties[3]: default_colours[3]
    }
    fig = go.Figure()

    rows, cols = 3, 4

    for i, state in enumerate(unique_states):

        state_data = df[df['province'] == state]
        color = colour_map[state_data['winner'].values[0]]
        str_format = lambda x: ', '.join([f'{k}: {v:.1f}%' for k,v in x.items()])
        individual_results = str_format((state_data[parties].fillna(0).round(2)*100).iloc[0].T.to_dict())
        breakdown = f"Election Breakdown:<br>{individual_results}"
        # Add a subplot for each state
        winner = state_data[parties].idxmax(axis=1).item()
        win_share = state_data[parties].max(axis=1).item()
        trace_text = f"Pred: {winner.upper()}<br>Probability: {100*win_share:.0f}%<br>"
        outline_color = color
        if have_observations:
            observed_winner = observed.loc[state].idxmax(axis=0)
            observed_win_share = observed.loc[state].max(axis=0)
            trace_text += f'Obs: {observed_winner.upper()}<br>Vote share {100*observed_win_share:.0f}%<br>'
            if observed_winner.upper()==winner.upper():
                outline_color = 'Green'
            else:
                outline_color = 'Red'
        fig.add_trace(
            go.Scatter(
                x=[(i //cols)],
                y=[i % cols],
                mode='markers+text',
                marker=dict(
                    symbol='square',
                    size=100,
                    color=color,
                    line=dict(
                        color=outline_color,  # Marker outline color
                        width=8               # Marker outline width
                    )
                ),
                text=trace_text,
                textposition='middle center',
                opacity=win_share,
                name=state,
                hovertext=breakdown,
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=(i //cols),
            y=i % cols + 0.5,
            text=state,
            showarrow=False,
            font=dict(size=12),
        )

    for party in parties:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(symbol='square', size=15, color=colour_map[party]),
                name=party.upper(),
                legendgroup=party,
                showlegend=True,
            )
        )


    # Update layout
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=True,
        title=f'Electoral Map{title_addition}',
        xaxis=dict(visible=False, showticklabels=False),
        yaxis=dict(visible=False, showticklabels=False),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        xaxis_ticks='',
        yaxis_ticks='',
        width=800,
        height=900,
        legend=dict(
            title='Party',
            traceorder='normal',
        )
    )
    return fig
