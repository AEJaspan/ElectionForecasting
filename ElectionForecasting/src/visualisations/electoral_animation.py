import plotly.graph_objects as go
import plotly
from ElectionForecasting.src.config import party_order as parties
from ElectionForecasting.src.root import ROOT_DIR
from ElectionForecasting.src.config import PLOTLY_TEMPLATE

class ElectionResultsAnimation:
    def __init__(self, results_dict, title_addition=""):
        self.results_dict = results_dict
        self.default_colours = plotly.colors.DEFAULT_PLOTLY_COLORS
        self.colour_map = {
            parties[0]: self.default_colours[0],
            parties[1]: self.default_colours[1],
            parties[2]: self.default_colours[2],
            parties[3]: self.default_colours[3]
        }
        self.Title = f"Electoral Map {title_addition}"
        self.rows, self.cols = 3, 4
        self.fig = go.Figure()

    def make_trace(self, state_data, state, n):
        color = self.colour_map[state_data['winner'].values[0]]
        str_format = lambda x: ', '.join([f'{k}: {v:.1f}%' for k,v in x.items()])
        individual_results = str_format((state_data[parties].fillna(0).round(2)*100).iloc[0].T.to_dict())
        breakdown = f"Election Breakdown:<br>{individual_results}"
        trace = go.Scatter(
                x=[(n // self.cols)],
                y=[n % self.cols],
                mode='markers+text',
                marker=dict(symbol='square', size=100, color=color),
                text=f"{state_data['vote_share'].iloc[0].round(2) * 100:.1f}%",
                textposition='middle center',
                name=state,
                showlegend=False,
                hovertext=breakdown,
            )        
        return trace

    def make_annotation(self, state, n):
        return go.layout.Annotation(
                    x=(n // self.cols),
                    y=(n % self.cols) + 0.5,
                    text=state,
                    showarrow=False,
                    font=dict(size=12),
                )

    def make_layout(self, subtitle):
        title=go.layout.Title(
            text=f"{self.Title}<br><sup>{subtitle}</sup>",
            xref="paper",
            x=0
        )
        return go.Layout(title=title)

    def create_animation_frames(self):
        animation_frames = []

        for i, (key, (results_df, popular_vote)) in enumerate(self.results_dict.items()):
            results_df = results_df.copy()
            unique_states = results_df['province'].values

            traces = []
            for j, state in enumerate(unique_states):
                state_data = results_df[results_df['province'] == state]
                trace = self.make_trace(state_data, state, j)
                annotation = self.make_annotation(state, j)
                traces.append(trace)
                self.fig.add_annotation(annotation)
            frame_layout = self.make_layout(subtitle=f"{key}: Popular vote: {popular_vote}")
            legend_traces = self.make_legend()
            traces+=legend_traces
            animation_frame = go.Frame(data=traces, name=str(i), layout=frame_layout)
            animation_frames.append(animation_frame)

        return animation_frames

    def make_legend(self):
        legend_traces = []
        for party in parties:
            legend_traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(symbol='square', size=10, color=self.colour_map[party]),
                    name=party,
                    legendgroup=party,
                    showlegend=True,
                )
            )
        return legend_traces

    def create_slider_steps(self):
        slider_steps = []
        for i, key in enumerate(self.results_dict.keys()):
            slider_step = {
                "args": [
                    [str(i)],
                    {
                        "frame": {"duration": 1000, "redraw": True},
                        "transition": {"duration": 300, "easing": "cubic-in-out"},
                    },
                ],
                "label": f"{key}",
                "method": "animate",
            }
            slider_steps.append(slider_step)

        return slider_steps

    def create_animation(self):
        animation_frames = self.create_animation_frames()
        slider_steps = self.create_slider_steps()
        for trace in animation_frames[0]['data']:
            self.fig.add_trace(trace)
        self.fig.frames = animation_frames
        self.fig.update_layout(
            sliders=[{
                "steps": slider_steps,
            }],
        )

        self.fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f'{self.Title}',
            xaxis=dict(visible=False, showticklabels=False),
            yaxis=dict(visible=False, showticklabels=False),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False,
            xaxis_ticks='',
            yaxis_ticks='',
            width=900,
            height=800,
        )

    def get_animation(self):
        self.create_animation()
        return self.fig
