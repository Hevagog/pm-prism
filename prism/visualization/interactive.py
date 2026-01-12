from __future__ import annotations
from typing import TYPE_CHECKING

from dash import Dash, html, dcc, Output, Input, State, no_update, ctx
import plotly.graph_objects as go

from prism.core.base import DecompositionResult
from prism.visualization.graph_viz import GraphVisualizer, get_subprocess_color

if TYPE_CHECKING:
    from prism.core.decompositions import ProcessDecomposer


class InteractiveVisualizer:
    """
    Interactive visualization app using Dash.

    Allows clicking on clusters to drill down into their internal structure,
    with a working back button to return to the overview.
    """

    def __init__(self, decomposer: ProcessDecomposer):
        """
        Initialize the interactive visualizer.

        Args:
            decomposer: A ProcessDecomposer with decomposition already performed.
        """
        self.decomposer = decomposer
        self.viz = GraphVisualizer()
        self._app: Dash | None = None

        self._hierarchy_results: list[DecompositionResult] = []
        self._node_to_color: dict[str, str] = {}  # original node -> color
        self._subprocess_colors: dict[str, str] = {}  # subprocess id -> color
        self._current_level: int = 0

    def _compute_hierarchy_data(self) -> None:
        """Compute hierarchy and color mappings."""
        self._hierarchy_results = self.decomposer.decompose_hierarchical()

        if self._hierarchy_results:
            final_result = self._hierarchy_results[-1]
            for i, sp in enumerate(final_result.subprocesses):
                color = get_subprocess_color(i)
                self._subprocess_colors[sp.id] = color
                for node in sp.nodes:
                    self._node_to_color[node] = color

    def _get_subprocess_color(self, sp) -> str:
        """Get color for a subprocess based on its nodes."""
        color_counts: dict[str, int] = {}
        for node in sp.nodes:
            color = self._node_to_color.get(node, "#888888")
            color_counts[color] = color_counts.get(color, 0) + 1

        if color_counts:
            return max(color_counts, key=lambda c: color_counts[c])
        return "#888888"

    def _create_hierarchy_figure(self, level: int = 0) -> go.Figure:
        """Create the hierarchy figure for a given level."""
        if not self._hierarchy_results:
            return go.Figure()

        level = max(0, min(level, len(self._hierarchy_results) - 1))
        result = self._hierarchy_results[level]

        graph = self.decomposer.get_graph()
        if graph is None:
            return go.Figure()

        abstract_g = self.decomposer.generate_abstract_graph(result)

        base_pos = self.viz.compute_layout(graph)
        level_pos = {}

        for sp_id in abstract_g.nodes():
            sp = result.get_subprocess_by_id(sp_id)
            if sp:
                sum_x, sum_y = 0.0, 0.0
                n_count = 0
                for node in sp.nodes:
                    if node in base_pos:
                        x, y = base_pos[node]
                        sum_x += x
                        sum_y += y
                        n_count += 1
                if n_count > 0:
                    level_pos[sp_id] = (sum_x / n_count, sum_y / n_count)
                else:
                    level_pos[sp_id] = (0.0, 0.0)

        node_colors = {}
        for sp in result.subprocesses:
            node_colors[sp.id] = self._get_subprocess_color(sp)

        traces, annotations = self.viz._create_traces_and_annotations(
            abstract_g, level_pos, node_colors=node_colors, show_labels=True
        )

        fig = go.Figure(data=traces)

        count = len(result.subprocesses)
        fig.update_layout(
            title=dict(
                text=f"Level {level + 1}: {count} Communities (click a cluster to drill down)",
                x=0.5,
            ),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            dragmode="zoom",
            margin=dict(l=20, r=20, t=80, b=20),
            annotations=annotations,
            clickmode="event+select",
        )

        return fig

    def _get_subprocess_from_level(self, subprocess_id: str, level: int):
        """Get subprocess from a specific hierarchy level."""
        if not self._hierarchy_results or level >= len(self._hierarchy_results):
            return None
        return self._hierarchy_results[level].get_subprocess_by_id(subprocess_id)

    def _create_drilldown_figure(self, subprocess_id: str, level: int) -> go.Figure:
        """Create drill-down figure for a subprocess."""
        graph = self.decomposer.get_graph()
        if graph is None:
            return go.Figure()

        sp = self._get_subprocess_from_level(subprocess_id, level)
        if sp is None:
            return go.Figure()

        color = self._get_subprocess_color(sp)
        return self.viz.create_drilldown_view(sp, graph, color=color)

    def create_app(self) -> Dash:
        """Create and configure the Dash application."""
        self._compute_hierarchy_data()

        app = Dash(__name__)

        max_level = len(self._hierarchy_results) - 1 if self._hierarchy_results else 0

        app.layout = html.Div(
            [
                html.H1(
                    "Process Decomposition Explorer",
                    style={
                        "textAlign": "center",
                        "color": "#333",
                        "marginBottom": "10px",
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "← Back to Overview",
                            id="back-button",
                            n_clicks=0,
                            style={
                                "display": "none",
                                "padding": "10px 20px",
                                "fontSize": "14px",
                                "backgroundColor": "#4ECDC4",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                                "marginRight": "20px",
                            },
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Abstraction Level:",
                                    style={
                                        "marginRight": "15px",
                                        "whiteSpace": "nowrap",
                                    },
                                ),
                                html.Div(
                                    [
                                        dcc.Slider(
                                            id="level-slider",
                                            min=0,
                                            max=max_level,
                                            step=1,
                                            value=0,
                                            marks={
                                                i: f"L{i + 1}"
                                                for i in range(max_level + 1)
                                            },
                                        ),
                                    ],
                                    style={"width": "400px", "minWidth": "200px"},
                                ),
                            ],
                            id="slider-container",
                            style={"display": "flex", "alignItems": "center"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "padding": "10px 40px",
                        "justifyContent": "center",
                    },
                ),
                html.Div(
                    id="view-info",
                    style={
                        "textAlign": "center",
                        "color": "#666",
                        "marginBottom": "10px",
                    },
                ),
                dcc.Graph(
                    id="main-graph",
                    figure=self._create_hierarchy_figure(0),
                    style={"height": "80vh"},
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
                dcc.Store(
                    id="view-state",
                    data={"mode": "hierarchy", "level": 0, "subprocess_id": None},
                ),
            ]
        )

        @app.callback(
            [
                Output("main-graph", "figure"),
                Output("main-graph", "clickData"),
                Output("view-state", "data"),
                Output("back-button", "style"),
                Output("slider-container", "style"),
                Output("view-info", "children"),
            ],
            [
                Input("main-graph", "clickData"),
                Input("back-button", "n_clicks"),
                Input("level-slider", "value"),
            ],
            [State("view-state", "data")],
            prevent_initial_call=True,
        )
        def update_view(click_data, back_clicks, slider_value, view_state):
            triggered_id = ctx.triggered_id

            back_hidden = {
                "display": "none",
                "padding": "10px 20px",
                "fontSize": "14px",
                "backgroundColor": "#4ECDC4",
                "color": "white",
                "border": "none",
                "borderRadius": "5px",
                "cursor": "pointer",
                "marginRight": "20px",
            }
            back_visible = {**back_hidden, "display": "inline-block"}
            slider_visible = {"display": "flex", "alignItems": "center"}
            slider_hidden = {"display": "none"}

            if triggered_id == "back-button":
                level = view_state.get("level", 0)
                fig = self._create_hierarchy_figure(level)
                new_state = {"mode": "hierarchy", "level": level, "subprocess_id": None}
                return (
                    fig,
                    None,
                    new_state,
                    back_hidden,
                    slider_visible,
                    f"Showing hierarchy level {level + 1}",
                )

            if triggered_id == "level-slider":
                if view_state.get("mode") == "hierarchy":
                    fig = self._create_hierarchy_figure(slider_value)
                    new_state = {
                        "mode": "hierarchy",
                        "level": slider_value,
                        "subprocess_id": None,
                    }
                    return (
                        fig,
                        None,
                        new_state,
                        back_hidden,
                        slider_visible,
                        f"Showing hierarchy level {slider_value + 1}",
                    )
                else:
                    return (
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                    )

            if triggered_id == "main-graph" and click_data:
                if view_state.get("mode") != "hierarchy":
                    return (
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                    )

                point = click_data.get("points", [{}])[0]
                customdata = point.get("customdata")

                if customdata:
                    subprocess_id = customdata
                    current_level = view_state.get("level", 0)
                    sp = self._get_subprocess_from_level(subprocess_id, current_level)

                    if sp and len(sp.nodes) > 1:
                        fig = self._create_drilldown_figure(
                            subprocess_id, current_level
                        )
                        new_state = {
                            "mode": "drilldown",
                            "level": current_level,
                            "subprocess_id": subprocess_id,
                        }
                        info = f"🔍 Inside: {sp.name} ({len(sp.nodes)} nodes)"
                        return (
                            fig,
                            no_update,
                            new_state,
                            back_visible,
                            slider_hidden,
                            info,
                        )

            return no_update, no_update, no_update, no_update, no_update, no_update

        self._app = app
        return app

    def run(self, debug: bool = False, port: int = 8050) -> None:
        """
        Run the interactive visualization app.

        Args:
            debug: Enable Dash debug mode.
            port: Port to run the server on.
        """
        app = self.create_app()

        print(f"\n🚀 Starting interactive visualization at http://localhost:{port}")
        print("   Click on any cluster to see its internal structure.")
        print("   Use the slider to change abstraction level.")
        print("   Press Ctrl+C to stop.\n")

        app.run(debug=debug, port=port)


def run_interactive(decomposer: ProcessDecomposer, port: int = 8050) -> None:
    """
    Convenience function to run the interactive visualization.

    Args:
        decomposer: ProcessDecomposer with decomposition already performed.
        port: Port to run the server on.
    """
    viz = InteractiveVisualizer(decomposer)
    viz.run(port=port)
