import networkx as nx
import plotly.graph_objects as go

from prism.core.base import DecompositionResult, Subprocess


SUBPROCESS_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7DC6F",
    "#BB8FCE",
    "#85C1E9",
    "#F8B500",
    "#00CED1",
    "#FF69B4",
    "#32CD32",
    "#FFD700",
]


def get_subprocess_color(index: int) -> str:
    return SUBPROCESS_COLORS[index % len(SUBPROCESS_COLORS)]


class GraphVisualizer:
    """
    Interactive graph visualizer using Plotly.

    Supports:
    - Zoom in/out of the entire graph
    - Focus on individual subprocesses
    - Hierarchical view of decomposition
    - Color-coded subprocess boundaries
    """

    def __init__(self):
        self.layout_algorithm = "kamada_kawai"

    def _compute_layout(self, G: nx.DiGraph) -> dict[str, tuple[float, float]]:
        if len(G) == 0:
            return {}

        layout_funcs = {
            "spring": nx.spring_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
            "shell": nx.shell_layout,
            "spectral": nx.spectral_layout,
        }

        layout_func = layout_funcs.get(self.layout_algorithm, nx.kamada_kawai_layout)

        try:
            return layout_func(G)
        except Exception:
            # Fallback to spring layout if the "chosen one" (the kawai) fails
            return nx.spring_layout(G)

    def visualize_graph(
        self,
        G: nx.DiGraph,
        decomposition: DecompositionResult | None = None,
        title: str = "Process Graph",
        show_labels: bool = True,
        highlight_subprocess: str | None = None,
    ) -> go.Figure:
        """
        Create an interactive visualization of the graph.

        Args:
            G: NetworkX graph to visualize
            decomposition: Optional decomposition result for coloring
            title: Figure title
            show_labels: Whether to show node labels
            highlight_subprocess: Subprocess ID to highlight (dim others)

        Returns:
            Plotly Figure object
        """
        pos = self._compute_layout(G)

        node_colors = {}
        node_subprocess = {}
        if decomposition:
            for i, sp in enumerate(decomposition.subprocesses):
                color = get_subprocess_color(i)
                for node in sp.nodes:
                    node_colors[node] = color
                    node_subprocess[node] = sp

        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_color = "rgba(150, 150, 150, 0.5)"
            if decomposition:
                if edge[0] in node_colors and edge[1] in node_colors:
                    if node_colors[edge[0]] == node_colors[edge[1]]:
                        edge_color = node_colors[edge[0]]

            if highlight_subprocess:
                sp = (
                    decomposition.get_subprocess_by_id(highlight_subprocess)
                    if decomposition
                    else None
                )
                if sp and (edge[0] not in sp.nodes or edge[1] not in sp.nodes):
                    edge_color = "rgba(200, 200, 200, 0.2)"

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=2, color=edge_color),
                hoverinfo="none",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

        node_x = []
        node_y = []
        node_text = []
        node_color_list = []
        node_size_list = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            hover_text = f"<b>{node}</b>"
            if node in node_subprocess:
                hover_text += f"<br>Subprocess: {node_subprocess[node].name}"
            if G.in_degree(node) > 0 or G.out_degree(node) > 0:
                hover_text += f"<br>In-degree: {G.in_degree(node)}"
                hover_text += f"<br>Out-degree: {G.out_degree(node)}"
            node_text.append(hover_text)

            color = node_colors.get(node, "#888888")
            if highlight_subprocess:
                sp = (
                    decomposition.get_subprocess_by_id(highlight_subprocess)
                    if decomposition
                    else None
                )
                if sp and node not in sp.nodes:
                    color = "rgba(200, 200, 200, 0.3)"
            node_color_list.append(color)

            degree = G.in_degree(node) + G.out_degree(node)
            size = max(20, min(50, 15 + degree * 3))
            node_size_list.append(size)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if show_labels else "markers",
            hoverinfo="text",
            hovertext=node_text,
            text=list(G.nodes()) if show_labels else None,
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=node_size_list,
                color=node_color_list,
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            dragmode="zoom",
            margin=dict(l=20, r=20, t=60, b=20),
        )

        if decomposition:
            for i, sp in enumerate(decomposition.subprocesses):
                if sp.parent_id is None:  # Only top-level subprocesses
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=dict(size=15, color=get_subprocess_color(i)),
                            name=sp.name,
                            showlegend=True,
                        )
                    )

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.1,
                    y=1.1,
                    buttons=[
                        dict(
                            label="Reset Zoom",
                            method="relayout",
                            args=[{"xaxis.autorange": True, "yaxis.autorange": True}],
                        ),
                    ],
                )
            ]
        )

        return fig

    def visualize_subprocess(
        self,
        subprocess: Subprocess,
        parent_graph: nx.DiGraph | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """
        Visualize a single subprocess in detail.

        Args:
            subprocess: The subprocess to visualize
            parent_graph: Original graph (for edge weights, etc.)
            title: Custom title (defaults to subprocess name)

        Returns:
            Plotly Figure object
        """
        G = subprocess.to_networkx()

        if parent_graph:
            for u, v in G.edges():
                if parent_graph.has_edge(u, v):
                    G.edges[u, v].update(parent_graph.edges[u, v])

        return self.visualize_graph(
            G, title=title or f"Subprocess: {subprocess.name}", show_labels=True
        )

