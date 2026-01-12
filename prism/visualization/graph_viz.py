import math
from typing import Any, cast
import networkx as nx
import plotly.graph_objects as go

from prism.core.base import DecompositionResult, Subprocess, START_EVENT_ID, END_EVENT_ID


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
    """Interactive graph visualizer using Plotly."""

    def __init__(self):
        self.layout_algorithm = "spring"

    def compute_layout(self, G: nx.DiGraph) -> dict[str, tuple[float, float]]:
        if len(G) == 0:
            return {}

        def find_start_end() -> tuple[str | None, str | None]:
            start_node: str | None = None
            end_node: str | None = None
            for node_id, node_data in G.nodes(data=True):
                node_key = str(node_id)
                if node_data.get("is_start_event") or node_key == START_EVENT_ID:
                    start_node = node_key
                if node_data.get("is_end_event") or node_key == END_EVENT_ID:
                    end_node = node_key
            return start_node, end_node

        def enforce_start_end(
            pos: dict[str, tuple[float, float]],
            start_node: str | None,
            end_node: str | None,
        ) -> dict[str, tuple[float, float]]:
            if start_node is None and end_node is None:
                return pos

            xs_other: list[float] = []
            ys_other: list[float] = []
            for node_id, (x, y) in pos.items():
                if node_id in {start_node, end_node}:
                    continue
                xs_other.append(x)
                ys_other.append(y)

            if xs_other:
                min_x = min(xs_other)
                max_x = max(xs_other)
                span = max(max_x - min_x, 1e-6)
                pad = 0.35 * span
                mid_y = sum(ys_other) / len(ys_other)
            else:
                min_x, max_x, pad, mid_y = -1.0, 1.0, 1.0, 0.0

            if start_node is not None and start_node in pos:
                pos[start_node] = (min_x - pad, mid_y)
            if end_node is not None and end_node in pos:
                pos[end_node] = (max_x + pad, mid_y)

            return pos

        layout_funcs = {
            "spring": nx.spring_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
            "shell": nx.shell_layout,
            "spectral": nx.spectral_layout,
        }

        layout_func = layout_funcs.get(self.layout_algorithm, nx.kamada_kawai_layout)

        start_node, end_node = find_start_end()

        try:
            if self.layout_algorithm == "spring":
                # Two-stage layout for a less tangled result:
                # 1) an inexpensive global layout (kamada-kawai for smaller graphs)
                # 2) a spring relaxation step that starts from (1)
                # This stays dependency-free and works for all abstraction levels.
                H = G.to_undirected()
                n = len(H)

                if n <= 200:
                    pos = nx.kamada_kawai_layout(H)
                else:
                    k_init = 2.0 / math.sqrt(n) if n > 0 else None
                    pos = nx.spring_layout(H, k=k_init, weight=None, seed=42, iterations=50)

                # Pin Start/End to far left/right and align them vertically *before* relaxation,
                # so the spring simulation can settle the rest of the nodes around that.
                pos = cast(dict[str, tuple[float, float]], pos)
                pos = enforce_start_end(pos, start_node, end_node)

                fixed: list[str] = []
                if start_node is not None and start_node in pos:
                    fixed.append(start_node)
                if end_node is not None and end_node in pos:
                    fixed.append(end_node)

                k_relax = 2.2 / math.sqrt(n) if n > 0 else None
                if fixed:
                    pos = nx.spring_layout(
                        H,
                        pos=pos,
                        fixed=fixed,
                        k=k_relax,
                        weight=None,
                        seed=42,
                        iterations=250,
                    )
                else:
                    pos = nx.spring_layout(
                        H,
                        pos=pos,
                        k=k_relax,
                        weight=None,
                        seed=42,
                        iterations=250,
                    )

                pos = cast(dict[str, tuple[float, float]], pos)
                return enforce_start_end(pos, start_node, end_node)

            pos = layout_func(G)
            pos = cast(dict[str, tuple[float, float]], pos)
            return enforce_start_end(pos, start_node, end_node)
        except Exception:
            pos = cast(dict[str, tuple[float, float]], nx.spring_layout(G.to_undirected()))
            return enforce_start_end(pos, start_node, end_node)

    def _create_traces_and_annotations(
        self,
        G: nx.DiGraph,
        pos: dict,
        node_colors: dict | None = None,
        decomposition: DecompositionResult | None = None,
        highlight_subprocess: str | None = None,
        show_labels: bool = True,
    ) -> tuple[list[go.Scatter], list[dict]]:
        node_colors = node_colors or {}

        node_sizes = {}
        for node in G.nodes():
            if "size" in G.nodes[node]:
                s = G.nodes[node]["size"]
                node_sizes[node] = max(20, min(80, 20 + s * 2))
            else:
                degree = G.in_degree(node) + G.out_degree(node)
                node_sizes[node] = max(20, min(50, 15 + degree * 3))

        edge_traces = []
        annotations = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_color = "rgba(100, 100, 100, 0.8)"
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
                line=dict(width=3, color=edge_color),
                hoverinfo="none",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

            target_radius = node_sizes[edge[1]] / 2 + 2
            dx = x1 - x0
            dy = y1 - y0
            dist = math.sqrt(dx**2 + dy**2)

            if dist > 0:
                scale = 1 - (target_radius / dist)
                if scale < 0.1:
                    curr_x1, curr_y1 = x1, y1
                else:
                    curr_x1 = x0 + dx * scale
                    curr_y1 = y0 + dy * scale

                annotations.append(
                    dict(
                        ax=x0,
                        ay=y0,
                        axref="x",
                        ayref="y",
                        x=curr_x1,
                        y=curr_y1,
                        xref="x",
                        yref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=2,
                        arrowwidth=2,
                        arrowcolor=edge_color,
                        opacity=1 if edge_color.startswith("#") else 0.8,
                    )
                )

        node_x = []
        node_y = []
        node_text = []
        node_color_list = []
        node_size_list = []
        node_ids = []  # Store node IDs for click handling

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_ids.append(node)  # Track node ID

            if "label" in G.nodes[node]:
                hover_text = f"<b>{G.nodes[node]['label']}</b><br>ID: {node}<br>Size: {G.nodes[node]['size']}"
            else:
                hover_text = f"<b>{node}</b>"
                if G.in_degree(node) > 0 or G.out_degree(node) > 0:
                    hover_text += f"<br>In-degree: {G.in_degree(node)}"
                    hover_text += f"<br>Out-degree: {G.out_degree(node)}"

            node_text.append(hover_text)

            color = node_colors.get(node, "#888888") if node_colors else "#888888"

            node_color_list.append(color)
            node_size_list.append(node_sizes[node])

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if show_labels else "markers",
            hoverinfo="text",
            hovertext=node_text,
            text=[G.nodes[n].get("label", n) for n in G.nodes()]
            if show_labels
            else None,
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=node_size_list,
                color=node_color_list,
                line=dict(width=2, color="white"),
            ),
            customdata=node_ids,  # Add node IDs for click handling
            showlegend=False,
        )

        return edge_traces + [node_trace], annotations

    def visualize_graph(
        self,
        G: nx.DiGraph,
        decomposition: DecompositionResult | None = None,
        title: str = "Process Graph",
        show_labels: bool = True,
        highlight_subprocess: str | None = None,
    ) -> go.Figure:
        pos = self.compute_layout(G)

        node_colors = {}
        node_subprocess = {}
        if decomposition:
            for i, sp in enumerate(decomposition.subprocesses):
                color = get_subprocess_color(i)
                for node in sp.nodes:
                    node_colors[node] = color
                    node_subprocess[node] = sp

        traces, annotations = self._create_traces_and_annotations(
            G, pos, node_colors, decomposition, highlight_subprocess, show_labels
        )

        fig = go.Figure(data=traces)

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
            annotations=annotations,
        )

        if decomposition:
            for i, sp in enumerate(decomposition.subprocesses):
                if sp.parent_id is None:
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
                        )
                    ],
                )
            ]
        )

        return fig

    def visualize_hierarchy(
        self,
        graphs: list[nx.DiGraph],
        titles: list[str],
        precomputed_layouts: list[dict] | None = None,
        subprocess_colors: dict[str, str] | None = None,
        **kwargs,
    ) -> go.Figure:
        """
        Visualize a hierarchy of graphs with an interactive slider.

        Args:
            graphs: List of graphs at different abstraction levels.
            titles: Titles for each level.
            precomputed_layouts: Optional precomputed positions for stable layouts.
            subprocess_colors: Optional mapping of subprocess_id -> color for consistency.
        """

        all_traces = []
        all_annotations = []

        steps: list[dict[str, Any]] = []

        step_trace_indices = []
        current_trace_idx = 0

        for i, G in enumerate(graphs):
            if precomputed_layouts and i < len(precomputed_layouts):
                pos = precomputed_layouts[i]
            else:
                pos = self.compute_layout(G)

            node_colors = {}
            if subprocess_colors:
                for node in G.nodes():
                    if node in subprocess_colors:
                        node_colors[node] = subprocess_colors[node]
                    else:
                        node_colors[node] = "#888888"
            elif i > 0:  # Abstract levels without explicit colors
                for j, node in enumerate(G.nodes()):
                    node_colors[node] = get_subprocess_color(j)

            traces, annotations = self._create_traces_and_annotations(
                G, pos, node_colors=node_colors, show_labels=True
            )

            visible = i == 0

            num_traces = len(traces)
            step_indices = list(
                range(current_trace_idx, current_trace_idx + num_traces)
            )
            step_trace_indices.append(step_indices)
            current_trace_idx += num_traces

            for trace in traces:
                trace.visible = visible
                all_traces.append(trace)

            all_annotations.append(annotations)

            step: dict[str, Any] = dict(
                method="update",
                args=[
                    {
                        "visible": [False] * len(graphs) * 1000
                    },  # Placeholder, fixed below
                    {"title": titles[i], "annotations": annotations},
                ],
                label=f"Level {i}",
            )
            steps.append(step)

        total_traces = len(all_traces)
        for i, step in enumerate(steps):
            visible_array = [False] * total_traces
            for idx in step_trace_indices[i]:
                visible_array[idx] = True
            args = cast(list[dict[str, Any]], step["args"])
            args[0]["visible"] = visible_array

        fig = go.Figure(data=all_traces)

        fig.update_layout(
            title=dict(text=titles[0], x=0.5),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            dragmode="zoom",
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=all_annotations[0],
            sliders=[
                dict(
                    active=0,
                    currentvalue={"prefix": "Granularity: "},
                    pad={"t": 50},
                    steps=steps,
                )
            ],
        )

        return fig

    def visualize_subprocess(
        self,
        subprocess: Subprocess,
        parent_graph: nx.DiGraph | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Visualize a single subprocess in detail, preserving parent context if provided."""
        G = subprocess.to_networkx()

        if parent_graph:
            for u, v in G.edges():
                if parent_graph.has_edge(u, v):
                    G.edges[u, v].update(parent_graph.edges[u, v])

        return self.visualize_graph(
            G, title=title or f"Subprocess: {subprocess.name}", show_labels=True
        )

    def visualize_hierarchy_with_drilldown(
        self,
        graphs: list[nx.DiGraph],
        titles: list[str],
        subprocesses_per_level: list[list[Subprocess]],
        original_graph: nx.DiGraph,
        precomputed_layouts: list[dict] | None = None,
        **kwargs,
    ) -> go.Figure:
        """
        Visualize hierarchy with ability to drill down into clusters.

        Click on a cluster node to see its internal structure in a separate view.
        """
        fig = self.visualize_hierarchy(graphs, titles, precomputed_layouts, **kwargs)

        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br><i>Click to drill down</i><extra></extra>",
            selector=dict(mode="markers+text"),
        )

        return fig

    def create_drilldown_view(
        self,
        subprocess: Subprocess,
        original_graph: nx.DiGraph,
        color: str | None = None,
    ) -> go.Figure:
        """
        Create a detailed view of a subprocess's internal structure.

        Shows internal nodes and edges, plus "ghost" nodes for external connections
        with dotted arrows indicating where data flows in/out of the subprocess.

        Args:
            subprocess: The subprocess to visualize.
            original_graph: The original graph (to get edge weights etc.)
            color: Optional color for the nodes (uses subprocess color).

        Returns:
            A Plotly figure showing the subprocess internals with external connections.
        """
        subgraph: nx.DiGraph = original_graph.subgraph(subprocess.nodes).copy()

        incoming_external: dict[
            str, list[str]
        ] = {}  # internal_node -> [external_sources]
        outgoing_external: dict[
            str, list[str]
        ] = {}  # internal_node -> [external_targets]

        for node in subprocess.nodes:
            for predecessor in original_graph.predecessors(node):
                if predecessor not in subprocess.nodes:
                    if node not in incoming_external:
                        incoming_external[node] = []
                    incoming_external[node].append(predecessor)
            for successor in original_graph.successors(node):
                if successor not in subprocess.nodes:
                    if node not in outgoing_external:
                        outgoing_external[node] = []
                    outgoing_external[node].append(successor)

        pos = self.compute_layout(subgraph)

        if pos:
            x_vals = [p[0] for p in pos.values()]
            y_vals = [p[1] for p in pos.values()]
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            x_range = x_max - x_min if x_max > x_min else 1
            y_range = y_max - y_min if y_max > y_min else 1
            # In drilldown views we add ghost nodes left/right. If we base the horizontal
            # margin on the (often larger) y-range, a mostly-vertical internal layout can
            # get stretched horizontally and appear "empty". Keep horizontal padding tied
            # to x-range only.
            x_margin = max(x_range * 0.35, 0.6)
            y_step = max(y_range * 0.08, 0.15)
        else:
            x_min, x_max, y_min, y_max = 0, 1, 0, 1
            x_margin = 0.6
            y_step = 0.15

        ghost_positions: dict[str, tuple[float, float]] = {}
        incoming_ghosts = set()
        for internal_node, externals in incoming_external.items():
            for i, ext in enumerate(externals):
                ghost_id = f"⟵ {ext}"
                incoming_ghosts.add(ghost_id)
                if ghost_id not in ghost_positions:
                    int_x, int_y = pos[internal_node]
                    offset = (i - (len(externals) - 1) / 2) * y_step
                    ghost_positions[ghost_id] = (x_min - x_margin, int_y + offset)

        outgoing_ghosts = set()
        for internal_node, externals in outgoing_external.items():
            for i, ext in enumerate(externals):
                ghost_id = f"{ext} ⟶"
                outgoing_ghosts.add(ghost_id)
                if ghost_id not in ghost_positions:
                    int_x, int_y = pos[internal_node]
                    offset = (i - (len(externals) - 1) / 2) * y_step
                    ghost_positions[ghost_id] = (x_max + x_margin, int_y + offset)

        node_colors = {}
        if color:
            node_colors = {node: color for node in subgraph.nodes()}

        traces, annotations = self._create_traces_and_annotations(
            subgraph, pos, node_colors=node_colors, show_labels=True
        )

        if ghost_positions:
            ghost_x = []
            ghost_y = []
            ghost_text = []
            ghost_colors = []

            for ghost_id in ghost_positions:
                gx, gy = ghost_positions[ghost_id]
                ghost_x.append(gx)
                ghost_y.append(gy)

                if ghost_id in incoming_ghosts:
                    ghost_text.append(f"From: {ghost_id[2:]}")  # Remove "⟵ " prefix
                    ghost_colors.append("#9E9E9E")  # Gray for incoming
                else:
                    ghost_text.append(f"To: {ghost_id[:-2]}")  # Remove " ⟶" suffix
                    ghost_colors.append("#9E9E9E")  # Gray for outgoing

            ghost_trace = go.Scatter(
                x=ghost_x,
                y=ghost_y,
                mode="markers+text",
                marker=dict(
                    size=25,
                    color=ghost_colors,
                    symbol="diamond",
                    line=dict(width=2, color="white"),
                    opacity=0.7,
                ),
                text=[
                    g.replace("⟵ ", "← ").replace(" ⟶", " →")
                    for g in ghost_positions.keys()
                ],
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                hovertext=ghost_text,
                hoverinfo="text",
                showlegend=False,
            )
            traces.append(ghost_trace)

        external_edge_traces = []
        external_annotations = []

        for internal_node, externals in incoming_external.items():
            int_x, int_y = pos[internal_node]
            for ext in externals:
                ghost_id = f"⟵ {ext}"
                gx, gy = ghost_positions[ghost_id]

                external_edge_traces.append(
                    go.Scatter(
                        x=[gx, int_x],
                        y=[gy, int_y],
                        mode="lines",
                        line=dict(width=2, color="#9E9E9E", dash="dot"),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )
                external_annotations.append(
                    dict(
                        ax=gx,
                        ay=gy,
                        axref="x",
                        ayref="y",
                        x=int_x,
                        y=int_y,
                        xref="x",
                        yref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=1.5,
                        arrowcolor="#9E9E9E",
                        opacity=0.7,
                    )
                )

        for internal_node, externals in outgoing_external.items():
            int_x, int_y = pos[internal_node]
            for ext in externals:
                ghost_id = f"{ext} ⟶"
                gx, gy = ghost_positions[ghost_id]

                external_edge_traces.append(
                    go.Scatter(
                        x=[int_x, gx],
                        y=[int_y, gy],
                        mode="lines",
                        line=dict(width=2, color="#9E9E9E", dash="dot"),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )
                external_annotations.append(
                    dict(
                        ax=int_x,
                        ay=int_y,
                        axref="x",
                        ayref="y",
                        x=gx,
                        y=gy,
                        xref="x",
                        yref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=1.5,
                        arrowcolor="#9E9E9E",
                        opacity=0.7,
                    )
                )

        all_traces = external_edge_traces + traces
        all_annotations = annotations + external_annotations

        fig = go.Figure(data=all_traces)

        total_incoming = sum(len(v) for v in incoming_external.values())
        total_outgoing = sum(len(v) for v in outgoing_external.values())
        ext_info = ""
        if total_incoming or total_outgoing:
            ext_info = f" | ←{total_incoming} →{total_outgoing}"

        fig.update_layout(
            title=dict(
                text=f"🔍 {subprocess.name} ({len(subprocess.nodes)} nodes{ext_info})",
                x=0.5,
            ),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="white",
            dragmode="zoom",
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=all_annotations,
        )

        return fig
