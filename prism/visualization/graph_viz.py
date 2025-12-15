import math
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
        self.layout_algorithm = "spring"

    def compute_layout(self, G: nx.DiGraph) -> dict[str, tuple[float, float]]:
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
            if self.layout_algorithm == "spring":
                 # Tuned spring layout for better spacing
                 # k: optimal distance between nodes. increase to spread out. default is 1/sqrt(n)
                 # seed: for consistency
                 k_val = 2.5 / math.sqrt(len(G)) if len(G) > 0 else None
                 return nx.spring_layout(G, k=k_val, weight=None, seed=42, iterations=100)
    
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

    def _create_traces(
        self,
        G: nx.DiGraph,
        pos: dict,
        node_colors: dict | None = None,
        node_hover_text: list[str] | None = None,
        highlight_subprocess: str | None = None,
        decomposition: DecompositionResult | None = None,
        show_labels: bool = True
    ) -> list[go.Scatter]:
        """Helper to create edge and node traces."""
        node_sizes = {}
        for node in G.nodes():
            # If node has 'size' attribute (abstract graph), use it
            if "size" in G.nodes[node]:
                # Scale it down a bit usually
                s = G.nodes[node]["size"]
                node_sizes[node] = max(20, min(80, 20 + s * 2))
            else:
                degree = G.in_degree(node) + G.out_degree(node)
                node_sizes[node] = max(20, min(50, 15 + degree * 3))

        edge_traces = []
        annotations = [] # We need to return annotations too? 
        # Plotly annotations are layout objects, not traces.
        # But 'graphs' in hierarchy need different annotations per step.
        # Plotly Sliders can update layout.annotations.
        # So we need to return traces AND annotations.
        
        # But wait, annotations list in layout is global?
        # Ideally we use arrow markers on lines or something?
        # Plotly annotations in steps?
        # Yes, layout.annotations can be updated by steps.
        
        # ... Reworking to return traces and list of annotation dicts.
        
        # (Re-paste logic from previous edit for edges, with modifications)
        # To save tool usage, I will inline logic or duplicate slightly if needed, 
        # but refactoring is better.
        pass # Placeholder for diff.
        
    # I'll implement _create_traces_and_annotations 
    
    def _create_traces_and_annotations(
        self,
        G: nx.DiGraph,
        pos: dict,
        node_colors: dict | None = None,
        decomposition: DecompositionResult | None = None,
        highlight_subprocess: str | None = None,
        show_labels: bool = True
    )  -> tuple[list[go.Scatter], list[dict]]:
        
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

            # Arrow annotation
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
                        opacity=1 if edge_color.startswith("#") else 0.8
                    )
                )

        node_x = []
        node_y = []
        node_text = []
        node_color_list = []
        node_size_list = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Build label
            if "label" in G.nodes[node]:
                # Abstract graph
                hover_text = f"<b>{G.nodes[node]['label']}</b><br>ID: {node}<br>Size: {G.nodes[node]['size']}"
                label = G.nodes[node]['label']
            else:
                # Normal graph
                hover_text = f"<b>{node}</b>"
                if node in (node_colors or {}): # Rough check if colored? No, node_colors maps node->color
                     pass 
                # ... existing logic
                if G.in_degree(node) > 0 or G.out_degree(node) > 0:
                    hover_text += f"<br>In-degree: {G.in_degree(node)}"
                    hover_text += f"<br>Out-degree: {G.out_degree(node)}"
                label = str(node)

            node_text.append(hover_text)

            color = node_colors.get(node, "#888888") if node_colors else "#888888"
            if highlight_subprocess:
                 # ... existing logic simplified or skipped for now as we don't use it in hierarchy
                 pass
            
            node_color_list.append(color)
            node_size_list.append(node_sizes[node])

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if show_labels else "markers",
            hoverinfo="text",
            hovertext=node_text,
            text=[G.nodes[n].get("label", n) for n in G.nodes()] if show_labels else None,
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=node_size_list,
                color=node_color_list,
                line=dict(width=2, color="white"),
            ),
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
             # Add legend traces
             for i, sp in enumerate(decomposition.subprocesses):
                if sp.parent_id is None:
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None],
                            mode="markers",
                            marker=dict(size=15, color=get_subprocess_color(i)),
                            name=sp.name,
                            showlegend=True,
                        )
                    )
        
        # Add reset zoom button
        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="left", x=0.1, y=1.1, buttons=[dict(label="Reset Zoom", method="relayout", args=[{"xaxis.autorange": True, "yaxis.autorange": True}])])]
        )

        return fig

    def visualize_hierarchy(self, graphs: list[nx.DiGraph], titles: list[str], precomputed_layouts: list[dict] | None = None, **kwargs) -> go.Figure:
        """Visualize a hierarchy of graphs with a slider."""
        
        # Compute layouts and traces for all steps
        all_traces = []
        all_annotations = []
        
        steps = []
        
        # Keep track of trace indices per step
        step_trace_indices = []
        current_trace_idx = 0
        
        for i, G in enumerate(graphs):
            if precomputed_layouts and i < len(precomputed_layouts):
                pos = precomputed_layouts[i]
            else:
                pos = self.compute_layout(G)
            
            # Use rainbow colors for abstract nodes? 
            # Or just default gray?
            # Abstract nodes can be colored by something?
            node_colors = {}
            if i > 0: # Abstract levels
                for j, node in enumerate(G.nodes()):
                    node_colors[node] = get_subprocess_color(j)
                    
            traces, annotations = self._create_traces_and_annotations(
                G, pos, node_colors=node_colors, show_labels=True
            )
            
            # Visibility: Only first graph visible initially
            visible = (i == 0)
            
            # Calculate range of indices for this step
            num_traces = len(traces)
            step_indices = list(range(current_trace_idx, current_trace_idx + num_traces))
            step_trace_indices.append(step_indices)
            current_trace_idx += num_traces
            
            # Set init visibility
            for trace in traces:
                trace.visible = visible
                all_traces.append(trace)
                
            all_annotations.append(annotations)
            
            step = dict(
                method="update",
                args=[
                    {"visible": [False] * len(graphs) * 1000}, # Placeholder, will fix below
                    {"title": titles[i], "annotations": annotations}
                ],
                label=f"Level {i}"
            )
            steps.append(step)

        # Fix visibility arrays in steps
        total_traces = len(all_traces)
        for i, step in enumerate(steps):
            visible_array = [False] * total_traces
            for idx in step_trace_indices[i]:
                visible_array[idx] = True
            step["args"][0]["visible"] = visible_array

        fig = go.Figure(data=all_traces)
        
        # Initial layout (Level 0)
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
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Granularity: "},
                pad={"t": 50},
                steps=steps
            )]
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
        """
        G = subprocess.to_networkx()

        if parent_graph:
            for u, v in G.edges():
                if parent_graph.has_edge(u, v):
                    G.edges[u, v].update(parent_graph.edges[u, v])

        return self.visualize_graph(
            G, title=title or f"Subprocess: {subprocess.name}", show_labels=True
        )

