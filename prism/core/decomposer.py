"""Process decomposer orchestrating loading, decomposition, and visualization."""

from typing import Any
import pandas as pd
import networkx as nx

from prism.core.base import (
    ProcessModelAdapter,
    DecompositionStrategy,
    DecompositionResult,
    Subprocess,
)
from prism.adapters import DFGAdapter
from prism.visualization import GraphVisualizer


class ProcessDecomposer:
    """
    Orchestrates the loading and decomposition of process models.

    Usage:
        from prism.core import EmbeddingClusteringStrategy, LLMLabeler

        # Create strategy with desired configuration
        labeler = LLMLabeler()
        strategy = EmbeddingClusteringStrategy(labeler=labeler, optimal_size=(5, 8))

        # Initialize decomposer with strategy
        decomposer = ProcessDecomposer(strategy)
        result = decomposer.decompose_from_csv("log.csv")
    """

    def __init__(self, strategy: DecompositionStrategy):
        """
        Initialize the decomposer with a decomposition strategy.

        Args:
            strategy: A configured DecompositionStrategy instance.
        """
        if not isinstance(strategy, DecompositionStrategy):
            raise TypeError(
                f"strategy must be a DecompositionStrategy instance, got {type(strategy).__name__}. "
                "Use EmbeddingClusteringStrategy() or CommunityDetectionStrategy()."
            )

        self._strategy = strategy
        self._adapter: ProcessModelAdapter | None = None
        self._graph: nx.DiGraph | None = None
        self._result: DecompositionResult | None = None

    def decompose_from_csv(
        self,
        csv_path: str,
        case_id: str = "case:concept:name",
        activity_key: str = "concept:name",
        timestamp_key: str = "time:timestamp",
        **kwargs,
    ) -> DecompositionResult:
        """Load an event log from a CSV file and decompose."""
        self._adapter = DFGAdapter()
        self._graph = self._adapter.load(
            csv_path,
            case_id=case_id,
            activity_key=activity_key,
            timestamp_key=timestamp_key,
        )
        return self._decompose(**kwargs)

    def decompose_from_xes(self, xes_path: str, **kwargs) -> DecompositionResult:
        """Load an event log from an XES file and decompose."""
        self._adapter = DFGAdapter()
        self._graph = self._adapter.load(xes_path)
        return self._decompose(**kwargs)

    def decompose_from_dataframe(
        self,
        df: pd.DataFrame,
        case_id: str = "case:concept:name",
        activity_key: str = "concept:name",
        timestamp_key: str = "time:timestamp",
        **kwargs,
    ) -> DecompositionResult:
        """Load from a pandas DataFrame and decompose."""
        self._adapter = DFGAdapter()
        self._graph = self._adapter.load(
            df, case_id=case_id, activity_key=activity_key, timestamp_key=timestamp_key
        )
        return self._decompose(**kwargs)

    def decompose_from_dfg(
        self, dfg: dict[tuple[str, str], int], **kwargs
    ) -> DecompositionResult:
        """Decompose a pre-computed DFG dictionary."""
        self._adapter = DFGAdapter()
        self._graph = self._adapter.load(dfg)
        return self._decompose(**kwargs)

    def decompose_graph(self, graph: nx.DiGraph, **kwargs) -> DecompositionResult:
        """Decompose an existing NetworkX graph."""
        self._graph = graph
        return self._decompose(**kwargs)

    def decompose_hierarchical(self, **kwargs) -> list[DecompositionResult]:
        """Decompose the graph into valid hierarchical levels."""
        if self._graph is None:
            raise ValueError("No graph loaded. Call one of the load methods first.")

        # Get hierarchy from strategy
        # Expects list[list[Subprocess]]
        levels_subprocesses = self._strategy.decompose_hierarchical(
            self._graph, **kwargs
        )

        results: list[DecompositionResult] = []
        for stage_subprocesses in levels_subprocesses:
            # Build hierarchy map for this level
            hierarchy: dict[str, list[str]] = {}
            for sp in stage_subprocesses:
                if sp.parent_id:
                    if sp.parent_id not in hierarchy:
                        hierarchy[sp.parent_id] = []
                    hierarchy[sp.parent_id].append(sp.id)

            result = DecompositionResult(
                original_graph=self._graph,
                subprocesses=stage_subprocesses,
                hierarchy=hierarchy,
                metadata={
                    "strategy": self._strategy.get_strategy_name(),
                    "level_index": len(results),
                    "subprocess_count": len(stage_subprocesses),
                },
            )
            results.append(result)

        return results

    def generate_abstract_graph(self, result: DecompositionResult) -> nx.DiGraph:
        """Generate an abstract graph where nodes represent subprocesses."""
        G_abstract = nx.DiGraph()

        # Map original node -> Subprocess ID
        node_to_sp: dict[str, str] = {}
        for sp in result.subprocesses:
            for node in sp.nodes:
                node_to_sp[node] = sp.id

            # Add node to abstract graph with metadata
            G_abstract.add_node(
                sp.id,
                label=sp.name,
                size=len(sp.nodes),  # Simple metric for size
            )

        # Add edges
        # We iterate over original edges and lift them
        for u, v in result.original_graph.edges():
            if u in node_to_sp and v in node_to_sp:
                sp_u = node_to_sp[u]
                sp_v = node_to_sp[v]

                if sp_u != sp_v:
                    if G_abstract.has_edge(sp_u, sp_v):
                        G_abstract[sp_u][sp_v]["weight"] += 1
                    else:
                        G_abstract.add_edge(sp_u, sp_v, weight=1)

        return G_abstract

    def _decompose(self, **kwargs) -> DecompositionResult:
        """Execute the decomposition pipeline."""
        if self._graph is None:
            raise ValueError("No graph loaded. Call one of the load methods first.")

        # Apply decomposition strategy
        subprocesses = self._strategy.decompose(self._graph, **kwargs)

        # Build hierarchy
        hierarchy: dict[str, list[str]] = {}
        for sp in subprocesses:
            if sp.parent_id:
                if sp.parent_id not in hierarchy:
                    hierarchy[sp.parent_id] = []
                hierarchy[sp.parent_id].append(sp.id)

        self._result = DecompositionResult(
            original_graph=self._graph,
            subprocesses=subprocesses,
            hierarchy=hierarchy,
            metadata={
                "strategy": self._strategy.get_strategy_name(),
                "adapter": (
                    self._adapter.get_model_type() if self._adapter else "direct"
                ),
                "node_count": self._graph.number_of_nodes(),
                "edge_count": self._graph.number_of_edges(),
                "subprocess_count": len(subprocesses),
            },
        )

        return self._result

    def visualize(self, method: str = "plotly", **kwargs) -> Any:
        """Visualize the decomposition result."""
        if self._result is None or self._graph is None:
            raise ValueError("No decomposition result. Call decompose_* first.")
        match method:
            case "plotly":
                viz = GraphVisualizer(**kwargs.get("visualizer_kwargs", {}))
                return viz.visualize_graph(self._graph, self._result, **kwargs)
            case "pm4py":
                if self._adapter and hasattr(self._adapter, "view_dfg"):
                    self._adapter.view_dfg()  # type: ignore[attr-defined]
                else:
                    raise ValueError("PM4Py visualization requires DFG adapter")
            case _:
                raise ValueError(f"Unknown visualization method: {method}")

    def visualize_hierarchical(
        self, results: list[DecompositionResult], method: str = "plotly", **kwargs
    ) -> Any:
        """Visualize a hierarchy of process decompositions."""
        if method != "plotly":
            raise ValueError("Hierarchical visualization only supported for Plotly.")
        if self._graph is None:
            raise ValueError("No graph loaded. Call decompose_* first.")

        viz = GraphVisualizer(**kwargs.get("visualizer_kwargs", {}))

        # 1. Compute Base Layout (Original Graph)
        base_pos = viz.compute_layout(self._graph)

        graphs = []
        titles = []
        layouts = []

        # 2. Build consistent color mapping based on original nodes
        # Each original node gets a color based on which final cluster it belongs to
        # Then subprocesses inherit colors from the nodes they contain
        from prism.visualization.graph_viz import get_subprocess_color
        
        # Get the final level (most abstracted) to determine cluster colors
        final_result = results[-1] if results else None
        node_to_color: dict[str, str] = {}
        subprocess_colors: dict[str, str] = {}
        
        if final_result:
            for i, sp in enumerate(final_result.subprocesses):
                color = get_subprocess_color(i)
                subprocess_colors[sp.id] = color
                for node in sp.nodes:
                    node_to_color[node] = color

        # 3. Build graphs for each level with consistent colors
        for level_idx, res in enumerate(results):
            abstract_g = self.generate_abstract_graph(res)
            count = len(res.subprocesses)
            graphs.append(abstract_g)
            titles.append(f"Level {level_idx + 1}: {count} Communities")

            # Compute stable layout based on centroids
            level_pos = {}
            for sp_id in abstract_g.nodes():
                sp = res.get_subprocess_by_id(sp_id)
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
                    
                    # Assign color based on the dominant final cluster
                    # (the color of the majority of nodes in this subprocess)
                    if node_to_color:
                        color_counts: dict[str, int] = {}
                        for node in sp.nodes:
                            c = node_to_color.get(node, "#888888")
                            color_counts[c] = color_counts.get(c, 0) + 1
                        # Pick the most common color
                        dominant_color = max(color_counts, key=lambda c: color_counts[c])
                        subprocess_colors[sp_id] = dominant_color

            layouts.append(level_pos)

        return viz.visualize_hierarchy(
            graphs, titles, 
            precomputed_layouts=layouts, 
            subprocess_colors=subprocess_colors,
            **kwargs
        )

    def get_subprocess(self, subprocess_id: str) -> Subprocess | None:
        """Get a subprocess by ID."""
        if self._result is None:
            return None
        return self._result.get_subprocess_by_id(subprocess_id)

    def get_subprocesses(self) -> list[Subprocess]:
        """Get all subprocesses."""
        if self._result is None:
            return []
        return self._result.subprocesses

    def visualize_subprocess(self, subprocess_id: str, color: str | None = None, **kwargs) -> Any:
        """
        Visualize the internal structure of a subprocess (drill-down view).
        
        Args:
            subprocess_id: ID of the subprocess to visualize.
            color: Optional color for the nodes. If None, uses index-based color.
            **kwargs: Additional arguments passed to visualizer.
            
        Returns:
            Plotly figure showing the subprocess internals.
        """
        if self._graph is None:
            raise ValueError("No graph loaded.")
        if self._result is None:
            raise ValueError("No decomposition result.")
        
        sp = self.get_subprocess(subprocess_id)
        if sp is None:
            raise ValueError(f"Subprocess '{subprocess_id}' not found.")
        
        # Determine color from subprocess index if not provided
        if color is None:
            from prism.visualization.graph_viz import get_subprocess_color
            sp_index = next(
                (i for i, s in enumerate(self._result.subprocesses) if s.id == subprocess_id),
                0
            )
            color = get_subprocess_color(sp_index)
        
        viz = GraphVisualizer(**kwargs.get("visualizer_kwargs", {}))
        return viz.create_drilldown_view(sp, self._graph, color=color)

    def get_graph(self) -> nx.DiGraph | None:
        return self._graph

    def get_result(self) -> DecompositionResult | None:
        return self._result

    def export(self) -> dict[str, Any]:
        if self._adapter and self._result and self._graph:
            return self._adapter.export(self._graph, self._result)
        return {}

    def summary(self) -> str:
        if self._result is None:
            return "No decomposition performed yet."

        lines = [
            "=== Decomposition Summary ===",
            f"Strategy: {self._result.metadata.get('strategy', 'unknown')}",
            f"Total nodes: {self._result.metadata.get('node_count', 0)}",
            f"Total edges: {self._result.metadata.get('edge_count', 0)}",
            f"Subprocesses found: {len(self._result.subprocesses)}",
            "",
            "=== Subprocesses ===",
        ]

        for sp in self._result.subprocesses:
            level = "  " if sp.parent_id else ""
            lines.append(
                f"{level}• {sp.name} ({len(sp.nodes)} nodes, {len(sp.edges)} edges)"
            )

        return "\n".join(lines)
