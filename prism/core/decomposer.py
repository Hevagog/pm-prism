from typing import Any
import pandas as pd
from enum import Enum
import networkx as nx

from prism.core.base import (
    ProcessModelAdapter,
    DecompositionStrategy,
    DecompositionResult,
    Subprocess,
)
from prism.core.decomposition import CommunityDetectionStrategy
from prism.adapters import DFGAdapter
from prism.visualization import GraphVisualizer


class Strategies(Enum):
    COMMUNITY = CommunityDetectionStrategy


class ProcessDecomposer:
    """Orchestrates the loading and decomposition of process models."""

    def __init__(
        self,
        strategy: str = "community",
        strategy_kwargs: dict | None = None,
    ):
        """Initialize the decomposer with a specific strategy."""
        self.strategies = Strategies
        self._strategy_name = strategy
        self._strategy_kwargs = strategy_kwargs or {}
        self._strategy: DecompositionStrategy = self._create_strategy(strategy)

        self._adapter: ProcessModelAdapter | None = None
        self._graph: nx.DiGraph | None = None
        self._result: DecompositionResult | None = None

    def _create_strategy(self, strategy: str) -> DecompositionStrategy:
        """Create a decomposition strategy instance."""
        strat_viable_names = [strat.name.lower() for strat in self.strategies]
        if strategy not in strat_viable_names:
            raise ValueError(
                f"Unknown strategy: {strategy}. Available: {strat_viable_names}"
            )

        strategy_class = self.strategies[strategy.upper()].value
        return strategy_class(**self._strategy_kwargs)

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

        # Label subprocesses
        # context = kwargs.get("labeling_context", None)
        # for sp in subprocesses:
        #     sp.name = self._labeler.label(sp, context)

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
        if self._result is None:
            raise ValueError("No decomposition result. Call decompose_* first.")
        match method:
            case "plotly":
                viz = GraphVisualizer(**kwargs.get("visualizer_kwargs", {}))
                return viz.visualize_graph(self._graph, self._result, **kwargs)
            case "pm4py":
                if self._adapter and hasattr(self._adapter, "view_dfg"):
                    self._adapter.view_dfg()
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

        viz = GraphVisualizer(**kwargs.get("visualizer_kwargs", {}))

        # 1. Compute Base Layout (Original Graph)
        # Uses the visualizer's algorithm (tuned spring)
        base_pos = viz.compute_layout(self._graph)

        graphs = []
        titles = []
        layouts = []  # Store stable layouts

        # Level 0 (Singletons) is naturally the first result from decomposition.

        # Add Abstract Graphs
        for i, res in enumerate(results):
            abstract_g = self.generate_abstract_graph(res)
            # Count subprocesses
            count = len(res.subprocesses)
            graphs.append(abstract_g)
            titles.append(f"Level {i + 1}: {count} Communities")

            # Compute stable layout for this level based on base_pos
            level_pos = {}
            for sp_id in abstract_g.nodes():
                sp = res.get_subprocess_by_id(sp_id)
                if sp:
                    # Centroid calculation
                    # Average x, y of all constituent nodes in original graph
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
                        # Fallback if no nodes found (shouldn't happen)
                        level_pos[sp_id] = (0.0, 0.0)

            layouts.append(level_pos)

        return viz.visualize_hierarchy(
            graphs, titles, precomputed_layouts=layouts, **kwargs
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

    def get_graph(self) -> nx.DiGraph | None:
        return self._graph

    def get_result(self) -> DecompositionResult | None:
        return self._result

    def export(self) -> dict[str, Any]:
        if self._adapter and self._result:
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
