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
    """
    Main class for decomposing process models.

    This orchestrates the entire decomposition pipeline:
    1. Load process model via adapter
    2. Apply decomposition strategy
    3. Label subprocesses
    4. Prepare for visualization

    Example usage:
        >>> decomposer = ProcessDecomposer()
        >>> result = decomposer.decompose_from_csv(
        ...     "event_log.csv",
        ...     case_id="Case ID",
        ...     activity_key="Activity",
        ...     timestamp_key="Timestamp"
        ... )
        >>> decomposer.visualize()
    """

    def __init__(
        self,
        strategy: str = "community",
        strategy_kwargs: dict | None = None,
    ):
        """
        Initialize the decomposer.

        Args:
            strategy: Decomposition strategy name. Options:
                - 'community': Louvain community detection
                - 'scc': Strongly connected components
                - 'cut_vertex': Articulation point based
                - 'hierarchical': Multi-level decomposition
                - 'gateway': Gateway/decision point based
            strategy_kwargs: Additional arguments for the strategy
        """
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

        if strategy == "hierarchical":
            primary = self._strategy_kwargs.pop("primary", "community")
            secondary = self._strategy_kwargs.pop("secondary", "community")
            return HierarchicalDecompositionStrategy(
                primary_strategy=self.strategies[primary.upper()].value(),
                secondary_strategy=self.strategies[secondary.upper()].value(),
                **self._strategy_kwargs,
            )

        return strategy_class(**self._strategy_kwargs)

    def decompose_from_csv(
        self,
        csv_path: str,
        case_id: str = "case:concept:name",
        activity_key: str = "concept:name",
        timestamp_key: str = "time:timestamp",
        **kwargs,
    ) -> DecompositionResult:
        """
        Load an event log from CSV and decompose the discovered DFG.

        Args:
            csv_path: Path to CSV file
            case_id: Column name for case ID
            activity_key: Column name for activity
            timestamp_key: Column name for timestamp
            **kwargs: Additional arguments for decomposition

        Returns:
            DecompositionResult with identified subprocesses
        """
        self._adapter = DFGAdapter()
        self._graph = self._adapter.load(
            csv_path,
            case_id=case_id,
            activity_key=activity_key,
            timestamp_key=timestamp_key,
        )
        return self._decompose(**kwargs)

    def decompose_from_xes(self, xes_path: str, **kwargs) -> DecompositionResult:
        """Load an event log from XES file and decompose."""
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
        """Load from pandas DataFrame and decompose."""
        self._adapter = DFGAdapter()
        self._graph = self._adapter.load(
            df, case_id=case_id, activity_key=activity_key, timestamp_key=timestamp_key
        )
        return self._decompose(**kwargs)

    def decompose_from_dfg(
        self, dfg: dict[tuple[str, str], int], **kwargs
    ) -> DecompositionResult:
        """Decompose a pre-computed DFG dict."""
        self._adapter = DFGAdapter()
        self._graph = self._adapter.load(dfg)
        return self._decompose(**kwargs)

    def decompose_graph(self, graph: nx.DiGraph, **kwargs) -> DecompositionResult:
        """Decompose an existing NetworkX graph."""
        self._graph = graph
        return self._decompose(**kwargs)

    def _decompose(self, **kwargs) -> DecompositionResult:
        """Run the decomposition pipeline."""
        if self._graph is None:
            raise ValueError("No graph loaded. Call one of the load methods first.")

        # Apply decomposition strategy
        subprocesses = self._strategy.decompose(self._graph, **kwargs)

        # Label subprocesses
        # context = kwargs.get("labeling_context", None)
        # for sp in subprocesses:
        #     sp.name = self._labeler.label(sp, context)

        # Build hierarchy
        hierarchy = {}
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
        """
        Visualize the decomposition result.

        Args:
            method: Visualization method
                - 'plotly': Interactive Plotly figure
                - 'pm4py': Use PM4Py's built-in visualization
            **kwargs: Additional arguments for visualization

        Returns:
            Visualization object (Figure)
        """
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


def quick_decompose(
    source: Any, strategy: str = "community", **kwargs
) -> ProcessDecomposer:
    """
    Quick helper function to decompose a process model.

    Args:
        source: Event log source (CSV path, DataFrame, XES path, or DFG dict)
        strategy: Decomposition strategy
        **kwargs: Additional arguments

    Returns:
        ProcessDecomposer instance with completed decomposition
    """
    decomposer = ProcessDecomposer(strategy=strategy)

    if isinstance(source, str):
        if source.endswith(".csv"):
            decomposer.decompose_from_csv(source, **kwargs)
        elif source.endswith(".xes"):
            decomposer.decompose_from_xes(source, **kwargs)
        else:
            raise ValueError(f"Unknown file format: {source}")
    elif isinstance(source, dict):
        decomposer.decompose_from_dfg(source, **kwargs)
    elif hasattr(source, "columns"):  # DataFrame
        decomposer.decompose_from_dataframe(source, **kwargs)
    else:
        raise TypeError(f"Unknown source type: {type(source)}")

    return decomposer
