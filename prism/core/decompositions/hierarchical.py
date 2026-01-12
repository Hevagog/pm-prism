import networkx as nx

from prism.core import DecompositionStrategy, Subprocess
from prism.core.decompositions.louvain import CommunityDetectionStrategy


class HierarchicalDecompositionStrategy(DecompositionStrategy):
    """
    Hierarchical decomposition combining multiple strategies.

    First applies a coarse decomposition, then refines each subprocess
    if it's still too large.
    """

    def __init__(
        self,
        primary_strategy: DecompositionStrategy,
        secondary_strategy: DecompositionStrategy | None = None,
        max_subprocess_size: int = 10,
    ):
        self.primary = primary_strategy
        self.secondary = secondary_strategy or CommunityDetectionStrategy()
        self.max_size = max_subprocess_size

    def decompose(self, graph: nx.DiGraph, **kwargs) -> list[Subprocess]:
        """Apply hierarchical decomposition."""
        # First pass
        primary_result = self.primary.decompose(graph, **kwargs)

        final_subprocesses = []

        for sp in primary_result:
            if len(sp.nodes) > self.max_size:
                # Further decompose large subprocesses
                subgraph = sp.to_networkx()
                # Copy edge data from original
                for u, v, data in graph.edges(data=True):
                    if u in sp.nodes and v in sp.nodes:
                        subgraph.edges[u, v].update(data)

                children = self.secondary.decompose(subgraph, **kwargs)

                if len(children) > 1:
                    # Update parent-child relationships
                    sp.children_ids = [c.id for c in children]
                    for child in children:
                        child.parent_id = sp.id
                    final_subprocesses.append(sp)
                    final_subprocesses.extend(children)
                else:
                    final_subprocesses.append(sp)
            else:
                final_subprocesses.append(sp)

        return final_subprocesses

    def get_strategy_name(self) -> str:
        return f"Hierarchical ({self.primary.get_strategy_name()} + {self.secondary.get_strategy_name()})"
