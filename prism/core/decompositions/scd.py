import networkx as nx
import uuid

from prism.core.base import DecompositionStrategy, Subprocess


class SCCDecompositionStrategy(DecompositionStrategy):
    """
    Decomposition based on Strongly Connected Components.

    Useful for identifying cycles and strongly connected regions
    in process flows.
    """

    def __init__(self, min_scc_size: int = 2):
        self.min_scc_size = min_scc_size

    def decompose(self, graph: nx.DiGraph, **kwargs) -> list[Subprocess]:
        """Find strongly connected components as subprocesses."""
        sccs = list(nx.strongly_connected_components(graph))

        subprocesses = []
        for i, scc in enumerate(sccs):
            if len(scc) >= self.min_scc_size:
                edges = set()
                for u, v in graph.edges():
                    if u in scc and v in scc:
                        edges.add((u, v))

                subprocess = Subprocess(
                    id=f"scc_{uuid.uuid4().hex[:8]}",
                    name=f"SCC_Subprocess_{i + 1}",
                    nodes=scc,
                    edges=edges,
                    metadata={"detection_method": "scc", "scc_index": i},
                )
                subprocesses.append(subprocess)

        return subprocesses

    def get_strategy_name(self) -> str:
        return "Strongly Connected Components"
