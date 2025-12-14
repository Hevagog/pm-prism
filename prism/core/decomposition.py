import networkx as nx
from networkx.algorithms import community
import uuid

from prism.core import DecompositionStrategy, Subprocess


class CommunityDetectionStrategy(DecompositionStrategy):
    """
    Decomposition based on community detection algorithms.

    Uses Louvain algorithm by default for detecting densely connected
    regions in the process graph.
    """

    def __init__(self, resolution: float = 1.0, min_community_size: int = 2):
        """
        Initialize the community detection strategy.

        Args:
            resolution: Resolution parameter for Louvain (higher = more communities)
            min_community_size: Minimum nodes required to form a subprocess
        """
        self.resolution = resolution
        self.min_community_size = min_community_size

    def decompose(self, graph: nx.DiGraph, **kwargs) -> list[Subprocess]:
        """
        Decompose graph using community detection.

        For directed graphs, we convert to undirected for community detection,
        then map back to preserve edge directions.
        """
        if graph.number_of_nodes() == 0:
            return []

        undirected = graph.to_undirected()

        try:
            communities = community.louvain_communities(
                undirected,
                resolution=kwargs.get("resolution", self.resolution),
                seed=kwargs.get("seed", 42),
            )
        except Exception:
            # Fallback to greedy modularity if Louvain fails
            communities = list(community.greedy_modularity_communities(undirected))

        subprocesses = []
        for i, comm in enumerate(communities):
            if len(comm) >= self.min_community_size:
                edges = set()
                for u, v in graph.edges():
                    if u in comm and v in comm:
                        edges.add((u, v))

                subprocess = Subprocess(
                    id=f"sp_{uuid.uuid4().hex[:8]}",
                    name=f"Subprocess_{i+1}",  # Placeholder, will be labeled later
                    nodes=comm,
                    edges=edges,
                    metadata={"detection_method": "louvain", "community_index": i},
                )
                subprocesses.append(subprocess)

        return subprocesses

    def get_strategy_name(self) -> str:
        return "Community Detection (Louvain)"


