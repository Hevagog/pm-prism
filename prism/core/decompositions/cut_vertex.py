import networkx as nx
import uuid

from prism.core.base import DecompositionStrategy, Subprocess


class CutVertexStrategy(DecompositionStrategy):
    """
    Decomposition based on articulation points (cut vertices).

    Identifies critical points in the process and splits around them.
    """

    def decompose(self, graph: nx.DiGraph, **kwargs) -> list[Subprocess]:
        """Split graph at articulation points."""
        # Convert to undirected for articulation point detection
        undirected = graph.to_undirected()

        try:
            articulation_points = set(nx.articulation_points(undirected))
        except nx.NetworkXError:
            # Graph might not be connected
            articulation_points = set()

        if not articulation_points:
            # No cut vertices, return whole graph as single subprocess
            return [
                Subprocess(
                    id=f"full_{uuid.uuid4().hex[:8]}",
                    name="Main_Process",
                    nodes=set(graph.nodes()),
                    edges=set(graph.edges()),
                    metadata={
                        "detection_method": "cut_vertex",
                        "note": "no_articulation_points",
                    },
                )
            ]

        # Find biconnected components
        biconnected = list(nx.biconnected_components(undirected))

        subprocesses = []
        for i, component in enumerate(biconnected):
            edges = set()
            for u, v in graph.edges():
                if u in component and v in component:
                    edges.add((u, v))

            subprocess = Subprocess(
                id=f"bcc_{uuid.uuid4().hex[:8]}",
                name=f"Block_{i + 1}",
                nodes=component,
                edges=edges,
                metadata={
                    "detection_method": "cut_vertex",
                    "block_index": i,
                    "articulation_points": list(articulation_points & component),
                },
            )
            subprocesses.append(subprocess)

        return subprocesses

    def get_strategy_name(self) -> str:
        return "Cut Vertex (Articulation Points)"
