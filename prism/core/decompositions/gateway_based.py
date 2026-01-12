import networkx as nx
import uuid

from prism.core.base import DecompositionStrategy, Subprocess


class GatewayBasedStrategy(DecompositionStrategy):
    """
    Decomposition based on gateway/decision points in the process.

    This strategy is particularly useful for BPMN models where
    gateways (XOR, AND, OR) define natural subprocess boundaries.

    For DFGs, we identify high-degree nodes as implicit gateways.
    """

    def __init__(self, degree_threshold: int = 3):
        """
        Args:
            degree_threshold: Minimum in+out degree to consider a node as gateway
        """
        self.degree_threshold = degree_threshold

    def decompose(self, graph: nx.DiGraph, **kwargs) -> list[Subprocess]:
        """Decompose based on gateway (high-degree) nodes."""
        # Identify gateway nodes
        gateways = set()
        for node in graph.nodes():
            total_degree = graph.in_degree(node) + graph.out_degree(node)
            if total_degree >= self.degree_threshold:
                gateways.add(node)

        if not gateways:
            return [
                Subprocess(
                    id=f"full_{uuid.uuid4().hex[:8]}",
                    name="Main_Process",
                    nodes=set(graph.nodes()),
                    edges=set(graph.edges()),
                    metadata={
                        "detection_method": "gateway",
                        "note": "no_gateways_found",
                    },
                )
            ]

        # Create a graph without gateway edges for component detection
        simplified = graph.copy()
        for gw in gateways:
            for pred in list(graph.predecessors(gw)):
                simplified.remove_edge(pred, gw)
            for succ in list(graph.successors(gw)):
                simplified.remove_edge(gw, succ)

        # Find weakly connected components in simplified graph
        components = list(nx.weakly_connected_components(simplified))

        subprocesses = []
        for i, component in enumerate(components):
            # Include relevant gateways in each component
            extended_component = component.copy()
            for gw in gateways:
                # Check if gateway connects to this component
                for pred in graph.predecessors(gw):
                    if pred in component:
                        extended_component.add(gw)
                        break
                for succ in graph.successors(gw):
                    if succ in component:
                        extended_component.add(gw)
                        break

            edges = set()
            for u, v in graph.edges():
                if u in extended_component and v in extended_component:
                    edges.add((u, v))

            subprocess = Subprocess(
                id=f"gw_{uuid.uuid4().hex[:8]}",
                name=f"Gateway_Block_{i + 1}",
                nodes=extended_component,
                edges=edges,
                metadata={
                    "detection_method": "gateway",
                    "gateways_included": list(gateways & extended_component),
                },
            )
            subprocesses.append(subprocess)

        return subprocesses

    def get_strategy_name(self) -> str:
        return "Gateway-Based Decomposition"
