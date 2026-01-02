from typing import Any
import networkx as nx
import pm4py

from prism.core.base import ProcessModelAdapter, DecompositionResult


class BPMNAdapter(ProcessModelAdapter):
    """
    Adapter for BPMN process models.

    BPMN-specific considerations:
    - Gateways (XOR, AND, OR) define natural subprocess boundaries
    - Subprocesses in BPMN are explicit container elements
    - Events (start, end, intermediate) provide context
    - Pools and lanes may indicate organizational boundaries
    """

    def __init__(self):
        self._bpmn_model = None
        self._event_log = None

    def load(self, source: Any, **kwargs) -> nx.DiGraph:
        """
        Load a BPMN model and convert to NetworkX.

        For Part II, this will support:
        - PM4Py BPMN objects discovered from event logs
        - BPMN XML files
        - bpmn-python objects

        Args:
            source: BPMN model source

        Returns:
            NetworkX DiGraph representation
        """
        # Placeholder implementation for Part II
        # Discovery from event log
        if hasattr(source, "__class__") and "EventLog" in str(type(source)):
            self._bpmn_model = pm4py.discover_bpmn_inductive(source)
            self._event_log = source
        elif hasattr(source, "get_nodes"):
            self._bpmn_model = source
        else:
            raise NotImplementedError(
                "BPMN adapter loading will be fully implemented in Part II. "
                "Currently only supports PM4Py EventLog for discovery."
            )

        return self._bpmn_to_networkx()

    def _bpmn_to_networkx(self) -> nx.DiGraph:
        """
        Convert BPMN model to NetworkX graph.

        BPMN elements are represented as:
        - Nodes: Tasks, Events, Gateways
        - Edges: Sequence flows
        - Node attributes: type (task/gateway/event), gateway_type, etc.
        """
        G = nx.DiGraph()

        if self._bpmn_model is None:
            return G

        for node in self._bpmn_model.get_nodes():
            node_type = self._classify_bpmn_node(node)
            G.add_node(
                node.get_name() or node.get_id(),
                bpmn_id=node.get_id(),
                node_type=node_type,
                is_gateway=node_type == "gateway",
                is_event=node_type
                in ("start_event", "end_event", "intermediate_event"),
                is_task=node_type == "task",
            )

        for flow in self._bpmn_model.get_flows():
            source = flow.get_source()
            target = flow.get_target()
            source_name = source.get_name() or source.get_id()
            target_name = target.get_name() or target.get_id()
            G.add_edge(source_name, target_name, flow_id=flow.get_id())

        return G

    def _classify_bpmn_node(self, node) -> str:
        class_name = node.__class__.__name__.lower()

        if "gateway" in class_name:
            return "gateway"
        elif "start" in class_name:
            return "start_event"
        elif "end" in class_name:
            return "end_event"
        elif "event" in class_name:
            return "intermediate_event"
        elif "task" in class_name or "activity" in class_name:
            return "task"
        else:
            return "other"

    def export(
        self, graph: nx.DiGraph, decomposition: DecompositionResult
    ) -> dict[str, Any]:
        """
        Export decomposed BPMN for visualization.

        For Part II, this will generate:
        - BPMN XML with subprocess containers
        - Data for bpmn.js visualization
        """
        return {
            "bpmn_model": self._bpmn_model,
            "decomposition": decomposition,
            "subprocesses": [
                {
                    "id": sp.id,
                    "name": sp.name,
                    "nodes": list(sp.nodes),
                    "edges": list(sp.edges),
                }
                for sp in decomposition.subprocesses
            ],
        }

    def get_model_type(self) -> str:
        return "BPMN"

    def view_bpmn(self) -> None:
        """Visualize BPMN model using PM4Py."""
        if self._bpmn_model is None:
            raise ValueError("No BPMN model loaded")
        pm4py.view_bpmn(self._bpmn_model)
