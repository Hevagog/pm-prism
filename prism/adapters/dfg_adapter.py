from __future__ import annotations
from typing import Any
from pathlib import Path
import networkx as nx
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog

from prism.core.base import ProcessModelAdapter, DecompositionResult


class DFGAdapter(ProcessModelAdapter):
    """
    Adapter for Directly-Follows Graphs.

    Supports loading from:
    - Event logs (PM4Py EventLog objects)
    - CSV files
    - XES files
    - Pre-computed DFG dictionaries
    """

    def __init__(self):
        self._dfg_data: dict | None = None
        self._start_activities: dict | None = None
        self._end_activities: dict | None = None
        self._event_log: Any | None = None

    def load(
        self,
        source: str | Path | pd.DataFrame | "EventLog" | dict,
        case_id: str = "case:concept:name",
        activity_key: str = "concept:name",
        timestamp_key: str = "time:timestamp",
    ) -> nx.DiGraph:
        """
        Load a DFG from various sources.

        Args:
            source: Can be:
                - Path to CSV/XES file
                - pandas DataFrame with event log data
                - PM4Py EventLog object
                - Dict in format {(activity1, activity2): frequency}
            case_id: Column name for case identifier
            activity_key: Column name for activity name
            timestamp_key: Column name for timestamp

        Returns:
            NetworkX DiGraph with activities as nodes and frequencies as edge weights
        """
        if isinstance(source, dict):
            return self._dfg_dict_to_networkx(source)

        event_log = self._load_event_log(source, case_id, activity_key, timestamp_key)
        self._event_log = event_log

        dfg, start_acts, end_acts = pm4py.discover_dfg(event_log)

        self._dfg_data = dfg
        self._start_activities = start_acts
        self._end_activities = end_acts

        return self._dfg_dict_to_networkx(dfg, start_acts, end_acts)

    def _load_event_log(
        self,
        source: str | Path | pd.DataFrame | EventLog,
        case_id: str,
        activity_key: str,
        timestamp_key: str,
    ) -> EventLog:
        if hasattr(source, "__class__") and "EventLog" in str(type(source)):
            return source

        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() == ".xes":
                return pm4py.read_xes(str(path))
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        df = pm4py.format_dataframe(
            df, case_id=case_id, activity_key=activity_key, timestamp_key=timestamp_key
        )

        return pm4py.convert_to_event_log(df)

    def _dfg_dict_to_networkx(
        self,
        dfg: dict[tuple[str, str], int],
        start_activities: dict[str, int] | None = None,
        end_activities: dict[str, int] | None = None,
    ) -> nx.DiGraph:
        G = nx.DiGraph()

        for (source, target), frequency in dfg.items():
            G.add_edge(source, target, weight=frequency, frequency=frequency)

        if start_activities:
            for activity, freq in start_activities.items():
                if activity not in G:
                    G.add_node(activity)
                G.nodes[activity]["is_start"] = True
                G.nodes[activity]["start_frequency"] = freq

        if end_activities:
            for activity, freq in end_activities.items():
                if activity not in G:
                    G.add_node(activity)
                G.nodes[activity]["is_end"] = True
                G.nodes[activity]["end_frequency"] = freq

        return G

    def export(
        self, graph: nx.DiGraph, decomposition: DecompositionResult
    ) -> dict[str, Any]:
        """
        Export decomposed DFG for visualization.

        Returns a dict containing:
        - Original DFG data
        - Subprocess information
        - Mapping of nodes to subprocesses
        """
        node_to_subprocess: dict[str, list[str]] = {}
        for sp in decomposition.subprocesses:
            for node in sp.nodes:
                if node not in node_to_subprocess:
                    node_to_subprocess[node] = []
                node_to_subprocess[node].append(sp.id)

        return {
            "dfg": self._dfg_data,
            "start_activities": self._start_activities,
            "end_activities": self._end_activities,
            "subprocesses": [
                {
                    "id": sp.id,
                    "name": sp.name,
                    "nodes": list(sp.nodes),
                    "edges": list(sp.edges),
                    "metadata": sp.metadata,
                    "parent_id": sp.parent_id,
                    "children_ids": sp.children_ids,
                }
                for sp in decomposition.subprocesses
            ],
            "node_mapping": node_to_subprocess,
            "hierarchy": decomposition.hierarchy,
        }

    def get_model_type(self) -> str:
        return "DFG"

    def get_event_log(self) -> "EventLog":
        return self._event_log

    def view_dfg(self) -> None:
        if self._dfg_data is None:
            raise ValueError("No DFG loaded. Call load() first.")
        pm4py.view_dfg(self._dfg_data, self._start_activities, self._end_activities)

    def get_performance_dfg(self) -> tuple[dict, dict, dict]:
        if self._event_log is None:
            raise ValueError("No event log loaded. Call load() with event log first.")
        return pm4py.discover_performance_dfg(self._event_log)

