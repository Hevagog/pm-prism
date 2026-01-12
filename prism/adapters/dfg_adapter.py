from __future__ import annotations
from typing import Any
from pathlib import Path
import sys
import networkx as nx
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog

from prism.core.base import (
    ProcessModelAdapter,
    DecompositionResult,
    START_EVENT_ID,
    END_EVENT_ID,
)


class DFGAdapter(ProcessModelAdapter):
    """Adapter for converting Directly-Follows Graphs (DFG) to NetworkX."""

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
        """Load a DFG from an event log (file, DataFrame, object) or dictionary."""
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

        # Be forgiving about accidental whitespace in column names.
        df.columns = [str(c).strip() for c in df.columns]

        case_id, activity_key, timestamp_key = self._resolve_columns(
            df,
            case_id=case_id,
            activity_key=activity_key,
            timestamp_key=timestamp_key,
        )

        if timestamp_key == "__pm_prism_generated_timestamp__":
            # Generate a stable per-case ordering timestamp when the log has none.
            # pm4py expects a timestamp column; this keeps the pipeline usable.
            if case_id not in df.columns:
                raise ValueError(
                    "Cannot generate timestamps without a valid case_id column."
                )
            base = pd.Timestamp("1970-01-01")
            df[timestamp_key] = base + pd.to_timedelta(
                df.groupby(case_id).cumcount(), unit="s"
            )

        df = pm4py.format_dataframe(
            df, case_id=case_id, activity_key=activity_key, timestamp_key=timestamp_key
        )

        return pm4py.convert_to_event_log(df)

    def _resolve_columns(
        self,
        df: pd.DataFrame,
        case_id: str,
        activity_key: str,
        timestamp_key: str,
    ) -> tuple[str, str, str]:
        columns = [str(c) for c in df.columns]

        resolved_case = self._resolve_one(
            columns,
            requested=case_id,
            role="case id",
            heuristics=["case", "caseid", "case id", "case:concept:name"],
            allow_generate=False,
        )
        resolved_activity = self._resolve_one(
            columns,
            requested=activity_key,
            role="activity",
            heuristics=["activity", "concept:name", "event", "task"],
            allow_generate=False,
        )
        resolved_time = self._resolve_one(
            columns,
            requested=timestamp_key,
            role="timestamp",
            heuristics=["timestamp", "time", "date"],
            allow_generate=True,
        )

        return resolved_case, resolved_activity, resolved_time

    def _resolve_one(
        self,
        columns: list[str],
        requested: str,
        role: str,
        heuristics: list[str],
        allow_generate: bool,
    ) -> str:
        if requested in columns:
            return requested

        def _norm(s: str) -> str:
            return "".join(ch.lower() for ch in s.strip() if ch.isalnum())

        norm_cols = {c: _norm(c) for c in columns}
        req_norm = _norm(requested)
        if req_norm and req_norm in norm_cols.values():
            for c, n in norm_cols.items():
                if n == req_norm:
                    return c

        hits: list[str] = []
        for c, n in norm_cols.items():
            if any(h in n for h in [_norm(h) for h in heuristics]):
                hits.append(c)

        # If there's exactly one plausible match, just use it (still simple + usable).
        if len(hits) == 1:
            print(
                f"Column '{requested}' not found; using '{hits[0]}' as {role}."
            )
            return hits[0]

        # Ambiguous (or none): prompt in TTY, otherwise fail with actionable message.
        if not sys.stdin.isatty():
            raise ValueError(
                f"{role} column '{requested}' not found. Available columns: {columns}. "
                f"Pass correct {role} column name(s) to decompose_from_csv()."
            )

        return self._prompt_for_column(columns, requested, role, hits, allow_generate)

    def _prompt_for_column(
        self,
        columns: list[str],
        requested: str,
        role: str,
        candidates: list[str],
        allow_generate: bool,
    ) -> str:
        print(f"\nCould not find {role} column: '{requested}'.")
        if candidates:
            print(f"Possible matches for {role}: {candidates}")

        print("Available columns:")
        for i, c in enumerate(columns, start=1):
            print(f"  {i}. {c}")

        if allow_generate and role == "timestamp":
            print("  0. (generate) Use per-case row order as timestamp")

        while True:
            raw = input(f"Select {role} column number: ").strip()
            if allow_generate and role == "timestamp" and raw == "0":
                return "__pm_prism_generated_timestamp__"

            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(columns):
                    return columns[idx - 1]

            # Allow typing the column name directly.
            if raw in columns:
                return raw

            print("Invalid choice. Enter a number from the list.")

    def _dfg_dict_to_networkx(
        self,
        dfg: dict[tuple[str, str], int],
        start_activities: dict[str, int] | None = None,
        end_activities: dict[str, int] | None = None,
    ) -> nx.DiGraph:
        G = nx.DiGraph()

        for (source, target), frequency in dfg.items():
            G.add_edge(source, target, weight=frequency, frequency=frequency)

        if start_activities is None:
            start_activities = {
                node: 1
                for node in G.nodes()
                if G.in_degree(node) == 0 and node not in (START_EVENT_ID, END_EVENT_ID)
            }

        if end_activities is None:
            end_activities = {
                node: 1
                for node in G.nodes()
                if G.out_degree(node) == 0
                and node not in (START_EVENT_ID, END_EVENT_ID)
            }

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

        G.add_node(
            START_EVENT_ID,
            label="Start",
            is_boundary=True,
            is_start_event=True,
        )
        G.add_node(
            END_EVENT_ID,
            label="End",
            is_boundary=True,
            is_end_event=True,
        )

        if start_activities:
            for activity, freq in start_activities.items():
                if activity == END_EVENT_ID:
                    continue
                G.add_edge(
                    START_EVENT_ID,
                    activity,
                    weight=freq,
                    frequency=freq,
                    is_boundary_edge=True,
                )

        if end_activities:
            for activity, freq in end_activities.items():
                if activity == START_EVENT_ID:
                    continue
                G.add_edge(
                    activity,
                    END_EVENT_ID,
                    weight=freq,
                    frequency=freq,
                    is_boundary_edge=True,
                )

        if G.number_of_nodes() == 2 and G.number_of_edges() == 0:
            G.add_edge(
                START_EVENT_ID,
                END_EVENT_ID,
                weight=1,
                frequency=1,
                is_boundary_edge=True,
            )

        return G

    def export(
        self, graph: nx.DiGraph, decomposition: DecompositionResult
    ) -> dict[str, Any]:
        """Export decomposed DFG mapping nodes to subprocesses."""
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

    def view_dfg(self) -> None:
        if self._dfg_data is None:
            raise ValueError("No DFG loaded. Call load() first.")
        pm4py.view_dfg(self._dfg_data, self._start_activities, self._end_activities)
