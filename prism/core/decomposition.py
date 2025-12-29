import networkx as nx
from networkx.algorithms import community
import uuid
import heapq
import itertools

from prism.core import DecompositionStrategy, Subprocess, SubprocessLabeler


class CommunityDetectionStrategy(DecompositionStrategy):
    """Decomposition strategy using the Louvain community detection algorithm."""

    def __init__(
        self,
        resolution: float = 1.0,
        min_community_size: int = 2,
        labeler: SubprocessLabeler | None = None,
    ):
        """Initialize with resolution parameter and minimum community size."""
        self.resolution = resolution
        self.min_community_size = min_community_size
        self.labeler = labeler

    def _create_subprocesses_for_partition(
        self,
        graph: nx.DiGraph,
        partition: list[set[str]],
        min_size: int | None = None,
        labeling_context: dict | None = None,
    ) -> list[Subprocess]:
        """Create subprocess objects from a node partition."""
        subprocesses = []
        limit = min_size if min_size is not None else self.min_community_size

        for i, comm in enumerate(partition):
            if len(comm) >= limit:
                edges = set()
                for u, v in graph.edges():
                    if u in comm and v in comm:
                        edges.add((u, v))

                # Name: singleton uses node name, multi-node uses placeholder
                name = f"Subprocess_{i + 1}"
                if len(comm) == 1:
                    name = str(list(comm)[0])

                subprocess = Subprocess(
                    id=f"sp_{uuid.uuid4().hex[:8]}",
                    name=name,
                    nodes=comm,
                    edges=edges,
                    metadata={"detection_method": "louvain", "community_index": i},
                )
                subprocesses.append(subprocess)

        # Apply labeler for multi-node subprocesses
        if self.labeler:
            for sp in subprocesses:
                if len(sp.nodes) > 1:
                    sp.name = self.labeler.label(sp, labeling_context)

        return subprocesses

    def decompose(self, graph: nx.DiGraph, **kwargs) -> list[Subprocess]:
        """Decompose graph using community detection (Louvain)."""
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

        return self._create_subprocesses_for_partition(graph, communities)

        return self._create_subprocesses_for_partition(graph, communities)

    def _calculate_merge_weight(
        self, graph: nx.DiGraph, comm1: set[str], comm2: set[str]
    ) -> float:
        """Calculate total edge weight between two communities."""
        weight = 0.0
        for u in comm1:
            for v in graph[u]:
                if v in comm2:
                    weight += graph[u][v].get("weight", 1.0)

        for u in comm2:
            for v in graph[u]:
                if v in comm1:
                    weight += graph[u][v].get("weight", 1.0)
        return weight

    def _interpolate_partitions(
        self,
        graph: nx.DiGraph,
        start_partition: list[set[str]],
        end_partition: list[set[str]],
    ) -> list[list[set[str]]]:
        """Generate intermediate partitions by incrementally merging communities based on edge weights."""
        if not start_partition or not end_partition:
            return []

        # Map each node to its target community index in end_partition
        node_to_target = {}
        for i, comm in enumerate(end_partition):
            for node in comm:
                node_to_target[node] = i

        # Current state pool
        current_partition = [set(c) for c in start_partition]

        # Priority Queue for potential merges
        # Items: (-weight, comm1_id, comm2_id)
        comm_id_map = {id(c): c for c in current_partition}

        # Build initial merge candidates
        pq = []

        def push_merge_candidate(c1, c2):
            t1 = node_to_target.get(next(iter(c1))) if c1 else -1
            t2 = node_to_target.get(next(iter(c2))) if c2 else -2

            if t1 == t2 and t1 is not None:
                w = self._calculate_merge_weight(graph, c1, c2)
                # Maximize weight -> minimize -weight
                heapq.heappush(pq, (-w, id(c1), id(c2)))

        # Group by target index first
        comp_by_target = {}
        for c in current_partition:
            if not c:
                continue
            t = node_to_target.get(next(iter(c)))
            if t not in comp_by_target:
                comp_by_target[t] = []
            comp_by_target[t].append(c)

        for t, comms in comp_by_target.items():
            for c1, c2 in itertools.combinations(comms, 2):
                push_merge_candidate(c1, c2)

        intermediate_partitions = []

        # Loop until no more valid merges
        while pq:
            w, id1, id2 = heapq.heappop(pq)

            if id1 not in comm_id_map or id2 not in comm_id_map:
                continue  # Already merged

            c1 = comm_id_map[id1]
            c2 = comm_id_map[id2]

            # Merge
            new_comm = c1 | c2

            # Remove old
            del comm_id_map[id1]
            del comm_id_map[id2]
            current_partition.remove(c1)
            current_partition.remove(c2)

            # Add new
            comm_id_map[id(new_comm)] = new_comm
            current_partition.append(new_comm)

            # Find new merges for new_comm
            # Only need to look at others with same target
            t_new = node_to_target.get(next(iter(new_comm)))

            # Identify other active comms with same target
            # Optimization: could update comp_by_target map?
            # Or just scan current_partition (slower but safer)
            # Since N is small usually, scanning is fine.
            for other_c in current_partition:
                if other_c is new_comm:
                    continue
                t_other = node_to_target.get(next(iter(other_c)))
                if t_other == t_new:
                    push_merge_candidate(new_comm, other_c)

            # Save state (Deep copy partition structure)
            intermediate_partitions.append([set(c) for c in current_partition])

        return intermediate_partitions

    def decompose_hierarchical(
        self, graph: nx.DiGraph, **kwargs
    ) -> list[list[Subprocess]]:
        """Decompose graph into a hierarchy using Louvain partitions and interpolation."""
        if graph.number_of_nodes() == 0:
            return []

        undirected = graph.to_undirected()

        try:
            partitions_iter = list(
                community.louvain_partitions(
                    undirected,
                    resolution=kwargs.get("resolution", self.resolution),
                    seed=kwargs.get("seed", 42),
                )
            )

            # Ensure "All singletons" partition is the base (Level 0)
            singletons = [{n} for n in graph.nodes()]

            all_partitions = [singletons] + partitions_iter

            # Deduplicate partitions if logic produces same
            unique_partitions = []
            seen_sigs = set()
            for p in all_partitions:
                # Signature: sorted tuple of sorted tuples
                sig = tuple(sorted(tuple(sorted(list(c))) for c in p))
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    unique_partitions.append(p)

            full_hierarchy = []

            # Process levels with interpolation
            for i in range(len(unique_partitions) - 1):
                p_fine = unique_partitions[i]
                p_coarse = unique_partitions[i + 1]

                # Add fine level
                full_hierarchy.append(
                    self._create_subprocesses_for_partition(graph, p_fine, min_size=1)
                )

                # Interpolate
                substeps = self._interpolate_partitions(graph, p_fine, p_coarse)

                # Exclude the last element (which equals p_coarse) to avoid duplication.
                if substeps:
                    for step in substeps[:-1]:
                        full_hierarchy.append(
                            self._create_subprocesses_for_partition(
                                graph, step, min_size=1
                            )
                        )

            # Add final level
            if unique_partitions:
                full_hierarchy.append(
                    self._create_subprocesses_for_partition(
                        graph, unique_partitions[-1], min_size=1
                    )
                )

            return full_hierarchy

        except Exception as e:
            # Fallback
            print(f"Hierarchical decomposition failed: {e}")
            return [self.decompose(graph, **kwargs)]

    def get_strategy_name(self) -> str:
        return "Community Detection (Louvain)"
