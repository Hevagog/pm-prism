"""Embedding-based decomposition strategy using agglomerative clustering."""

import uuid
import heapq
from dataclasses import dataclass, field

import numpy as np
import networkx as nx

from sentence_transformers import SentenceTransformer

from prism.core.base import (
    DecompositionStrategy,
    Subprocess,
    SubprocessLabeler,
    is_boundary_node,
)
from prism.core.labeler import LLMLabeler


def _generate_cluster_id() -> int:
    """Generate a unique cluster ID."""
    return int(uuid.uuid4().hex, 16)


@dataclass
class Cluster:
    """Represents a cluster of nodes with an aggregated embedding."""

    nodes: set[str]
    embedding: np.ndarray
    id: int = field(default_factory=_generate_cluster_id)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Cluster) and self.id == other.id


class EmbeddingProvider:
    """Provides embeddings for activity names using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence-transformers model."""
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b))


def cluster_size_quality(size: int, optimal_range: tuple[int, int] = (6, 8)) -> float:
    """
    Score cluster quality based on size.
    Returns value in [0, 1], higher is better.
    Favors clusters in the optimal range.
    """
    lo, hi = optimal_range
    if lo <= size <= hi:
        return 1.0
    elif size < lo:
        return size / lo
    else:
        return hi / size


class EmbeddingClusteringStrategy(DecompositionStrategy):
    """
    Decomposition strategy using semantic embeddings and agglomerative clustering.

    Nodes are embedded using a language model, then merged bottom-up based on
    cosine similarity. Merging decisions consider a size-based quality function
    to favor clusters of optimal size.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        optimal_size: tuple[int, int] = (6, 8),
        similarity_threshold: float = 0.3,
        use_graph_distance: bool = True,
        graph_distance_weight: float = 0.15,
        labeler: SubprocessLabeler | None = None,
    ):
        """
        Initialize the embedding clustering strategy.

        Args:
            model_name: Name of the sentence-transformers model to use.
            optimal_size: Tuple (min, max) for ideal cluster size.
            similarity_threshold: Minimum similarity to consider merging.
            use_graph_distance: Whether to factor in graph topology.
            graph_distance_weight: Weight for graph distance in hybrid similarity.
            labeler: Optional labeler for generating cluster names.
        """
        self.model_name = model_name
        self.optimal_size = optimal_size
        self.similarity_threshold = similarity_threshold
        self.use_graph_distance = use_graph_distance
        self.graph_distance_weight = graph_distance_weight
        self.labeler = labeler

        self._embedder: EmbeddingProvider | None = None

    def _get_embedder(self) -> EmbeddingProvider:
        """Lazy initialization of the embedding provider."""
        if self._embedder is None:
            self._embedder = EmbeddingProvider(self.model_name)
        return self._embedder

    def _compute_embeddings(self, graph: nx.DiGraph) -> dict[str, np.ndarray]:
        """Compute embeddings for all nodes in the graph."""
        nodes = sorted(
            node_id
            for node_id, node_data in graph.nodes(data=True)
            if not is_boundary_node(str(node_id), node_data)
        )
        embedder = self._get_embedder()
        embeddings_array = embedder.embed(nodes)
        return {node: embeddings_array[i] for i, node in enumerate(nodes)}

    def _compute_graph_distances(self, graph: nx.DiGraph) -> dict[tuple[str, str], int]:
        """Compute shortest path distances between all node pairs."""
        undirected = graph.to_undirected()
        distances = {}

        for node in graph.nodes():
            lengths = nx.single_source_shortest_path_length(undirected, node)
            for other, dist in lengths.items():
                if node != other:
                    key = (min(node, other), max(node, other))
                    distances[key] = dist

        return distances

    def _hybrid_similarity(
        self,
        cluster_a: Cluster,
        cluster_b: Cluster,
        graph_distances: dict[tuple[str, str], int] | None,
        max_graph_dist: int,
    ) -> float:
        """
        Compute similarity between two clusters.
        Combines cosine similarity of embeddings with graph proximity.
        """
        cos_sim = cosine_similarity(cluster_a.embedding, cluster_b.embedding)

        if not self.use_graph_distance or graph_distances is None:
            return cos_sim

        total_distance = 0.0
        pair_count = 0

        worst_distance = max(max_graph_dist, 1)
        for left_node in cluster_a.nodes:
            for right_node in cluster_b.nodes:
                key = (min(left_node, right_node), max(left_node, right_node))
                total_distance += float(graph_distances.get(key, worst_distance))
                pair_count += 1

        if pair_count == 0:
            graph_similarity = 0.0
        else:
            average_distance = total_distance / pair_count
            graph_similarity = 1.0 - (average_distance / worst_distance)

        semantic_weight = 1.0 - self.graph_distance_weight
        structural_weight = self.graph_distance_weight
        return semantic_weight * cos_sim + structural_weight * graph_similarity

    def _aggregate_embeddings(self, clusters: list[Cluster]) -> np.ndarray:
        """Aggregate embeddings by averaging (normalized)."""
        stacked = np.vstack([c.embedding for c in clusters])
        mean_emb = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        return mean_emb

    def _should_merge(self, size_before: list[int], size_after: int) -> bool:
        """
        Decide whether merging improves overall quality.
        Compares average quality before vs quality after merge.
        """
        quality_sum_before = 0.0
        for cluster_size in size_before:
            quality_sum_before += cluster_size_quality(cluster_size, self.optimal_size)
        quality_before = quality_sum_before / len(size_before)

        quality_after = cluster_size_quality(size_after, self.optimal_size)

        return quality_after >= quality_before - 0.05

    def _are_neighbors(
        self, cluster_a: Cluster, cluster_b: Cluster, graph: nx.DiGraph
    ) -> bool:
        """Check if two clusters have any edge between them.

        This is implemented by scanning adjacency lists rather than checking all
        node-pairs (which is much slower).
        """
        other_nodes = cluster_b.nodes
        for candidate_node in cluster_a.nodes:
            for successor in graph.successors(candidate_node):
                if successor in other_nodes:
                    return True
            for predecessor in graph.predecessors(candidate_node):
                if predecessor in other_nodes:
                    return True
        return False

    def _run_clustering(
        self,
        graph: nx.DiGraph,
        initial_clusters: list[Cluster],
        graph_distances: dict[tuple[str, str], int] | None,
        record_history: bool = False,
    ) -> tuple[list[Cluster], list[list[Cluster]], list[Cluster]]:
        """
        Run one round of agglomerative clustering.

        Args:
            graph: Original graph (for neighbor relationships).
            initial_clusters: Starting clusters to merge.
            graph_distances: Precomputed distances between original nodes.
            record_history: If True, record state after each merge.

        Returns:
            Tuple of (final clusters, history of states after each merge).
        """
        clusters: dict[int, Cluster] = {
            cluster.id: cluster for cluster in initial_clusters
        }

        max_graph_dist = max(graph_distances.values()) if graph_distances else 1

        priority_queue: list[tuple[float, str, str, int, int]] = []

        def _cluster_tie_breaker_key(cluster: Cluster) -> str:
            return min(cluster.nodes) if cluster.nodes else ""

        def add_merge_candidates(base_cluster: Cluster) -> None:
            """Add potential merges for a cluster with its neighbors."""
            base_key = _cluster_tie_breaker_key(base_cluster)
            for other_cluster in clusters.values():
                if other_cluster.id == base_cluster.id:
                    continue
                if not self._are_neighbors(base_cluster, other_cluster, graph):
                    continue

                sim = self._hybrid_similarity(
                    base_cluster, other_cluster, graph_distances, max_graph_dist
                )
                if sim >= self.similarity_threshold:
                    other_key = _cluster_tie_breaker_key(other_cluster)
                    heapq.heappush(
                        priority_queue,
                        (-sim, base_key, other_key, base_cluster.id, other_cluster.id),
                    )

        for starting_cluster in list(clusters.values()):
            add_merge_candidates(starting_cluster)

        history: list[list[Cluster]] = []
        merge_sequence: list[Cluster] = []

        while priority_queue:
            neg_similarity, _, _, left_cluster_id, right_cluster_id = heapq.heappop(
                priority_queue
            )

            if left_cluster_id not in clusters or right_cluster_id not in clusters:
                continue

            left_cluster = clusters[left_cluster_id]
            right_cluster = clusters[right_cluster_id]

            merged_size = len(left_cluster.nodes) + len(right_cluster.nodes)
            sizes_before = [len(left_cluster.nodes), len(right_cluster.nodes)]

            if not self._should_merge(sizes_before, merged_size):
                continue

            merged_nodes = left_cluster.nodes | right_cluster.nodes
            merged_embedding = self._aggregate_embeddings([left_cluster, right_cluster])
            merged_cluster = Cluster(nodes=merged_nodes, embedding=merged_embedding)

            del clusters[left_cluster_id]
            del clusters[right_cluster_id]

            clusters[merged_cluster.id] = merged_cluster

            add_merge_candidates(merged_cluster)

            if record_history:
                history.append(list(clusters.values()))
                merge_sequence.append(merged_cluster)

        return list(clusters.values()), history, merge_sequence

    def _clusters_to_subprocesses(
        self, graph: nx.DiGraph, clusters: list[Cluster], context: dict | None = None
    ) -> list[Subprocess]:
        """Convert clusters to Subprocess objects."""
        subprocesses: list[Subprocess] = []

        for i, cluster in enumerate(clusters):
            edges = set()
            for u, v in graph.edges():
                if u in cluster.nodes and v in cluster.nodes:
                    edges.add((u, v))

            if len(cluster.nodes) == 1:
                node_id = next(iter(cluster.nodes))
                name = str(graph.nodes[node_id].get("label", node_id))
            else:
                name = f"Group ({len(cluster.nodes)} activities)"

            subprocess = Subprocess(
                id=f"sp_{uuid.uuid4().hex[:8]}",
                name=name,
                nodes=cluster.nodes,
                edges=edges,
                metadata={
                    "detection_method": "embedding_clustering",
                    "cluster_index": i,
                    "size": len(cluster.nodes),
                },
            )
            subprocesses.append(subprocess)

        labeler = self.labeler
        if labeler is None:
            labeler = LLMLabeler()

        if isinstance(labeler, LLMLabeler):
            labels = labeler.label_batch(subprocesses, context)
            for subprocess in subprocesses:
                if len(subprocess.nodes) > 1:
                    subprocess.name = labels[subprocess.id]
        else:
            for subprocess in subprocesses:
                if len(subprocess.nodes) > 1:
                    subprocess.name = labeler.label(subprocess, context)

        return subprocesses

    def decompose(self, graph: nx.DiGraph, **kwargs) -> list[Subprocess]:
        """Decompose graph using embedding-based clustering."""
        if graph.number_of_nodes() == 0:
            return []

        boundary_nodes = {
            node_id
            for node_id, node_data in graph.nodes(data=True)
            if is_boundary_node(str(node_id), node_data)
        }

        node_embeddings = self._compute_embeddings(graph)

        graph_distances = None
        if self.use_graph_distance:
            graph_distances = self._compute_graph_distances(graph)

        initial_clusters = [
            Cluster(nodes={node}, embedding=emb)
            for node, emb in node_embeddings.items()
        ]

        final_clusters, _, _ = self._run_clustering(
            graph, initial_clusters, graph_distances, record_history=False
        )

        labeling_context = kwargs.get("labeling_context")

        subprocesses = self._clusters_to_subprocesses(graph, final_clusters, labeling_context)

        for node_id in sorted(boundary_nodes):
            if node_id not in graph:
                continue
            subprocesses.append(
                Subprocess(
                    id=f"sp_{uuid.uuid4().hex[:8]}",
                    name=str(graph.nodes[node_id].get("label", node_id)),
                    nodes={node_id},
                    edges=set(),
                    metadata={"detection_method": "boundary", "is_boundary": True},
                )
            )

        return subprocesses

    def decompose_hierarchical(
        self, graph: nx.DiGraph, max_levels: int = 10, **kwargs
    ) -> list[list[Subprocess]]:
        """
        Decompose graph into a hierarchy of abstraction levels.

        Shows clusters forming one at a time. Each level adds one complete cluster
        (of optimal size ~6-8 nodes) to the visualization.

        Args:
            graph: The process graph to decompose.
            max_levels: Maximum hierarchy depth (safety limit).

        Returns:
            List of levels, from finest (singletons) to coarsest.
            Each level shows one more cluster fully formed.
        """
        if graph.number_of_nodes() == 0:
            return []

        boundary_nodes = {
            node_id
            for node_id, node_data in graph.nodes(data=True)
            if is_boundary_node(str(node_id), node_data)
        }

        node_embeddings = self._compute_embeddings(graph)

        graph_distances = None
        if self.use_graph_distance:
            graph_distances = self._compute_graph_distances(graph)

        initial_clusters = [
            Cluster(nodes={node}, embedding=emb)
            for node, emb in node_embeddings.items()
        ]

        final_clusters, _, merge_sequence = self._run_clustering(
            graph, initial_clusters, graph_distances, record_history=True
        )

        final_multi_node_clusters = [
            cluster for cluster in final_clusters if len(cluster.nodes) > 1
        ]

        formation_step_by_nodeset: dict[frozenset[str], int] = {}
        for step_index, merged_cluster in enumerate(merge_sequence):
            key = frozenset(merged_cluster.nodes)
            formation_step_by_nodeset.setdefault(key, step_index)

        final_multi_node_clusters.sort(
            key=lambda cluster: formation_step_by_nodeset.get(
                frozenset(cluster.nodes), 10**9
            )
        )

        hierarchy: list[list[Subprocess]] = []

        labeling_context = kwargs.get("labeling_context")

        hierarchy.append(
            self._clusters_to_subprocesses(graph, initial_clusters, labeling_context)
        )

        if boundary_nodes:
            boundary_subprocesses = [
                Subprocess(
                    id=f"sp_{uuid.uuid4().hex[:8]}",
                    name=str(graph.nodes[node_id].get("label", node_id)),
                    nodes={node_id},
                    edges=set(),
                    metadata={"detection_method": "boundary", "is_boundary": True},
                )
                for node_id in sorted(boundary_nodes)
                if node_id in graph
            ]
            hierarchy[0].extend(boundary_subprocesses)

        if max_levels <= 1:
            return hierarchy

        max_clusters_to_reveal = max_levels - 1

        revealed_nodes: set[str] = set()
        for cluster_index, final_cluster in enumerate(final_multi_node_clusters):
            if cluster_index >= max_clusters_to_reveal:
                break

            revealed_nodes.update(final_cluster.nodes)

            current_clusters: list[Cluster] = []
            current_clusters.extend(final_multi_node_clusters[: cluster_index + 1])

            for node_name, node_embedding in node_embeddings.items():
                if node_name not in revealed_nodes:
                    current_clusters.append(
                        Cluster(nodes={node_name}, embedding=node_embedding)
                    )

            hierarchy.append(
                self._clusters_to_subprocesses(
                    graph, current_clusters, labeling_context
                )
            )

            if boundary_nodes:
                hierarchy[-1].extend(
                    [
                        Subprocess(
                            id=f"sp_{uuid.uuid4().hex[:8]}",
                            name=str(graph.nodes[node_id].get("label", node_id)),
                            nodes={node_id},
                            edges=set(),
                            metadata={
                                "detection_method": "boundary",
                                "is_boundary": True,
                            },
                        )
                        for node_id in sorted(boundary_nodes)
                        if node_id in graph
                    ]
                )

        return hierarchy

    def get_subprocess_internal_graph(
        self, graph: nx.DiGraph, subprocess: Subprocess
    ) -> nx.DiGraph:
        """
        Extract the internal subgraph of a subprocess.

        Useful for "drilling down" into a cluster to see its internal structure.

        Args:
            graph: The original process graph.
            subprocess: The subprocess to extract.

        Returns:
            A subgraph containing only the nodes and edges within the subprocess.
        """
        return nx.DiGraph(graph.subgraph(subprocess.nodes).copy())

    def get_strategy_name(self) -> str:
        return "Embedding Clustering (Agglomerative)"
