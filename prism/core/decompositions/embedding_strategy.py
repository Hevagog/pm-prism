"""Embedding-based decomposition strategy using agglomerative clustering."""

import uuid
import heapq
from dataclasses import dataclass, field

import numpy as np
import networkx as nx

from prism.core import DecompositionStrategy, Subprocess, SubprocessLabeler


def _generate_cluster_id() -> int:
    """Generate a unique cluster ID."""
    return hash(uuid.uuid4())


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
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Vectors are already normalized, so dot product = cosine similarity
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
        # Linear penalty for small clusters
        return size / lo
    else:
        # Inverse penalty for large clusters
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
        nodes = list(graph.nodes())
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
        # Cosine similarity of aggregated embeddings
        cos_sim = cosine_similarity(cluster_a.embedding, cluster_b.embedding)

        if not self.use_graph_distance or graph_distances is None:
            return cos_sim

        # Average graph distance between nodes in the two clusters
        total_dist = 0
        count = 0
        for node_a in cluster_a.nodes:
            for node_b in cluster_b.nodes:
                key = (min(node_a, node_b), max(node_a, node_b))
                if key in graph_distances:
                    total_dist += graph_distances[key]
                    count += 1

        if count == 0:
            # No path exists - treat as maximum distance
            graph_sim = 0.0
        else:
            avg_dist = total_dist / count
            # Convert distance to similarity (closer = higher)
            graph_sim = 1.0 - (avg_dist / max_graph_dist) if max_graph_dist > 0 else 1.0

        # Weighted combination
        alpha = 1.0 - self.graph_distance_weight
        beta = self.graph_distance_weight
        return alpha * cos_sim + beta * graph_sim

    def _aggregate_embeddings(self, clusters: list[Cluster]) -> np.ndarray:
        """Aggregate embeddings by averaging (normalized)."""
        stacked = np.vstack([c.embedding for c in clusters])
        mean_emb = np.mean(stacked, axis=0)
        # Normalize
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        return mean_emb

    def _should_merge(self, size_before: list[int], size_after: int) -> bool:
        """
        Decide whether merging improves overall quality.
        Compares average quality before vs quality after merge.
        """
        quality_before = sum(cluster_size_quality(s, self.optimal_size) for s in size_before)
        quality_before /= len(size_before)

        quality_after = cluster_size_quality(size_after, self.optimal_size)

        # Merge if quality improves or stays similar
        return quality_after >= quality_before - 0.05

    def _are_neighbors(
        self, cluster_a: Cluster, cluster_b: Cluster, graph: nx.DiGraph
    ) -> bool:
        """Check if two clusters have any edge between them."""
        for node_a in cluster_a.nodes:
            for node_b in cluster_b.nodes:
                if graph.has_edge(node_a, node_b) or graph.has_edge(node_b, node_a):
                    return True
        return False

    def _run_clustering(
        self,
        graph: nx.DiGraph,
        initial_clusters: list[Cluster],
        graph_distances: dict[tuple[str, str], int] | None,
        record_history: bool = False,
    ) -> tuple[list[Cluster], list[list[Cluster]]]:
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
        # Build cluster dict
        clusters: dict[int, Cluster] = {c.id: c for c in initial_clusters}

        # Compute max graph distance for normalization
        max_graph_dist = max(graph_distances.values()) if graph_distances else 1

        # Priority queue: (-similarity, cluster_id_a, cluster_id_b)
        pq: list[tuple[float, int, int]] = []

        def add_merge_candidates(cluster: Cluster):
            """Add potential merges for a cluster with its neighbors."""
            for other in clusters.values():
                if other.id == cluster.id:
                    continue
                # Only consider neighbors in the graph
                if not self._are_neighbors(cluster, other, graph):
                    continue

                sim = self._hybrid_similarity(
                    cluster, other, graph_distances, max_graph_dist
                )
                if sim >= self.similarity_threshold:
                    heapq.heappush(pq, (-sim, cluster.id, other.id))

        # Initialize merge candidates
        for cluster in list(clusters.values()):
            add_merge_candidates(cluster)

        history: list[list[Cluster]] = []

        # Main clustering loop
        while pq:
            neg_sim, id_a, id_b = heapq.heappop(pq)

            # Check if clusters still exist (might have been merged)
            if id_a not in clusters or id_b not in clusters:
                continue

            cluster_a = clusters[id_a]
            cluster_b = clusters[id_b]

            # Check if merge improves quality
            merged_size = len(cluster_a.nodes) + len(cluster_b.nodes)
            sizes_before = [len(cluster_a.nodes), len(cluster_b.nodes)]

            if not self._should_merge(sizes_before, merged_size):
                continue

            # Perform merge
            new_nodes = cluster_a.nodes | cluster_b.nodes
            new_embedding = self._aggregate_embeddings([cluster_a, cluster_b])
            new_cluster = Cluster(nodes=new_nodes, embedding=new_embedding)

            # Remove old clusters
            del clusters[id_a]
            del clusters[id_b]

            # Add new cluster
            clusters[new_cluster.id] = new_cluster

            # Add new merge candidates
            add_merge_candidates(new_cluster)

            # Record state after this merge
            if record_history:
                history.append(list(clusters.values()))

        return list(clusters.values()), history

    def _clusters_to_subprocesses(
        self, graph: nx.DiGraph, clusters: list[Cluster], context: dict | None = None
    ) -> list[Subprocess]:
        """Convert clusters to Subprocess objects."""
        subprocesses = []

        for i, cluster in enumerate(clusters):
            # Collect internal edges
            edges = set()
            for u, v in graph.edges():
                if u in cluster.nodes and v in cluster.nodes:
                    edges.add((u, v))

            # Name: use single node name if singleton, otherwise placeholder
            if len(cluster.nodes) == 1:
                name = str(list(cluster.nodes)[0])
            else:
                name = f"Cluster_{i + 1}"  # Will be replaced by labeler if available

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

        # Apply labeler if available (for multi-node clusters)
        if self.labeler:
            for sp in subprocesses:
                if len(sp.nodes) > 1:
                    sp.name = self.labeler.label(sp, context)

        return subprocesses

    def decompose(self, graph: nx.DiGraph, **kwargs) -> list[Subprocess]:
        """Decompose graph using embedding-based clustering."""
        if graph.number_of_nodes() == 0:
            return []

        # Compute embeddings
        node_embeddings = self._compute_embeddings(graph)

        # Compute graph distances if needed
        graph_distances = None
        if self.use_graph_distance:
            graph_distances = self._compute_graph_distances(graph)

        # Initialize singleton clusters
        initial_clusters = [
            Cluster(nodes={node}, embedding=emb)
            for node, emb in node_embeddings.items()
        ]

        # Run clustering
        final_clusters, _ = self._run_clustering(
            graph, initial_clusters, graph_distances, record_history=False
        )

        # Get labeling context from kwargs
        labeling_context = kwargs.get("labeling_context")

        return self._clusters_to_subprocesses(graph, final_clusters, labeling_context)

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

        # Compute embeddings once
        node_embeddings = self._compute_embeddings(graph)

        # Compute graph distances once
        graph_distances = None
        if self.use_graph_distance:
            graph_distances = self._compute_graph_distances(graph)

        # Initial singleton clusters
        initial_clusters = [
            Cluster(nodes={node}, embedding=emb)
            for node, emb in node_embeddings.items()
        ]

        # Run clustering to get final clusters
        final_clusters, _ = self._run_clustering(
            graph, initial_clusters, graph_distances, record_history=False
        )

        # Separate: multi-node clusters vs singletons that didn't merge
        formed_clusters = [c for c in final_clusters if len(c.nodes) > 1]

        # Build hierarchy: show clusters forming one at a time
        hierarchy: list[list[Subprocess]] = []

        # Get labeling context from kwargs
        labeling_context = kwargs.get("labeling_context")

        # Level 0: all singletons
        hierarchy.append(self._clusters_to_subprocesses(graph, initial_clusters, labeling_context))

        # For each formed cluster, create a level showing it formed
        # while others remain as singletons
        nodes_in_formed = set()
        for i, cluster in enumerate(formed_clusters):
            nodes_in_formed.update(cluster.nodes)

            # Current state: formed clusters so far + remaining singletons
            current_state: list[Cluster] = []

            # Add all clusters formed so far
            current_state.extend(formed_clusters[: i + 1])

            # Add singletons for nodes not yet in any formed cluster
            for node, emb in node_embeddings.items():
                if node not in nodes_in_formed:
                    current_state.append(Cluster(nodes={node}, embedding=emb))

            hierarchy.append(self._clusters_to_subprocesses(graph, current_state, labeling_context))

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
        subgraph: nx.DiGraph = graph.subgraph(subprocess.nodes).copy()  # type: ignore
        return subgraph

    def get_strategy_name(self) -> str:
        return "Embedding Clustering (Agglomerative)"
