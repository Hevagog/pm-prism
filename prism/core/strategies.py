"""
Convenience factory functions for creating decomposition strategies.

This module provides easy-to-use factory functions for common strategy configurations.
For full customization, instantiate strategy classes directly.
"""

from prism.core.decompositions import CommunityDetectionStrategy
from prism.core.decompositions import EmbeddingClusteringStrategy
from prism.core.labeler import LLMLabeler, SimpleLabeler


def embedding_strategy(
    *,
    use_llm_labeler: bool = False,
    optimal_size: tuple[int, int] = (6, 8),
    similarity_threshold: float = 0.3,
    model_name: str = "all-MiniLM-L6-v2",
) -> EmbeddingClusteringStrategy:
    """
    Create an embedding-based clustering strategy.

    Uses sentence embeddings + agglomerative clustering to group
    semantically similar activities.

    Args:
        use_llm_labeler: If True, use LLM for cluster naming (requires GROQ_API_KEY).
        optimal_size: Target cluster size range (min, max).
        similarity_threshold: Minimum similarity for merging (0-1).
        model_name: Sentence-transformers model name.

    Returns:
        Configured EmbeddingClusteringStrategy.

    Example:
        >>> strategy = embedding_strategy(use_llm_labeler=True)
        >>> decomposer = ProcessDecomposer(strategy)
    """
    labeler = LLMLabeler() if use_llm_labeler else SimpleLabeler()

    return EmbeddingClusteringStrategy(
        model_name=model_name,
        optimal_size=optimal_size,
        similarity_threshold=similarity_threshold,
        labeler=labeler,
    )


def community_strategy(
    *,
    use_llm_labeler: bool = False,
    resolution: float = 1.0,
) -> CommunityDetectionStrategy:
    """
    Create a community detection strategy.

    Uses Louvain algorithm to detect communities based on graph structure.

    Args:
        use_llm_labeler: If True, use LLM for cluster naming (requires GROQ_API_KEY).
        resolution: Louvain resolution parameter (higher = more communities).

    Returns:
        Configured CommunityDetectionStrategy.

    Example:
        >>> strategy = community_strategy(resolution=1.5)
        >>> decomposer = ProcessDecomposer(strategy)
    """
    labeler = LLMLabeler() if use_llm_labeler else SimpleLabeler()

    return CommunityDetectionStrategy(
        resolution=resolution,
        labeler=labeler,
    )
