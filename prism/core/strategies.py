"""
Convenience factory functions for creating decomposition strategies.

This module provides easy-to-use factory functions for common strategy configurations.
For full customization, instantiate strategy classes directly.
"""

from prism.core.decompositions import EmbeddingClusteringStrategy
from prism.core.labeler import LLMLabeler, SimpleLabeler


def embedding_strategy(
    *,
    use_llm_labeler: bool = True,
    optimal_size: tuple[int, int] = (6, 8),
    similarity_threshold: float = 0.3,
    model_name: str = "all-mpnet-base-v2",
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
