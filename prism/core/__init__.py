"""Core module for process decomposition."""

from prism.core.base import (
    Subprocess,
    DecompositionResult,
    ProcessModelAdapter,
    DecompositionStrategy,
    SubprocessLabeler,
)
from prism.core.decomposer import ProcessDecomposer
from prism.core.decomposition import CommunityDetectionStrategy
from prism.core.embedding_strategy import (
    EmbeddingClusteringStrategy,
    EmbeddingProvider,
    cluster_size_quality,
)
from prism.core.labeler import LLMLabeler, SimpleLabeler
from prism.core.strategies import embedding_strategy, community_strategy

__all__ = [
    # Base classes
    "Subprocess",
    "DecompositionResult",
    "ProcessModelAdapter",
    "DecompositionStrategy",
    "SubprocessLabeler",
    # Main orchestrator
    "ProcessDecomposer",
    # Strategy classes (for full customization)
    "EmbeddingClusteringStrategy",
    "CommunityDetectionStrategy",
    # Factory functions (convenience)
    "embedding_strategy",
    "community_strategy",
    # Labelers
    "LLMLabeler",
    "SimpleLabeler",
    # Utilities
    "EmbeddingProvider",
    "cluster_size_quality",
]
