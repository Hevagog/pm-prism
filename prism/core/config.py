from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StrategyType(Enum):
    LOUVAIN = "louvain"
    CUT_VERTEX = "cut_vertex"
    GATEWAY = "gateway"
    SCC = "scc"
    HIERARCHICAL = "hierarchical"
    EMBEDDING = "embedding"


@dataclass
class DecompositionConfig:
    """Configuration for decomposition strategies."""

    strategy_type: StrategyType = StrategyType.LOUVAIN

    # Common parameters
    min_size: int = 2

    # Louvain specific
    resolution: float = 1.0
    seed: int = 42

    # Gateway specific
    degree_threshold: int = 3

    # Hierarchical specific
    max_subprocess_size: int = 10
    # For hierarchical, we need to know which strategies to use as primary/secondary
    # If None, defaults are used (usually Louvain)
    primary_type: Optional[StrategyType] = None
    secondary_type: Optional[StrategyType] = None

    # Embedding specific
    model_name: str = "all-mpnet-base-v2"
    optimal_size: tuple[int, int] = (6, 8)
    similarity_threshold: float = 0.3
    use_graph_distance: bool = True
    graph_distance_weight: float = 0.15

    # Generic kwargs for any other parameters or future extensions
    extra_params: dict[str, Any] = field(default_factory=dict)
