from typing import Optional
from prism.core.base import DecompositionStrategy, SubprocessLabeler
from prism.core.config import DecompositionConfig, StrategyType
from prism.core.decompositions.louvain import CommunityDetectionStrategy
from prism.core.decompositions.cut_vertex import CutVertexStrategy
from prism.core.decompositions.gateway_based import GatewayBasedStrategy
from prism.core.decompositions.scd import SCCDecompositionStrategy
from prism.core.decompositions.hierarchical import HierarchicalDecompositionStrategy
from prism.core.embedding_strategy import EmbeddingClusteringStrategy


class DecompositionStrategyFactory:
    """Factory for creating decomposition strategies based on configuration."""

    @staticmethod
    def create_strategy(
        config: DecompositionConfig, labeler: Optional[SubprocessLabeler] = None
    ) -> DecompositionStrategy:
        """
        Create a decomposition strategy instance from configuration.

        Args:
            config: The decomposition configuration.
            labeler: Optional labeler to be used by strategies that support it.

        Returns:
            An instance of a DecompositionStrategy.
        """
        return DecompositionStrategyFactory._create_from_type(
            config.strategy_type, config, labeler
        )

    @staticmethod
    def _create_from_type(
        strategy_type: StrategyType,
        config: DecompositionConfig,
        labeler: Optional[SubprocessLabeler] = None,
    ) -> DecompositionStrategy:
        match strategy_type:
            case StrategyType.LOUVAIN:
                return CommunityDetectionStrategy(
                    resolution=config.resolution,
                    min_community_size=config.min_size,
                    labeler=labeler,
                )
            case StrategyType.CUT_VERTEX:
                return CutVertexStrategy()
            case StrategyType.GATEWAY:
                return GatewayBasedStrategy(degree_threshold=config.degree_threshold)
            case StrategyType.SCC:
                return SCCDecompositionStrategy(min_scc_size=config.min_size)
            case StrategyType.HIERARCHICAL:
                primary_type = config.primary_type or StrategyType.LOUVAIN
                primary_strategy = DecompositionStrategyFactory._create_from_type(
                    primary_type, config, labeler
                )

                secondary_strategy = None
                if config.secondary_type:
                    secondary_strategy = DecompositionStrategyFactory._create_from_type(
                        config.secondary_type, config, labeler
                    )

                return HierarchicalDecompositionStrategy(
                    primary_strategy=primary_strategy,
                    secondary_strategy=secondary_strategy,
                    max_subprocess_size=config.max_subprocess_size,
                )
            case StrategyType.EMBEDDING:
                return EmbeddingClusteringStrategy(
                    model_name=config.model_name,
                    optimal_size=config.optimal_size,
                    similarity_threshold=config.similarity_threshold,
                    use_graph_distance=config.use_graph_distance,
                    graph_distance_weight=config.graph_distance_weight,
                    labeler=labeler,
                )
            case _:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
