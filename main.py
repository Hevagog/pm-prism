"""Demo script showing process decomposition capabilities."""

import logging

from prism.utils import download_sample_logs
from prism.core import ProcessDecomposer, embedding_strategy
from prism.visualization import run_interactive

# Imports available for Option 2 (full customization):
# from prism.core import EmbeddingClusteringStrategy, LLMLabeler
# Imports available for Option 3 (community detection):
# from prism.core import community_strategy

SAMPLE_URL = "http://home.agh.edu.pl/~kluza/sample_logs.zip"

logger = logging.Logger(__name__)


def demo_basic_decomposition():
    """Demo using convenience factory functions."""
    sample_dir = download_sample_logs(SAMPLE_URL)
    csv_path = sample_dir / "purchasingExample.csv"

    if not csv_path.exists():
        logger.error(f"Sample file not found: {csv_path}")
        return None

    # Option 1: Use factory function (simplest)
    strategy = embedding_strategy(
        optimal_size=(3, 5),
        similarity_threshold=0.3,
    )

    # Option 2: Full customization with strategy class
    # labeler = LLMLabeler()  # Requires GROQ_API_KEY in .env
    # strategy = EmbeddingClusteringStrategy(
    #     optimal_size=(3, 5),
    #     similarity_threshold=0.3,
    #     use_graph_distance=True,
    #     graph_distance_weight=0.2,
    #     labeler=labeler,
    # )

    # Option 3: Community detection
    # strategy = community_strategy(resolution=1.0)

    decomposer = ProcessDecomposer(strategy)

    decomposer.decompose_from_csv(
        str(csv_path),
        case_id="Case ID",
        activity_key="Activity",
        timestamp_key="Start Timestamp",
    )

    print("\n" + decomposer.summary())

    return decomposer


def demo_interactive_app(decomposer: ProcessDecomposer | None):
    """
    Launch interactive visualization app.
    
    - Use slider to change abstraction level
    - Click on any cluster to drill down into its internal structure
    - Click 'Back to Overview' to return
    """
    if decomposer is None:
        return
    
    run_interactive(decomposer, port=8050)


def main():
    decomposer = demo_basic_decomposition()
    demo_interactive_app(decomposer)


if __name__ == "__main__":
    main()
