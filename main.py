import logging

from prism.utils import download_sample_logs
from prism.visualization import run_interactive
from prism.core import ProcessDecomposer, DecompositionConfig, StrategyType


SAMPLE_URL = "http://home.agh.edu.pl/~kluza/sample_logs.zip"

logger = logging.Logger(__name__)


def demo_basic_decomposition():
    """Demo using convenience factory functions."""
    sample_dir = download_sample_logs(SAMPLE_URL)
    csv_path = sample_dir / "purchasingExample.csv"

    if not csv_path.exists():
        logger.error(f"Sample file not found: {csv_path}")
        return None

    config = DecompositionConfig(
        strategy_type=StrategyType.EMBEDDING,
        optimal_size=(3, 5),
        similarity_threshold=0.3,
    )
    decomposer = ProcessDecomposer(config)

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
