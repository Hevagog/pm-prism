import logging

from prism.utils import download_sample_logs
from prism.core import ProcessDecomposer

SAMPLE_URL = "http://home.agh.edu.pl/~kluza/sample_logs.zip"

logger = logging.Logger(__name__)

 
def demo_basic_decomposition():
    sample_dir = download_sample_logs(SAMPLE_URL)
    csv_path = sample_dir / "repairExample.csv"

    if not csv_path.exists():
        logger.error(f"Sample file not found: {csv_path}")
        return

    decomposer = ProcessDecomposer(
        strategy="community",
        strategy_kwargs={"resolution": 1.0, "min_community_size": 2},
    )

    result = decomposer.decompose_from_csv(
        str(csv_path),
        case_id="Case ID",
        activity_key="Activity",
        timestamp_key="Start Timestamp",
    )

    print("\n" + decomposer.summary())

    return decomposer


def demo_interactive_visualization(decomposer):
    fig = decomposer.visualize(method="plotly", title="Repair Process - Decomposed")
    fig.show()


def main():
    # download_sample_logs(SAMPLE_URL)
    decomposer = demo_basic_decomposition()
    if decomposer:
        try:
            response = input(
                "\nWould you like to see the interactive visualization? (y/n): "
            )
            if response.lower() in ("y", "yes"):
                demo_interactive_visualization(decomposer)
        except EOFError:
            pass

if __name__ == "__main__":
    main()
