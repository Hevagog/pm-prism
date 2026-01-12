from prism.visualization.graph_viz import GraphVisualizer

from prism.visualization.interactive import InteractiveVisualizer


def run_interactive(decomposer, port: int = 8050) -> None:
    """Run the interactive visualization app."""
    viz = InteractiveVisualizer(decomposer)
    viz.run(port=port)


__all__ = ["GraphVisualizer", "run_interactive"]
