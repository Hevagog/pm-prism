from prism.visualization.graph_viz import GraphVisualizer

# Lazy imports to avoid circular dependency
def run_interactive(decomposer, port: int = 8050) -> None:
    """Run the interactive visualization app."""
    from prism.visualization.interactive import InteractiveVisualizer
    viz = InteractiveVisualizer(decomposer)
    viz.run(port=port)

__all__ = ["GraphVisualizer", "run_interactive"]
