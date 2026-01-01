#PM-PRISM

**P**rocess **M**ining - **P**rocess **R**epresentation with **I**ntelligent **S**ubprocess **M**odeling

PM-PRISM decomposes process models (currently DFG-focused) into smaller subprocesses with multiple algorithms, optional semantic labeling, and interactive visualization.

## Highlights
- Multiple decomposition strategies: Louvain community detection, SCC, cut-vertex, gateway-based, hierarchical, and embedding-based clustering.
- Configuration-first API via `DecompositionConfig` and `StrategyType` with optional labelers.
- Input flexibility: CSV, XES, pandas DataFrame, or pre-computed DFG.
- Visualization: Plotly-based graph views and hierarchical exploration; interactive Dash helper in `visualization/interactive.py`.

## Quick Start

```bash
pip install -e .
python main.py
```

Minimal programmatic example:

```python
from prism.core import ProcessDecomposer, DecompositionConfig, StrategyType

config = DecompositionConfig(
    strategy_type=StrategyType.LOUVAIN,
    resolution=1.0,
    min_size=2,
)

decomposer = ProcessDecomposer(config)

result = decomposer.decompose_from_csv(
    "sample_logs/repairExample.csv",
    case_id="Case ID",
    activity_key="Activity",
    timestamp_key="Start Timestamp",
)

print(decomposer.summary())
fig = decomposer.visualize(method="plotly")
fig.show()
```

## Repository Layout

```

├── main.py                      # Demo entry point
├── prism/
│   ├── core/
│   │   ├── base.py             # Core dataclasses and abstract interfaces
│   │   ├── config.py           # DecompositionConfig and StrategyType
│   │   ├── decompositions/     # Strategy implementations
│   │   ├── decomposer.py       # ProcessDecomposer orchestrator
│   │   └── embedding_strategy.py
│   ├── adapters/dfg_adapter.py # DFG loading via pm4py
│   ├── visualization/          # Plotly/interactive helpers
│   └── utils/                  # Download helpers
├── sample_logs/                # Sample CSV event logs
└── tests/                      # Unit tests
```

## Configurable Strategies

Use `StrategyType` with `DecompositionConfig`:

- `StrategyType.LOUVAIN` – community detection
- `StrategyType.SCC` – strongly connected components
- `StrategyType.CUT_VERTEX` – articulation-point blocks
- `StrategyType.GATEWAY` – high-degree gateway-like splits
- `StrategyType.HIERARCHICAL` – primary/secondary strategy chaining
- `StrategyType.EMBEDDING` – semantic clustering with sentence-transformers

Create strategies manually with `DecompositionStrategyFactory` if needed.

## Features
- Decompose DFGs from CSV, XES, DataFrame, or dict DFG.
- Optional subprocess labeling via `SubprocessLabeler` (LLM or simple labeler).
- Hierarchical decomposition support and abstract graph generation.
- Plotly-based visualization for flat and hierarchical decompositions.

## Environment

Optional for LLM labeling:
```bash
export GROQ_API_KEY="your-key"
```

## Development

```bash
python -m pytest
```

Key dependencies: networkx, pandas, pm4py, plotly, dash, sentence-transformers (for embedding strategy).
python main.py
