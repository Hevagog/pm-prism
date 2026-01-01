# PM-PRISM

**P**rocess **M**ining - **P**rocess **R**epresentation with **I**ntelligent **S**ubprocess **M**odeling

A tool for automatic decomposition of process models (DFG, BPMN) into smaller, manageable subprocesses.

## Project Overview

This project implements an algorithm for automatically decomposing complex process models (Directly-Follows Graphs and BPMN models) into smaller subprocesses by identifying key model components.

### Part I: DFG Decomposition (Deadline: 13.01)
- Automatic decomposition of DFG models using graph algorithms
- Multiple decomposition strategies (community detection, SCC, cut vertices, etc.)
- GPT-based subprocess labeling
- Interactive visualization with zoom capabilities

### Part II: BPMN Adaptation (Deadline: 27.01)
- Adaptation of the solution to BPMN models
- Support for BPMN-specific elements (gateways, events, subprocesses)
- Integration with bpmn.js for visualization

## Quick Start

```python
from src.decomposer import ProcessDecomposer

# Create decomposer
decomposer = ProcessDecomposer(strategy='community')

# Load and decompose from CSV event log
result = decomposer.decompose_from_csv(
    "sample_logs/repairExample.csv",
    case_id='Case ID',
    activity_key='Activity',
    timestamp_key='Start Timestamp'
)

# Print summary
print(decomposer.summary())

# Visualize with Plotly (interactive)
fig = decomposer.visualize(method='plotly')
fig.show()

# Or launch full Dash web UI
app = decomposer.visualize(method='dash')
app.run(debug=True)
```

## Architecture

```
pm-prism/
├── src/
│   ├── core/
│   │   ├── base.py           # Abstract base classes
│   │   └── decomposition.py  # Decomposition strategies
│   ├── adapters/
│   │   ├── dfg_adapter.py    # DFG loading (PM4Py)
│   │   └── bpmn_adapter.py   # BPMN support (Part II)
│   ├── visualization/
│   │   └── graph_viz.py      # Plotly/Dash visualization
│   ├── labeling/
│   │   └── __init__.py       # GPT/heuristic labeling
│   └── decomposer.py         # Main orchestrator
├── main.py                   # Demo entry point
└── tests/                    # Unit tests
```

## Decomposition Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `community` | Louvain community detection | General decomposition |
| `scc` | Strongly connected components | Identifying loops/cycles |
| `cut_vertex` | Articulation point based | Finding critical nodes |
| `gateway` | High-degree node based | Decision-heavy processes |
| `hierarchical` | Multi-level decomposition | Large, complex models |

## Features

- **Multiple Input Formats**: CSV, XES, pandas DataFrame, or pre-computed DFG
- **Flexible Strategies**: Choose from various graph decomposition algorithms
- **Intelligent Labeling**: GPT-based or heuristic subprocess naming
- **Interactive UI**: Zoom, pan, and explore subprocesses with Dash
- **Extensible Design**: Easy to add new strategies and model types

## Environment Variables

For GPT-based labeling:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Running the Demo

```bash
python main.py
```

This will:
1. Download sample event logs
2. Decompose the repair process example
3. Optionally show interactive visualization
4. Compare different strategies

## Development

```bash
# Run tests
pytest

# Lint code
ruff check src/

# Format code
ruff format src/
```

## Dependencies

- **networkx**: Graph analysis and decomposition algorithms
- **pandas**: Event log handling
- **pm4py**: Process mining (DFG/BPMN discovery)
- **plotly**: Interactive visualization
- **dash**: Web UI framework
- **openai** (optional): GPT-based labeling
