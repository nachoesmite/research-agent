# Research Agent

A modular research agent built with LangGraph, BAML, and LangSmith. This project orchestrates research workflows using graphs, integrates LLM-based evaluation, and supports custom tools and memory.

## Project Structure
- `main.py`: Just a main to exercise the agent, dev experiment do not use; will be deprecated.
- `baml_src/`: BAML source files for graph and client definitions.
- `graphs/`: Python files for graph logic and node implementations.
- `tests/`: Evaluation and test scripts.
- `research_graph_notebook.ipynb`: Interactive notebook for graph exploration and experimentation.

## Quick Links
- [Interactive Notebook](./research_graph_notebook.ipynb)

## How to Run

### 1. BAML Tests
Run BAML tests using the CLI:
```bash
make test-baml
```

### 2. Evaluations
Run logic-based and LLM-based evaluations:
```bash
make evals
```

### 3. LangGraph Development Server
Start LangGraph Studio for interactive graph development:
```bash
make dev
```

## Environment Setup
- Copy `.env.example` to `.env` and fill in your API keys.
- Install dependencies and get ready to use:
```bash
uv sync
source .venv/bin/activate
```

## License
MIT
