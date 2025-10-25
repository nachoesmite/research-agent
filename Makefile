.PHONY: dev generate-baml research test-baml clean help demo-ai demo-quantum interactive test

# Default target
help:
	@echo "Available commands:"
	@echo "  make dev          - Start LangGraph development server"
	@echo "  make generate-baml - Generate BAML client code"
	@echo "  make research     - Run default research (AI productivity)"
	@echo "  make interactive  - Run interactive research mode"
	@echo "  make test         - Test analyst creation only"
	@echo "  make demo-ai      - Demo: AI coding assistants research"
	@echo "  make demo-quantum - Demo: Quantum computing research"
	@echo "  make test-baml    - Test BAML client"
	@echo "  make clean        - Clean generated files"
	@echo "  make help         - Show this help message"

# Start LangGraph development server
dev:
	uv run langgraph dev

# Generate BAML client code
generate-baml:
	uv run baml generate

# Run default research
research:
	uv run python main.py

# Run interactive research mode
interactive:
	uv run python main.py interactive

# Test analyst creation only
test:
	uv run python main.py test

# Demo: AI coding assistants research
demo-ai:
	uv run python main.py demo-ai

# Demo: Quantum computing research
demo-quantum:
	uv run python main.py demo-quantum

# Test BAML client
test-baml:
	uv run python test_baml.py

# Clean generated files and cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

# Install dependencies
install:
	uv sync