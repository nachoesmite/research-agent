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

# Test BAML client
test-baml:
	uv run baml-cli test

# Run Evaluations
evals:
	python tests/evaluations.py

# Clean generated files and cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

# Install dependencies
install:
	uv sync