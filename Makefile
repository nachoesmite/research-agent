.PHONY: dev generate-baml research test-baml clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make dev          - Start LangGraph development server"
	@echo "  make generate-baml - Generate BAML client code"
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
	uv run python test_baml.py

# Clean generated files and cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

# Install dependencies
install:
	uv sync