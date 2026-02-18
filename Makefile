.PHONY: setup test clean help install-uv test-dummy test-figures run-all run-example1

help:
	@echo "Deep-PrAE Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create environment and install dependencies (using uv)"
	@echo "  make install-uv     - Install uv package manager"
	@echo ""
	@echo "Testing:"
	@echo "  make test-dummy     - Run all examples in dummy/test mode"
	@echo "  make test-figures   - Dummy run + generate figures"
	@echo "  make test-import    - Verify package imports correctly"
	@echo ""
	@echo "Running:"
	@echo "  make run-example1   - Run Example 1 with paper specs"
	@echo "  make run-all        - Run all experiments (requires Gurobi)"
	@echo ""
	@echo "Figures:"
	@echo "  make figures        - Generate paper-like figures (dummy mode)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove generated files"
	@echo "  make clean-all      - Remove environment and generated files"

install-uv:
	@echo "Installing uv..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "uv installed. Please run 'source $$HOME/.cargo/env' or restart your terminal"

setup:
	@echo "Setting up Deep-PrAE environment..."
	@./setup_env.sh

setup-manual:
	@echo "Manual setup instructions:"
	@echo "1. Create venv: uv venv .venv --python 3.10"
	@echo "2. Activate: source .venv/bin/activate"
	@echo "3. Install deps: uv pip install -e ."
	@echo "4. Test: python run_all_examples.py --all --test"

test-import:
	@echo "Testing package import..."
	@python -c "from deepprae import DeepPrAE; print('Import OK')"

test-dummy:
	@echo "Running all examples in dummy/test mode..."
	@python run_all_examples.py --all --test

test-figures:
	@echo "Running dummy mode with figure generation..."
	@python run_all_examples.py --all --test --figures

figures:
	@echo "Generating paper-like figures (dummy mode)..."
	@python run_all_examples.py --all --test --figures

run-example1:
	@echo "Running Example 1 (requires Gurobi)..."
	@python run_all_examples.py --examples 1 --gamma 1.8

run-all:
	@echo "Running all experiments (requires Gurobi)..."
	@python run_all_examples.py --all

clean:
	@echo "Cleaning generated files..."
	@rm -f *.json *.png *.pdf
	@rm -rf results/
	@rm -rf __pycache__ deepprae/__pycache__ deepprae/*/__pycache__
	@find . -name "*.pyc" -delete
	@find . -name ".DS_Store" -delete
	@echo "Cleaned"

clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf .venv
	@echo "Complete cleanup done"

check-env:
	@if [ -d ".venv" ]; then \
		echo "Virtual environment exists"; \
		.venv/bin/python --version; \
	else \
		echo "Virtual environment not found"; \
		echo "Run 'make setup' first"; \
	fi
