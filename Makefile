.PHONY: clean install test lint docs

clean:
	rm -rf ./build ./dist ./src/xileh.egg-info ./_site ./.quarto ./docs/api
	find . -name __pycache__ | xargs rm -rf
	find . -name '*.pyc' | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf

install:
	uv sync --extra all

test:
	uv run --extra dev pytest tests/ --cov=src/xileh

lint:
	uv run --extra dev ruff check src/ tests/

docs:
	uv run --extra docs quartodoc build --config _quartodoc.yml
	quarto render
