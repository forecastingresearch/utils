PYTEST_ARGS ?=

lint: pyproject.toml setup.cfg
	isort .
	black .
	flake8 .
	pydocstyle .

clean:
	find . -type f -name "*~" -exec rm -f {} +

test:
	pytest $(PYTEST_ARGS)

test-integration:
	pytest --integration $(PYTEST_ARGS)

test-integration-parallel:
	pytest --integration -n auto $(PYTEST_ARGS)

coverage:
	pytest --cov=utils --cov-report=term-missing --cov-report=html $(PYTEST_ARGS)
