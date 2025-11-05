lint: pyproject.toml setup.cfg
	isort .
	black .
	flake8 .
	pydocstyle .

clean:
	find . -type f -name "*~" -exec rm -f {} +

test:
	pytest

coverage:
	pytest --cov=utils --cov-report=term-missing --cov-report=html