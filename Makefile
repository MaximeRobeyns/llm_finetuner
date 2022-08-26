.PHONY: install test alltest docs

install:  ## Install the project (assuming cuda 11.3)
	pip3 install -e .[all] --extra-index-url https://download.pytorch.org/whl/cu113

test:  ## Run the unit tests
	python -m pytest

kernel:  ## To setup a Jupyter kernel to run notebooks in the project's virtual env
	python -m ipykernel install --user --name llm_vae \
		--display-name "llm_vae (Python 3.9)"

lab: ## To start a Jupyter Lab server
	jupyter lab --notebook-dir=explore --ip=0.0.0.0 --port 8883 # --collaborative --no-browser

alltest: ## Run all the tests.
	python -m pytest --runslow

docs:  ## Start the mkdocs server
	@./docs/writedocs.sh

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
