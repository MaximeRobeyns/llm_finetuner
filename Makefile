.PHONY: install test alltest

install:
	pip3 install -e .[all] --extra-index-url https://download.pytorch.org/whl/cu113

test:
	python -m pytest

kernel:  ## To setup a Jupyter kernel to run notebooks in the project's virtual env
	python -m ipykernel install --user --name llm_vae \
		--display-name "llm_vae (Python 3.9)"

lab: ## To start a Jupyter Lab server
	jupyter lab --notebook-dir=explore --ip=0.0.0.0 --port 8883 # --collaborative --no-browser

alltest:
	python -m pytest --runslow
