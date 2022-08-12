.PHONY: install test alltest

install:
	pip3 install -e .[all] --extra-index-url https://download.pytorch.org/whl/cu113

test:
	python -m pytest

alltest:
	python -m pytest --runslow
