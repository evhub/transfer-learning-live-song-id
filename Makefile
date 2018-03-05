.PHONY: setup
setup:
	pip install keras kapre

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
