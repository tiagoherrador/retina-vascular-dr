IMAGES ?= data/images
CSV ?= data/train.csv
.PHONY: install test extract evaluate
install:
	pip install -e ".[dev,fcunet,explain,ordinal]"
test:
	pytest -q
extract:
	retina-extract --csv $(CSV) --images $(IMAGES) --out results/features.csv
evaluate:
	retina-evaluate --features results/features.csv --out results
