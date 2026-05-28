.PHONY: test preprocess train pipeline infer all

PYTHON ?= python

test:
	$(PYTHON) -m pytest -q

preprocess:
	$(PYTHON) src/preprocess.py

train:
	$(PYTHON) src/train.py

pipeline: preprocess train

infer:
	$(PYTHON) src/predict.py --input-features data/processed/polygram_features.csv --model-path models/best_model.pkl --output-path reports/predictions_inference.csv

all: test pipeline infer
