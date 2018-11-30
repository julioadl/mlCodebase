#!/bin/sh
pipenv run python run_experiment.py --gpu=-1 --save '{"dataset": "EmnistDataset", "model": "EmnistModel", "algorithm": "lenet", "train_args": {"epochs": 8}}'
