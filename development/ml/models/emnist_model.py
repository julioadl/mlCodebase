from typing import Callable, Dict, Tuple
from sklearn import svm, metrics
import numpy as np
import pathlib
import tensorflow
from tensorflow.keras.models import load_model
from sklearn.externals import joblib

from .base import ModelTf as Model
from datasets.sklearn_digits import sklearnDigits
from datasets.sequence import DatasetSequence

DIRNAME = pathlib.Path('__file__').parents[0].resolve() / 'weights'

class EmnistModel(Model):
    def __init__(self, dataset_cls: type=sklearnDigits, algorithm_fn: Callable=None, dataset_args: Dict=None, algorithm_args: Dict=None):
        super().__init__(dataset_cls, algorithm_fn, dataset_args, algorithm_args)

'''
Correct these to work with Tf
    def save_model(self):
        joblib.dump(self.algorithm, self.weights_filename)

    def load_model(self):
        self.algorithm = joblib.load(self.weights_filename)
'''
