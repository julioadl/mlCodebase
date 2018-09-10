from typing import Callable, Dict, Tuple
from sklearn import svm

from .base import Model
from datasets.sklearn_digits import sklearnDigits

class SVMModel(Model):
    def __init__(self, dataset_cls: type=sklearnDigits, algorithm_fn: Callable=None, dataset_args: Dict=None, algorithm_args: Dict=None):
        super().__init__(dataset_cls, algorithm_fn, dataset_args, algorithm_args)

    def evaluate(self, x, y, verbose=True):
        return 'Ok'
