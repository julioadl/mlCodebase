from typing import Callable, Dict, Tuple
from sklearn import svm, metrics
import numpy as np
import pickle

from .base import Model
from datasets.sklearn_digits import sklearnDigits
from datasets.sequence import DatasetSequence

class SVMModel(Model):
    def __init__(self, dataset_cls: type=sklearnDigits, algorithm_fn: Callable=None, dataset_args: Dict=None, algorithm_args: Dict=None):
        super().__init__(dataset_cls, algorithm_fn, dataset_args, algorithm_args)

    def evaluate(self, x, y):
        sequence = DatasetSequence(x, y, batch_size=12)
        preds = self.algorithm.predict(sequence.x)
        report = metrics.classification_report(sequence.y, preds)
        return report

    def predict(self, input: np.ndarray) -> Tuple[str, float]:
        pred = self.algorithm.predict(input)
        probability_all_classes = self.algorithm.predict_proba(input)
        max_prob_idx = np.argmax(probability_all_classes)
        return (str(pred), probability_all_classes[max_prob_idx])

    #A better name would be save model, but trying to preserve original names
    def save_weights(self):
        return 'ok'
