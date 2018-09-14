from typing import Callable, Dict, Tuple
from sklearn import svm, metrics
import numpy as np
import pathlib
from sklearn.externals import joblib

from .base import Model
from datasets.sklearn_digits import sklearnDigits
from datasets.sequence import DatasetSequence

DIRNAME = pathlib.Path('__file__').parents[0].resolve() / 'weights'

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

    @property
    def weights_filename(self):
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.pkl')

    def save_model(self):
        joblib.dump(self.algorithm, self.weights_filename)

    def load_model(self):
        self.algorithm = joblib.load(self.weights_filename)
