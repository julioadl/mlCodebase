from typing import Callable, Dict, Tuple
import numpy as np
import pathlib
import tensorflow
from tensorflow.keras.models import load_model

from .base import ModelTf as Model
from datasets.emnist import EmnistDataset
from datasets.sequence import DatasetSequence
from algorithms.lenet import lenet

DIRNAME = pathlib.Path('__file__').parents[0].resolve() / 'weights'

class EmnistModel(Model):
    def __init__(self, dataset_cls: type=EmnistDataset, algorithm_fn: Callable=lenet, dataset_args: Dict=None, algorithm_args: Dict=None):
        super().__init__(dataset_cls, algorithm_fn, dataset_args, algorithm_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.algorithm.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[ind]
        predicted_character = self.data.mapping[ind]
        return predicted_character, confidence_of_prediction
