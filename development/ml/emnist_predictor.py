from typing import Tuple, Union

import numpy as np

from models import EmnistModel
from utils import *

class EmnistPredictor:
    def __init__(self):
        self.model = EmnistModel()
        self.model.load_weights()

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        if isinstance(image_or_filename, str):
            image = read_image(image_or_filename, grayscale=True)
        else:
            image = image_or_filename
        return self.model.predict_on_image(image)

    def evaluate(self, dataset):
        return self.model.evaluate(dataset.x_test, dataset.y_test)
