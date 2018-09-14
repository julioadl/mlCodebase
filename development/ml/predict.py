from typing import Tuple, Union

from numpy as np

from models import SVMModel

class predictor:
    def __init__(self):
        self.model = SVMModel()
        self.model.load_model()

    def predict(self, data_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        if isinstance(data_or_filename, str):
            #write util to retrieve data from url and convert it to numpy ndarray
            #Substitute the line below
            data = 'data'
        else:
            data = data_or_filename
        return self.model.predict(data)
