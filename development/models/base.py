from typing import Callable, Dict
import pathlib
import numpy as np

from datasets.sequence import DataSequence

DIRNAME = pathlib.Path(__file__).parents[1].resolve() / 'weights'

class Model:
    def __init__(self, dataset_cls: type, algorithm_fn: Callable, dataset_args: Dict=None, algorithm_args: Dict=None):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}{algorithm_fn.__name__}'

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if algorithm_args is None:
            algorithm_args = {}
        self.algorithm = algorithm_fn(self.data.input_shape, self.data.output_shape, **algorithm_args)
        #self.algorithm.summary()

        self.batch_augment_fn = None
        self.batch_format_fn = None

    @property
    def weights_filename(self):
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')

#''''
#Functions to be filled out see: https://github.com/gradescope/fsdl-text-recognizer-project/blob/master/lab6_sln/text_recognizer/models/base.py
#''''

    def fit(self, dataset, batch_size=None, epochs=None, callbacks=[]):
        #Define fit sequence
        train_sequence = DatasetSequence(dataset.x_train, dataset.y_train, batch_size)
        test_sequence = DatasetSequence(dataset.x_train, dataset.y_train, batch_size)
        self.algorithm(
            train_sequence,
            epochs = epochs,
            callbacks = callbacks,
            validation_data = test_sequence,
            use_multiprocessing = False,
            workers = 1,
            shuffle = True
        )

    def evaluate(self, x, y):
        #Define evaluate sequence
        return 'Done'

    def loss(self):
        #Return loss
        return 'Loss type'

    def optimizer(self):
        #Return optimizer
        return 'optimizer'

    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.algorithm.load_weights(self.weights_filename)

    def save_weights(self):
        self.algorithm.save_weights(self.weights_filename)
