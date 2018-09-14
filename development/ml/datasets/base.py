#### Base script for loading datasets
import pathlib


class Dataset:
    @classmethod
    def data_dirname(cls):
        return pathlib.Path('__file__').parents[0].resolve() / 'data'

    def load_or_generate_data(self):
        pass
