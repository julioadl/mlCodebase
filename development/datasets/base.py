#### Base script for loading datasets
import pathlib


class Dataset:
    @classmethod
    def data_dirname(cls):
        return pathlib.Path('__file__').parents[3].resolve() / 'data'

    def load_or_generate_data(self):
        pass
