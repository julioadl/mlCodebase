'''
EMNIST dataset
'''
import json
import os
import pathlib
from urllib.request import urlretrieve
import zipfile

from boltons.cacheutils import cachedproperty
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

from .base import Dataset

RAW_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'emnist'
PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'emnist'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'data.h5'
ESSENTIALS_FILENAME = pathlib.Path('__file__').parents[0].resolve() / 'emnist_essentials.json'

def _download_and_process_data(sample_to_balance=False):
    import scipy.io

    RAW_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)

    os.chdir(RAW_DATA_DIRNAME)

    if not os.path.exists('matlab.zip'):
        print('Downloading EMNIST...')
        urlretrieve(RAW_URL, 'matlab.zip')

    print('Unzipping EMNIST and locating .mat file...')
    zip_file = zipfile.ZipFile('matlab.zip', 'r')
    zip_file.extract('matlab/emnist-byclass.mat',)
    data = scipy.io.loadmat('matlab/emnist-byclass.mat')

    print('Saving to HDF5...')
    x_train = data['dataset']['train'][0, 0]['images'][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data['dataset']['train'][0, 0]['labels'][0, 0]
    x_test = data['dataset']['test'][0, 0]['images'][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data['dataset']['test'][0, 0]['labels'][0, 0]

    if sample_to_balance:
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)

    with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
        f.create_dataset('x_train', data=x_train, dtype='u1', compression='lzf')
        f.create_dataset('y_train', data=y_train, dtype='u1', compression='lzf')
        f.create_dataset('x_test', data=x_test, dtype='u1', compression='lzf')
        f.create_dataset('y_test', data=y_test, dtype='u1', compression='lzf')

    print('Saving essential dataset parameters...')
    mapping = {int(k): chr(v) for k, v in data['dataset']['mapping'][0, 0]}
    essentials = {'mapping': list(mapping.items()), 'input_shape': list(x_train.shape[1:])}
    with open(ESSENTIALS_FILENAME, 'w') as f:
        json.dump(essentials, f)

    print('EMNIST downloaded and processed')

def _sample_to_balance(x, y):
    '''Because the dataset is not balances, we take at most the number of instances per class'''
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled

def _augment_emnist_mapping(mapping):
    '''Extra symbols for mapping'''
    extra_symbols = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?', '_']
    max_key = max(mapping.keys())
    extra_mapping = {}
    for i, symbol in enumerate(extra_symbols):
        extra_mapping[max_key + 1 + i] = symbol

    return {**mapping, **extra_mapping}

class EmnistDataset(Dataset):
    def __init__(self):
        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_data()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        self.mapping = _augment_emnist_mapping(dict(essentials['mapping']))
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.num_classes = len(self.mapping)
        self.input_shape = essentials['input_shape']
        self.output_shape = (self.num_classes,)

    def load_or_generate_data(self):
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_data()
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train_int = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test_int = f['y_test'][:]

    @cachedproperty
    def y_train(self):
        return to_categorical(self.y_train_int, self.num_classes)

    @cachedproperty
    def y_test(self):
        return to_categorical(self.y_test_int, self.num_classes)

    def __repr__(self):
        return (
            'Emnist Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Mapping: {self.mapping}\n'
            f'Input shape: {self.input_shape}\n'
        )

if __name__ == '__main__':
    data = EmnistDataset()
    data.load_or_generate_data()
    print(data)
    print(data.x_train.shape, data.y_train.shape)
    print(data.x_test.shape, data.y_test.shape)
