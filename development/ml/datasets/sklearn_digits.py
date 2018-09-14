from sklearn import datasets
import h5py
import os
from sklearn.model_selection import train_test_split

from .base import Dataset

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'digits'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'data.h5'

def _download_and_process_digits():
    #RAW_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    os.chdir(PROCESSED_DATA_DIRNAME)
    print("Unzipping Digits")
    digits_data = datasets.load_digits()
    n_samples = len(digits_data.images)
    X = digits_data.images.reshape((n_samples, -1))
    Y = digits_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

    with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
        f.create_dataset('x_train', data=X_train, dtype='u1', compression='lzf')
        f.create_dataset('y_train', data=y_train, dtype='u1', compression='lzf')
        f.create_dataset('x_test', data=X_test, dtype='u1', compression='lzf')
        f.create_dataset('y_test', data=y_test, dtype='u1', compression='lzf')

    print('Digits done')



def _sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    '''See line 128: https://github.com/gradescope/fsdl-text-recognizer-project/blob/master/lab6_sln/text_recognizer/datasets/emnist.py'''
    print('To be implemented')

class sklearnDigits(Dataset):
    def __init__(self):
        if not os.path.exists(PROCESSED_DATA_DIRNAME):
            _download_and_process_digits()

    def load_or_generate_data(self):
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]

if __name__ ==  '__main__':
    data = sklearnDigits()
    data.load_or_generate_data()
    print(data)
