from sklearn import datasets
import h5py
import os
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from .base import Dataset

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'newsgroup'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'data.h5'

def _download_and_process_data(categories: Optional['list'] = None, tokenizer: Optional['str'] = 'CountVectorizer'):
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    os.chdir(PROCESSED_DATA_DIRNAME)
    print("Unzipping data")
    newsgroup_data_train = datasets.fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    X_text = newsgroup_data_train['data']
    Y = newsgroup_data_train['target']

    if tokenizer == 'CountVectorizer':
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X_text).todense()
    elif tokenizer == 'TfidfTransformer':
        vectorizer = TfidfTransformer()
        X = vectorizer.fit_transform(X_text).todense()
    else:
        valid_transformer = False
        print("Not a valid transformer")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

    with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
        f.create_dataset('x_train', data=X_train, dtype='u1', compression='lzf')
        f.create_dataset('y_train', data=y_train, dtype='u1', compression='lzf')
        f.create_dataset('x_test', data=X_test, dtype='u1', compression='lzf')
        f.create_dataset('y_test', data=y_test, dtype='u1', compression='lzf')

    print("Done")

class sklearnNewsGroup(Dataset):
    def __init__(self, categories: Optional['list'] = None):
        if not os.path.exists(PROCESSED_DATA_DIRNAME):
            _download_and_process_data(categories=categories)

    def load_or_generate_data(self):
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]

if __name__== '__main__':
    data = sklearnNewsGroup()
    data.load_or_generate_data()
    print(data)
