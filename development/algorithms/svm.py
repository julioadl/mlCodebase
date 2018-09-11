from typing import Optional
from sklearn import svm

def SVM(input_shape, optput_shape, kernel: Optional['str'] = None):
    classifier = svm.SVC(kernel=kernel)
    return classifier
