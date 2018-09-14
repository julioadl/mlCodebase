from typing import Optional
from sklearn import svm

def SVM(kernel: Optional['str'] = None):
    classifier = svm.SVC()
    return classifier
