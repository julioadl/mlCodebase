from time import time
from typing import Callable, Optional, Union, Tuple

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import RMSprop
#Try to use wandb
#import wandb
#from wandb.keras import WandbCallback

from datasets.base import Dataset
from models.base import Model
#from training.gpu_util_sampler import GPUUtilizationSample

EARLY_STOPPING = True
GPU_UTIL_SAMPLE = True

def train_model(model: Model, dataset: Dataset, epochs: Optional[int] = None, gpu_ind: Optional[int] = None, use_wandb=False) -> Model:
    callbacks = []

#   Early stopping with tensorflow
#    if EARLY_STOPPING:
#       early_stopping = EarlyStopping(monitor='val_loss', mon_delta=0.01, patience = 3, verbose=1, mode='auto')
#       ballbacks.append(early_stopping)

    if GPU_UTIL_SAMPLE and gpu_ind is not None:
        gpu_utilization = GPUUtilizationSampler(gpu_ind)
        callbacks.append(gpu_utilization)

    if use_wandb:
        wandb = WandbCallback()
        callbacks.append(wandb)

    t = time()
    #Train Neural Network. Originally:
    history = model.fit(dataset, batch_size, epochs, callbacks)
    '''
    See line 41 of https://github.com/gradescope/fsdl-text-recognizer-project/blob/master/lab6_sln/training/util.py
    '''
    #history = model.fit(dataset)
    print('Training took {:2f} s'.format(time() - t))

    if GPU_UTIL_SAMPLE and gpu_ind is not None:
        gpu_utilizations = gpu_utilization.samples
        print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)} +- {round(np.std(gpu_utilizations), 2)}')

    return model

    print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)}')
