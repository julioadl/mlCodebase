import pathlib
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from .lenet import lenet
from .utils import slide_window


def lenet_sliding_windows(input_shape: Tuple[int, ...],
                          output_shape: Tuple[int, ...],
                          window_width: float=16,
                          window_stride: float=10) -> KerasModel:
    '''
    Input is an image with shape (image_height, image_width)
    Output is of shape (output_length, num_classes)
    '''
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    #Instantiate Keras tensor with input of dimensions input_shape
    image_input = Input(shape=input_shape)

    #Reshape to keras required form
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    '''
    Obtain "patches" from the long-line image in shape (num_windows, image_height, window_width, 1);
    use lambda to wrap it as a layer object
    '''
    image_patches = Lambda(
        slide_window,
        arguments={'window_width': window_width, 'window_stride': window_stride}
        )(image_reshaped)

    #Make a lenet and get rid of the last two layers
    convnet = lenet((image_height, window_width, 1), (num_classes,))
    convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
    #Make use of Wrapper Time Distributed to apply convnet independently to each of the patches
    #This will result on an output of (num_windows, 128)
    convnet_outputs = TimeDistributed(convnet)(image_patches)

    #Calculate the output_dimensions. Notice it has to be of size at least 2. tf.expand_dims will add a dim to the last dim
    #This will result in (num_windows, 128, 1)
    convnet_outputs_extra_dim = Lambda(lambda x: tf.expand_dims(x, -1))(convnet_outputs)

    #Calculate num_windows
    num_windows = int((image_width - window_width) / window_stride) + 1
    width = int(num_windows / output_length)

    #Add a conv2d to the outputs with the extra dimension
    conved_convnet_outputs = Conv2D(num_classes, (width, 128), (width, 1), activation='softmax')(convnet_outputs_extra_dim)

    #Squeeze the extra dimension
    squeezed_conved_convnet_outputs = Lambda(lambda x: tf.squeeze(x, 2))(conved_convnet_outputs)
    # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
    softmax_output = Lambda(lambda x: x[:, :output_length, :])(squeezed_conved_convnet_outputs)

    model = KerasModel(inputs=image_input, outputs=softmax_output)
    model.summary()
    return model
