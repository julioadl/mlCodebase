from boltons.cacheutils import cachedproperty
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from models.line_model import LineModel
from algorithms.lenet import lenet
from algorithms.utils import slide_window


def line_lstm(input_shape, output_shape, window_width=20, window_stride=14, decoder_dim=128, encoder_dim=128):
    
    image_height, image_width = input_shape
    output_length, num_classes = output_shape
    
    image_input = Input(shape=input_shape)    
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    image_patches = Lambda(slide_window, arguments={'window_width': window_width, 'window_stride': window_stride})(image_reshaped)
    
    convnet = lenet((image_height, window_width, 1), (num_classes,))
    convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
    
    convnet_outputs = TimeDistributed(convnet)(image_patches)
    
    lstm = CuDNNLSTM
    
    encoder_output = lstm(encoder_dim, return_sequences=False, go_backwards=True)(convnet_outputs)
    repeated_encoding = RepeatVector(output_length)(encoder_output)
    decoder_output = lstm(decoder_dim, return_sequences=True)(repeated_encoding)
    softmax_output = TimeDistributed(Dense(num_classes, activation='softmax'))(decoder_output)
    
    model = KerasModel(inputs=image_input, outputs=softmax_output)
    return model