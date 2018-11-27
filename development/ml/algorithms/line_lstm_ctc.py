import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from models.line_model import LineModel
from algorithms.lenet import lenet
from algorithms.utils import slide_window
from algorithms.ctc import ctc_decode


def line_lstm_ctc(input_shape, output_shape, window_width=28, window_stride=14):
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    #Take the length of the window, divide it between the number of desired divisions and add 1 to pad
    num_windows = int((image_width - window_width) / window_stride) + 1
    if num_windows < output_length:
        raise ValueError(f'Window width/stride needs to generate at least {output_length} windows (currently {num_windows})')

    image_input = Input(shape=input_shape, name='image')
    y_true = Input(shape=(output_length,), name='y_true')
    input_length = Input(shape=(1,), name='input_length')
    label_length = Input(shape=(1,), name='label_length')

    #Cuda implementation for LSTM; much faster
    lstm_fn = CuDNNLSTM

    #Reshape image to add 3rd dimension as required by keras
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)

    #get image patches into Keras Layer; generates (num_windows, image_height, window_width, 1)
    image_patches = Lambda(slide_window,
                           arguments={'window_width': window_width, 'window_stride': window_stride})(image_reshaped)

    #Make a lenet and get rid of the last two layers; outputs (num_windows, 128)
    #128 because of convolutions?
    convnet = lenet((image_height, window_width, 1), (num_classes,))
    convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
    convnet_outputs = TimeDistributed(convnet)(image_patches)

    #Include LSTM; outputs (num_windows, 128)
    lstm_output = lstm_fn(128, return_sequences=True)(convnet_outputs)

    #And do predictions; output (num_windows, num_classes)
    softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)

    #Implement CTC loss
    input_length_processed = Lambda(
        lambda x, num_windows=None: x * num_windows,
        arguments={'num_windows': num_windows}
    )(input_length)

    ctc_loss_output = Lambda(
        lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]),
        name='ctc_loss'
    )([y_true, softmax_output, input_length_processed, label_length])

    ctc_decoded_output = Lambda(
        lambda x: ctc_decode(x[0], x[1], output_length),
        name='ctc_decoded'
    )([softmax_output, input_length_processed])

    model = KerasModel(
        inputs=[image_input, y_true, input_length, label_length],
        outputs=[ctc_loss_output, ctc_decoded_output]
    )
    return model
    
