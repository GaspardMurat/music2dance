from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D, Conv2D, BatchNormalization, TimeDistributed
from keras import optimizers


def convLSTM2dModel(input_shape, output_shape, learning_rate):
    time_step = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    depth = input_shape[3]

    main_input = Input(shape=(time_step, height, width, depth), name='main_input')

    # Conv-Encoder

    conv_enc1 = TimeDistributed(
        Conv2D(filters=32, kernel_size=(6, 4), strides=(4, 1), padding='same', activation='Relu'))(main_input)
    conv_enc1_bn = TimeDistributed(BatchNormalization)(conv_enc1)

    conv_enc2 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 1), padding='same', activation='Relu'))(conv_enc1_bn)
    conv_enc2_bn = TimeDistributed(BatchNormalization)(conv_enc2)

    conv_enc3 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='Relu'))(conv_enc2_bn)
    conv_enc3_bn = TimeDistributed(BatchNormalization)(conv_enc3)

    # LSTM-Encoder

    convLSTM_enc1 = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='Relu',
                               return_state='True', initial_state=None)(conv_enc3_bn)

    convLSTM_enc2_layer = ConvLSTM2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='Relu',
                                     return_state='True', initial_state=None)
    encoder_outputs, state_h, state_c = convLSTM_enc2_layer(convLSTM_enc1)
    encoder_states = [state_h, state_c]

    # LSTM-Decoder
