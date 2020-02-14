from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D, Conv2D, BatchNormalization, TimeDistributed, Conv2DTranspose
from keras.layers import ConvLSTM2DCell
from keras import optimizers

from .readout import ConvRNN2D_readout


def convLSTM2dModel(input_shape, output_shape, learning_rate):
    time_step = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    depth = input_shape[3]

    encoder_input = Input(shape=(time_step, height, width, depth), name='Encoder_input')

    # Conv-Encoder

    conv_enc1 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(6, 4), strides=(4, 1), padding='valid', activation='Relu'))(encoder_input)
    conv_enc1_bn = TimeDistributed(BatchNormalization)(conv_enc1)

    conv_enc2 = TimeDistributed(
        Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 1), padding='valid', activation='Relu'))(conv_enc1_bn)
    conv_enc2_bn = TimeDistributed(BatchNormalization)(conv_enc2)

    conv_enc3 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='Relu'))(conv_enc2_bn)
    conv_enc3_bn = TimeDistributed(BatchNormalization)(conv_enc3)

    # LSTM-Encoder

    convLSTM_enc1 = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='Relu',
                               return_state='False', initial_state=None, return_sequences=True)(conv_enc3_bn)

    convLSTM_enc2_layer = ConvLSTM2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='Relu',
                                     return_state='True', initial_state=None)
    encoder_outputs, state_h, state_c = convLSTM_enc2_layer(convLSTM_enc1)
    encoder_states_2 = [state_h, state_c]

    # LSTM-Decoder

    decoder_input = Input(shape=(height, width, depth), name='decoder_input')

    conv_dec1 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(6, 4), strides=(4, 1), padding='same', activation='Relu'))(decoder_input)
    conv_dec1_bn = TimeDistributed(BatchNormalization)(conv_dec1)

    conv_dec2 = TimeDistributed(
        Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 1), padding='same', activation='Relu'))(conv_dec1_bn)
    conv_dec2_bn = TimeDistributed(BatchNormalization)(conv_dec2)

    conv_dec3 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='Relu'))(conv_dec2_bn)
    conv_enc3_bn = TimeDistributed(BatchNormalization)(conv_dec3)

    # TODO: test integration convLSTM_dec1
    convLSTMCell1 = ConvLSTM2DCell(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='Relu')
    convLSTM_dec1 = ConvRNN2D_readout(convLSTMCell1, time_step, return_sequences=True, initial_state=encoder_states_2)(encoder_outputs)

    convLSTM_dec2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='Relu',
                               return_state='False', initial_state=None, return_sequences=True)(convLSTM_dec1)

    # Conv-Decoder

    conv_dec1 = TimeDistributed(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='Relu'))(convLSTM_dec2)
    conv_dec1_bn = TimeDistributed(BatchNormalization)(conv_dec1)

    conv_dec2 = TimeDistributed(
        Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 1), padding='valid', activation='Relu'))(
        conv_dec1_bn)
    conv_dec2_bn = TimeDistributed(BatchNormalization)(conv_dec2)

    conv_dec3 = TimeDistributed(
        Conv2DTranspose(filters=16, kernel_size=(6, 4), strides=(4, 1), padding='valid', activation='Relu'))(
        conv_dec2_bn)
    conv_dec3_bn = TimeDistributed(BatchNormalization)(conv_dec3)

    main_output = conv_dec3_bn

    # Model compile

    model = Model(inputs=[encoder_input, decoder_input], outputs=main_output)

    optimizer = optimizers.adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model



