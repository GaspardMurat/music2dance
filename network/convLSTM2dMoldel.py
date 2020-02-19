from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D, BatchNormalization, TimeDistributed, Conv2DTranspose
from keras.layers import ConvLSTM2DCell
# from keras.layers import ConvLSTM2D
from keras import optimizers

from .readout import ConvRNN2D_readout
from .convolutional_recurrent import ConvLSTM2D


def ConvLSTM2dModel(input_shape, output_shape, learning_rate):
    time_step = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    depth = input_shape[3]
    time_step_out = output_shape[0]
    output_dim = output_shape[1]

    encoder_input = Input(shape=(None, height, width, depth))

    # Conv-Encoder

    conv_enc1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(6, 4), strides=(4, 1), padding='same', activation='relu'))(encoder_input)
    conv_enc1_bn = TimeDistributed(BatchNormalization(axis=-1))(conv_enc1)

    conv_enc2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 1), padding='same', activation='relu'))(conv_enc1_bn)
    conv_enc2_bn = TimeDistributed(BatchNormalization(axis=-1))(conv_enc2)
    conv_enc3 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))(conv_enc2_bn)
    conv_enc3_bn = TimeDistributed(BatchNormalization(axis=-1))(conv_enc3)

    # LSTM-Encoder
    convLSTM_enc1_cell = ConvLSTM2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                    return_state=True, return_sequences=False)
    _, state_h, state_c = convLSTM_enc1_cell(conv_enc3_bn)
    encoder_states = [state_h, state_c]

    # LSTM-Decoder

    decoder_input = Input(shape=(height, width, depth))

    conv_mono_dec1 = Conv2D(filters=8, kernel_size=(6, 4), strides=(4, 1), padding='same', activation='relu')(
        decoder_input)
    conv_mono_dec1_bn = BatchNormalization(axis=-1)(conv_mono_dec1)

    conv_mono_dec2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 1), padding='same', activation='relu')(
        conv_mono_dec1_bn)
    conv__mono_dec2_bn = BatchNormalization(axis=-1)(conv_mono_dec2)

    conv_mono_dec3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(
        conv__mono_dec2_bn)
    conv_mono_enc3_bn = BatchNormalization(axis=-1)(conv_mono_dec3)

    convLSTMCell2 = ConvLSTM2DCell(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
    convLSTM_dec2 = ConvRNN2D_readout(convLSTMCell2, time_step_out, return_sequences=True, initial_state=encoder_states)(conv_mono_enc3_bn)

    '''output_shape = (None, 8, 4, 16)
    convLSTM_dec2.compute_output_shape(output_shape)
    quit()'''

    # Conv-Decoder

    conv_dec1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))(
        convLSTM_dec2)
    conv_dec1_bn = TimeDistributed(BatchNormalization(axis=-1))(conv_dec1)

    conv_dec2 = TimeDistributed(
        Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 1), padding='same', activation='relu'))(
        conv_dec1_bn)
    conv_dec2_bn = TimeDistributed(BatchNormalization(axis=-1))(conv_dec2)

    conv_dec3 = TimeDistributed(
        Conv2DTranspose(filters=16, kernel_size=(6, 4), strides=(4, 1), padding='same', activation='relu'))(
        conv_dec2_bn)
    conv_dec3_bn = TimeDistributed(BatchNormalization(axis=-1))(conv_dec3)

    flat = TimeDistributed(Flatten())(conv_dec3_bn)
    decoder_outputs = TimeDistributed(Dense(output_dim))(flat)

    # Model compile

    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)

    optimizer = optimizers.adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model
