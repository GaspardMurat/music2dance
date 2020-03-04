from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D, BatchNormalization, TimeDistributed, Conv2DTranspose
from keras.layers import ConvLSTM2D
from keras import optimizers


# TODO: to modify !

def convlstm(input_shape, output_shape, learning_rate):
    time_step = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    depth = input_shape[3]

    main_input = Input(shape=(time_step, height, width, depth), name='main_input')

    # Encoder
    conv_enc1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(6, 4), strides=(4, 1), padding='same', activation='relu'))(main_input)
    conv_enc1_bn = TimeDistributed(BatchNormalization(axis=-1))(conv_enc1)

    conv_enc2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 1), padding='same', activation='relu'))(conv_enc1_bn)
    conv_enc2_bn = TimeDistributed(BatchNormalization(axis=-1))(conv_enc2)

    conv_enc3 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))(conv_enc2_bn)
    conv_enc3_bn = TimeDistributed(BatchNormalization(axis=-1))(conv_enc3)

    convLSTM_enc1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                    activation='relu', return_sequences=True)(conv_enc3)
    convLSTM_enc2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu', return_sequences=True)(convLSTM_enc1)
    convLSTM_enc2_bn = BatchNormalization()(convLSTM_enc2)

    convLSTM_dec1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu', return_sequences=True)(convLSTM_enc2_bn)
    convLSTM_dec2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu', return_sequences=False)(convLSTM_dec1)
    convLSTM_dec2_bn = BatchNormalization()(convLSTM_dec2)

    conv_dec1 = Conv2D(filters=48, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(convLSTM_dec2_bn)
    conv_dec1_bn = BatchNormalization()(conv_dec1)

    conv_dec2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(
        conv_dec1_bn)
    conv_dec2_bn = BatchNormalization()(conv_dec2)

    conv_dec3 = Conv2D(filters=69, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(
        conv_dec2_bn)
    conv_dec3_bn = BatchNormalization()(conv_dec3)

    x = Flatten()(conv_dec3_bn)
    main_output = Dense(output_shape, name='main_output')(x)

    model = Model(inputs=main_input, outputs=main_output)

    optimizer = optimizers.adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model
