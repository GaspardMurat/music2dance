from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D, MaxPooling3D, BatchNormalization
from keras import optimizers

# TODO: to modify !

def convlstm(input_shape, output_shape, learning_rate):
    time_step = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    depth = input_shape[3]

    main_input = Input(shape=(time_step, height, width, depth), name='main_input')

    # Encoder
    x = ConvLSTM2D(16, (32, 2), padding='same', return_sequences=True)(main_input)
    x = ConvLSTM2D(32, (32, 2), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(64, (32, 2), padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(69, (32, 2), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((1, 2, 2))(x)
    x = Flatten()(x)
    main_output = Dense(output_shape, name='main_output')(x)

    model = Model(inputs=main_input, outputs=main_output)

    optimizer = optimizers.adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model
