import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D, BatchNormalization, TimeDistributed
from keras.layers import LSTM

import keras.backend as K


class AudioFeat(Model):

    def __init__(self, dim, name='Audiofeat', **kwargs):
        super(AudioFeat, self).__init__(name='Audiofeat', **kwargs)
        self.conv1 = Conv2D(filters=12, kernel_size=(33, 3), strides=(1, 1), padding='valid',
                            data_format='channels_first',
                            activation='elu')
        self.conv1bn = BatchNormalization()
        self.conv2 = Conv2D(filters=24, kernel_size=(33, 3), strides=(1, 1), padding='valid',
                            data_format='channels_first',
                            activation='elu')
        self.conv2bn = BatchNormalization()
        self.conv3 = Conv2D(filters=48, kernel_size=(33, 2), strides=(1, 1), padding='valid',
                            data_format='channels_first',
                            activation='elu')
        self.conv3bn = BatchNormalization()
        self.conv4 = Conv2D(filters=dim, kernel_size=(32, 2), strides=(1, 1), padding='valid',
                            data_format='channels_first',
                            activation='elu')
        self.conv4bn = BatchNormalization()

    def call(self, inputs, mask=None):
        feats1 = self.conv1bn(self.conv1(inputs))
        feats2 = self.conv2bn(self.conv2(feats1))
        feats3 = self.conv3bn(self.conv3(feats2))
        feats = self.conv4bn(self.conv4(feats3))
        feats = K.squeeze(feats, axis=-1)
        return feats


class Music2dance(keras.Model):

    def __init__(self, input_shape, skeleton_dim, feat_dim, units,
                 batchsize,
                 name='Music2dance',
                 **kwargs):
        super(Music2dance, self).__init__(name=name, **kwargs)
        self.batchsize = batchsize
        self.time_step = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.depth = input_shape[3]
        self.skeleton_dim = skeleton_dim

        self.conv1 = Conv2D(filters=12, kernel_size=(33, 3), strides=(1, 1), padding='valid',
                            data_format='channels_first',
                            activation='elu')
        self.conv1_bn = BatchNormalization()
        self.conv2 = Conv2D(filters=24, kernel_size=(33, 3), strides=(1, 1), padding='valid',
                            data_format='channels_first',
                            activation='elu')
        self.conv2bn = BatchNormalization()
        self.conv3 = Conv2D(filters=48, kernel_size=(33, 2), strides=(1, 1), padding='valid',
                            data_format='channels_first',
                            activation='elu')
        self.conv3bn = BatchNormalization()
        self.conv4 = Conv2D(filters=feat_dim, kernel_size=(32, 2), strides=(1, 1), padding='valid',
                            data_format='channels_first',
                            activation='elu')
        self.conv4bn = BatchNormalization()

        self.enc_lstm1 = LSTM(units, activation='elu', return_sequences=True, return_state=False)
        self.enc_lstm2 = LSTM(units, activation='elu', return_sequences=True, return_state=False)
        self.enc_lstm3 = LSTM(units, activation='elu', return_sequences=False, return_state=False)
        self.fc0 = Dense(feat_dim, activation='elu')
        self.dec_lstm1 = LSTM(units, activation='elu', return_sequences=True, return_state=False)
        self.dec_lstm2 = LSTM(units, activation='elu', return_sequences=True, return_state=False)
        self.dec_lstm3 = LSTM(units, activation='elu', return_sequences=False, return_state=False)
        self.signal_out = Dense(skeleton_dim, activation='elu')
        self.out = TimeDistributed(Dense(skeleton_dim))

        self.inputs1 = Input(
            batch_shape=(self.batchsize, self.time_step, self.height, self.width, self.depth))
        self.inputs2 = Input(batch_shape=(self.batchsize, self.skeleton_dim))
        #self.outputs = self.compute_outputs([self.inputs1, self.inputs2])

    def call(self, inputs):
        audio_input = inputs[0]
        context = inputs[1]
        batchsize, sequence = audio_input.shape[0:2]
        outputs_list = []
        for i in range(sequence):
            h = self.audiofeat(audio_input[:, i])
            y = self.forward(context, h)
            context = y
            outputs_list.append(context)
        outputs = K.stack(outputs_list, axis=1) # Because K.stack in not a keras layer, call does not return a trainable outputs...
        outputs = self.out(outputs)
        return outputs

    def forward(self, h1, h, eval=False):
        # TODO: change dims of h (before and after concat).
        enc1 = self.enc_lstm1(h)
        enc2 = self.enc_lstm2(enc1)
        enc3 = self.enc_lstm3(enc2)
        _h = self.fc0(enc3)
        h = K.concatenate([h1, _h])
        h = K.expand_dims(h, axis=-1)
        dec1 = self.dec_lstm1(h)
        dec2 = self.dec_lstm2(dec1)
        dec3 = self.dec_lstm3(dec2)
        h = self.signal_out(dec3)
        if eval:
            return _h, h
        return h

    def audiofeat(self, inputs):
        feats1 = self.conv1_bn(self.conv1(inputs))
        feats2 = self.conv2bn(self.conv2(feats1))
        feats3 = self.conv3bn(self.conv3(feats2))
        feats = self.conv4bn(self.conv4(feats3))
        feats = K.squeeze(feats, axis=-1)
        return feats

    def compute_outputs(self, inputs):
        outputs = self.call(inputs)
        return outputs
