from abc import ABC

import h5py
import glob
import logging
import keras
import numpy as np


class DataGenerator2(keras.utils.Sequence, ABC):
    '''
        Dataset class:
            __get_item__: return a batch of data.
                         X_encoder = sequence of STFT from i to i+sequence
                         X_decoder = X_encoder[-1]
                         Y = sequence of motion from i+step+sequence to i+step+sequence
        '''

    def __init__(self, folder, batch_size, sequence, sequence_out, stage, init_step, shuffle=True):
        self.n_channels = 1
        self.batch_size = batch_size
        self._inputs = ["input", "motion"]
        self.sequence = sequence
        self.sequence_out = sequence_out
        self.steps = 1
        self.shuffle = shuffle
        _dims = None
        _types = None
        logging.info('Searching in {} for files:'.format(folder))
        self.list_file = glob.glob('{}/{}_*'.format(folder, stage))
        print('Searching in {} for files:'.format(folder))
        index = []
        for i in range(len(self.list_file)):
            with h5py.File(self.list_file[i], 'r') as f:
                current_lenght = f[self._inputs[0]].shape[0]
                if self.sequence + self.sequence_out >= current_lenght:
                    logging.error('The lenght of the sequence is larger thant the lenght of the file...')
                    raise ValueError('')
                max_size = current_lenght - (self.sequence + self.sequence_out + self.steps)
                if _dims is None:
                    _dims = [None] * len(self._inputs)
                    _types = [None] * len(self._inputs)
                    for j in range(len(self._inputs)):
                        testfile = f[self._inputs[j]]
                        _dim = testfile[0].shape
                        _dims[j] = _dim if len(_dim) != 0 else []
                        _types[j] = testfile.dtype
                        logging.info('  data label: {} \t dim: {} \t dtype: {}'.format(self._inputs[j], list(_dims[j]),
                                                                                       _types[i]))
            _index = [[i, x] for x in np.arange(max_size)]
            index += _index
        try:
            self._dims = _dims
        except Exception as e:
            logging.error('Cannot assign dimensions, data not found...')
            raise TypeError(e)
        self._type = _types
        self.idxs = index
        self.init_step = init_step
        self.on_epoch_end()
        logging.info('sequence: {}'.format(self.sequence))
        logging.info('Total of {} files...'.format(len(self.idxs)))

    def __len__(self):
        return int(np.floor(len(self.idxs) / self.batch_size))

    def get_example(self, i):
        iDB, iFL = self.idxs[i]
        data_labels = [None] * 3
        with h5py.File(self.list_file[iDB], 'r') as f:
            data_labels[0] = f[self._inputs[0]][iFL: iFL + self.sequence]
            data_labels[1] = f[self._inputs[0]][iFL + self.sequence: iFL + self.sequence + self.sequence_out]
            data_labels[2] = f[self._inputs[1]][
                             iFL + self.steps + self.sequence: iFL + self.steps + self.sequence + self.sequence_out]
        return data_labels

    def __getitem__(self, index):
        X_encoder = np.empty((self.batch_size, self.sequence, *self._dims[0], self.n_channels))
        X_decoder = np.empty((self.batch_size, *self._dims[0], self.n_channels))
        y = np.empty((self.batch_size, self.sequence_out, *self._dims[1]))
        for i in range(index, index + self.batch_size):
            example = self.get_example(i)
            t = i - index
            input_encoder = np.expand_dims(example[0], axis=3)
            output_decoder = example[2]
            X_encoder[t] = input_encoder
            X_decoder[t] = input_encoder[-1]
            y[t] = output_decoder
        return [X_encoder, X_decoder], y
        # return [X_encoder, X_encoder], y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)



