from abc import ABC

import h5py
import glob
import logging
import keras
import numpy as np


class DataGenerator(keras.utils.Sequence, ABC):

    def __init__(self, folder, batch_size, sequence, stage, init_step, shuffle=True):
        self.n_channels = 1
        self.batch_size = batch_size
        self._inputs = ["input", "motion"]
        self.sequence = sequence
        self.steps = 1
        self.shuffle = shuffle
        logging.info('Searching in {}/{} for files:'.format(folder, stage))
        self.list_file = glob.glob('{}/{}_*'.format(folder, stage))
        index = []
        for i in range(len(self.list_file)):
            with h5py.File(self.list_file[i], 'r') as f:
                current_lenght = f[self._inputs[0]].shape[0]
                if self.sequence >= current_lenght:
                    logging.error('The lenght of the sequence is larger thant the lenght of the file...')
                    raise ValueError('')
                max_size = current_lenght - (self.sequence + self.steps)
                if not '_dims' in locals():
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

    # TODO: change get_example, get_item for output sequence

    def get_example(self, i):
        iDB, iFL = self.idxs[i]
        data_labels = [None] * 3
        with h5py.File(self.list_file[iDB], 'r') as f:
            data_labels[0] = f[self._inputs[0]][iFL: iFL + self.sequence][None, :]
            if self.init_step == 0:
                data_labels[1] = np.zeros((1, self._dims[1][0]), dtype=np.float32)  # TODO(nelson): to variable size
            else:
                #data_labels[1] = f[self._inputs[1]][iFL: iFL + self.steps]
                data_labels[1] = f[self._inputs[1]][iFL + self.sequence: iFL + self.sequence + self.steps]

            data_labels[2] = f[self._inputs[1]][iFL + self.steps: iFL + self.steps + self.sequence][None, :]
        return data_labels

    def __getitem__(self, index):
        '''
        X = np.empty((self.batch_size, self.sequence, *self._dims[0], self.n_channels))
        y = np.empty((self.batch_size, self.sequence, *self._dims[1]))

        for i in range(index, index + self.batch_size):
            example = self.get_example(i)
            t = i - index
            input = np.expand_dims(np.squeeze(example[0]), axis=3)
            X[t] = input
            #y[t] = np.squeeze(example[1])
            y[t] = np.squeeze(example[1])
        '''
        X = np.empty((self.batch_size, self.sequence, *self._dims[0], self.n_channels))
        y = np.empty((self.batch_size, *self._dims[1]))
        for i in range(index, index + self.batch_size):
            example = self.get_example(i)
            t = i - index
            input = np.expand_dims(np.squeeze(example[0]), axis=3)
            X[t] = input
            y[t] = np.squeeze(example[1])
        return X, y


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
