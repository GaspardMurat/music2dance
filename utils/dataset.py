from abc import ABC

import h5py
import glob
import logging
import keras
import numpy as np


class DataGenerator(keras.utils.Sequence, ABC):
    '''
    Dataset class:
        __get_item__: return a batch of data.
                     X = sequence of STFT from i to i+sequence
                     Y = sequence of motion from i+step to i+step+sequence
    '''

    def __init__(self, folder, batch_size, sequence, stage, init_step, shuffle=True):
        self.n_channels = 1
        self.batch_size = batch_size
        self._inputs = ["input", "motion"]
        self.sequence = sequence
        self.steps = 1
        self.shuffle = shuffle
        print("############################################################################")
        print("\n")
        print('Searching in {} for files:'.format(folder))
        print("\n")
        print("############################################################################")
        logging.info('Searching in {} for files:'.format(folder))
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

    def get_example(self, i):
        iDB, iFL = self.idxs[i]
        data_labels = [None] * 3
        with h5py.File(self.list_file[iDB], 'r') as f:
            data_labels[0] = f[self._inputs[0]][iFL: iFL + self.sequence][None, :]
            if self.init_step == 0:
                data_labels[1] = np.zeros((1, 71), dtype=np.float32)  # TODO(nelson): to variable size
            else:
                data_labels[1] = f[self._inputs[1]][iFL: iFL + self.steps]
            data_labels[2] = f[self._inputs[1]][iFL + self.steps: iFL + self.steps + self.sequence][None, :]
        return data_labels

    def __getitem__(self, index):
        X_seq = np.empty((self.batch_size, self.sequence, self.n_channels, *self._dims[0]))
        context = np.empty((self.batch_size, *self._dims[1]))
        y = np.empty((self.batch_size, self.sequence, *self._dims[1]))
        for i in range(index, index + self.batch_size):
            example = self.get_example(i)
            t = i - index
            input = np.expand_dims(np.squeeze(example[0]), axis=1)
            ctxt = example[1][0]
            output = example[2][0]
            X_seq[t] = input
            context[t] = ctxt
            y[t] = output
        return [X_seq, context], y


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
