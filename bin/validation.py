#!/usr/bin/env python3
import os
import argparse
import glob
import logging
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt

import sys

module_network = os.getcwd().replace('bin', 'network')
sys.path.append(module_network)
from network.convLSTM2dMoldel import ConvLSTM2dModel

module_utils = os.getcwd().replace('bin', 'utils')
sys.path.append(module_utils)
from utils.dataset2 import DataGenerator2


def get_motion_sequence(i, index, file):
    iFL = index[i]
    data_labels = [None] * 2
    with h5py.File(file, 'r') as f:
        data_labels[0] = f['input'][iFL: iFL + config['sequence']]
        data_labels[1] = f['input'][iFL + config['sequence']]
    return data_labels


def main():
    path = config['test']
    file_list = glob.glob(os.path.join(path, '*'))
    logging.info('number of h5 files: {}'.format(len(file_list)))

    train_dataset = DataGenerator2(path, config['batch_size'],  config['sequence'], config['sequence_out'], 'test',
                                   config['init_step'], shuffle=True)
    batch_0 = train_dataset[270]
    input_encoder_shape = batch_0[0][0].shape[1:]
    input_decoder_shape = batch_0[0][0].shape[1:]
    output_shape = batch_0[1].shape[1:]

    model = ConvLSTM2dModel(input_encoder_shape, output_shape, config['base_lr'])
    model.load_weights(args.model_weights)

    validation_set = 'validation_set'
    if not os.path.exists(os.path.join(args.folder_out, validation_set)):
        os.makedirs(os.path.join(args.folder_out, validation_set))

    # TODO: transformed == True:
    if args.transformed:
        STEPS = 1
        file_list = glob.glob(os.path.join(args.folder_in, '*'))
        for file in file_list:
            with h5py.File(file, 'r') as f:
                motion_data = f['input']
                current_lenght = motion_data.shape[0]
                if config['sequence'] + config['sequence_out'] >= current_lenght:
                    logging.error('The lenght of the sequence is larger thant the lenght of the file...')
                    raise ValueError('')
                max_size = current_lenght - (config['sequence'] + config['sequence_out'] + STEPS)
                index = [x for x in np.arange(max_size)]

            sequence_predictions = []
            for i in range(max_size):
                x_labels = get_motion_sequence(i, index, file)
                x_labels[0] = np.expand_dims(x_labels[0], axis=0)
                x_labels[0] = np.expand_dims(x_labels[0], axis=-1)
                x_labels[1] = np.expand_dims(x_labels[1], axis=0)
                x_labels[1] = np.expand_dims(x_labels[1], axis=-1)
                prediction = model.predict(x_labels)
                sequence_predictions.append(prediction)

            # TODO: about the concatenation of outputs...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_weights', '-m', type=str,
                        help='path to model')
    parser.add_argument('--folder_in', '-i', type=str,
                        help='path to music files')
    parser.add_argument('--folder_out', '-o', type=str,
                        help='path to out')
    parser.add_argument('--transformed', '-t', type=str,
                        help='music h5 files', default=True)
    parser.add_argument('--final_json', '-f', type=str,
                        help='music h5 files', default=True)
    parser.add_argument('--snr', '-r', nargs='+', type=int,
                        help='List of SNR', default=[0])
    parser.add_argument('--configuration', '-c', type=str,
                        help='path to configuration file')
    parser.add_argument('--verbose', '-v', type=int,
                        help='verbose', default=1)
    parser.add_argument('--multiprocessing', '-mu', type=str,
                        help='use_multiprocessing', default=False)
    parser.add_argument('--workers', '-w', type=int,
                        help='nb of workers', default=1)

    args = parser.parse_args()

    with open(args.configuration, 'rb') as f:
        config = pickle.load(f)

    main()
