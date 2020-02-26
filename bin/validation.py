#!/usr/bin/env python3
import os
import argparse
import glob
import logging
import pickle
import h5py
import matplotlib.pyplot as plt

from keras.models import load_model

import sys

module_network = os.getcwd().replace('bin', 'network')
sys.path.append(module_network)
from network.readout import ConvRNN2D_readout


def main():
    model = load_model(args.model, custom_objects=True)

    validation_set = 'validation_set'
    if not os.path.exists(os.path.join(args.folder_out, validation_set)):
        os.makedirs(os.path.join(args.folder_out, validation_set))

    # TODO: transformed == True:
    file_list = glob.glob(os.path.join(args.folder_in, '*'))
    for file in file_list:
        with h5py.File(file, 'r') as f:
            x_data = f['input']
            predicts = model.predict(x_data)
            print('predicts shape: ', predicts.shape)
            quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', type=str,
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
    parser.add_argument('--batch', '-b', type=int,
                        help='Minibatch size', default=50)
    parser.add_argument('--sequence', '-q', type=int,
                        help='Training sequence', default=1)
    parser.add_argument('--sequence_out', '-p', type=int,
                        help='Training out sequence', default=1)
    parser.add_argument('--multiprocessing', '-mu', type=str,
                        help='use_multiprocessing', default=False)
    parser.add_argument('--workers', '-w', type=int,
                        help='nb of workers', default=1)

    args = parser.parse_args()

    with open(args.configuration, 'rb') as f:
        config = pickle.load(f)

    main()
