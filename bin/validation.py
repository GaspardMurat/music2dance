#!/usr/bin/env python3
import os
import argparse
import glob
import logging
import pickle
import h5py
import json
import codecs
import shutil
import numpy as np
import matplotlib.pyplot as plt

import sys

module_utils = os.getcwd().replace('bin', 'utils')
sys.path.append(module_utils)
from utils.motion_transform import reverse_motion_transform


def save(output, start_position, end_position, path):
    nb_data = output.shape[0]
    output = np.reshape(output, (nb_data, 23, 3))
    output = output.tolist()
    skeletons = {"length": nb_data, "skeletons": output}
    with open(path + '/skeletons.json', "w") as write_file:
        json.dump(skeletons, codecs.open(path + '/skeletons.json', 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True,
                  indent=4)

    start_position = int(start_position)
    end_position = int(end_position)
    config = {"start_position": start_position, "end_position": end_position}

    with open(path + '/config.json', "w") as write_file:
        json.dump(config, write_file)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def main():

    if mode == 0:

        module_network = os.getcwd().replace('bin', 'network')
        sys.path.append(module_network)
        from network.convlstm import convlstm

        module_utils = os.getcwd().replace('bin', 'utils')
        sys.path.append(module_utils)
        from utils.dataset import DataGenerator

        path = config['test']
        file_list = glob.glob(os.path.join(path, '*'))
        logging.info('number of h5 files: {}'.format(len(file_list)))
        train_dataset = DataGenerator(path, config['batch_size'], config['sequence'], 'test', config['init_step'], shuffle=True)
        batch_0 = train_dataset[270]
        input_encoder_shape = batch_0[0].shape[1:]
        output_shape = batch_0[1].shape[1]

        model = convlstm(input_encoder_shape, output_shape, config['base_lr'])
        model.load_weights(args.model_weights)

        validation_set = 'validation_set'
        prefix = os.path.join(args.folder_out, validation_set)
        utils = os.path.join(prefix, 'utils')
        if not os.path.exists(prefix):
            os.makedirs(prefix)
            os.makedirs(utils)
            copytree(src='./draw', dst=utils)

        def get_audio_sequence(i, index, file):
            iFL = index[i]
            with h5py.File(file, 'r') as f:
                data_labels = f['input'][iFL: iFL + config['sequence']]
            return data_labels

        # TODO: transformed == True:
        if args.transformed:
            STEPS = 1
            file_list = glob.glob(os.path.join(args.folder_in, '*'))
            for file in file_list:
                with h5py.File(file, 'r') as f:
                    soundPath = str(np.array(f['sound_path']))[2:-1]
                    position = f['position']
                    audio_data = f['input']
                    current_lenght = audio_data.shape[0]
                    if config['sequence'] + config['sequence_out'] >= current_lenght:
                        logging.error('The lenght of the sequence is larger thant the lenght of the file...')
                        raise ValueError('')
                    max_size = current_lenght - (config['sequence'] + STEPS)
                    index = [x for x in np.arange(max_size)]

                sequence_predictions = []
                for i in range(max_size):
                    x_labels = get_audio_sequence(i, index, file)
                    x_labels = np.expand_dims(x_labels, axis=0)
                    x_labels = np.expand_dims(x_labels, axis=-1)
                    prediction = model.predict(x_labels)
                    prediction = np.squeeze(prediction)
                    sequence_predictions.append(prediction)

                sequence_predictions = np.array(sequence_predictions)
                sequence_predictions = reverse_motion_transform(sequence_predictions, config)

                # This are entries for save.
                # Use previous 'pos = [start_position, end_position]' parameter.
                sound_name = soundPath.split('/')[-1]
                start_pos = config['sequence']
                end_pos = sequence_predictions.shape[0] + start_pos

                os.makedirs(os.path.join(prefix, 'DANCE_' + sound_name))
                save(sequence_predictions, start_pos, end_pos, path=os.path.join(prefix, 'DANCE_' + sound_name))

    elif mode == 1:

        module_network = os.getcwd().replace('bin', 'network')
        sys.path.append(module_network)
        from network.convLSTM2dMoldel2 import ConvLSTM2dModel

        module_utils = os.getcwd().replace('bin', 'utils')
        sys.path.append(module_utils)
        from utils.dataset2 import DataGenerator2

        path = config['test']
        file_list = glob.glob(os.path.join(path, '*'))
        logging.info('number of h5 files: {}'.format(len(file_list)))
        train_dataset = DataGenerator2(path, config['batch_size'], config['sequence'], config['sequence_out'],
                                       'test',
                                       config['init_step'], shuffle=True)
        batch_0 = train_dataset[0]
        input_encoder_shape = batch_0[0][0].shape[1:]
        input_decoder_shape = batch_0[0][0].shape[1:]
        output_shape = batch_0[1].shape[1:]

        model = ConvLSTM2dModel(input_encoder_shape, output_shape, config['base_lr'])
        model.load_weights(args.model_weights)

        validation_set = 'validation_set'
        prefix = os.path.join(args.folder_out, validation_set)
        utils = os.path.join(prefix, 'utils')
        if not os.path.exists(prefix):
            os.makedirs(prefix)
            os.makedirs(utils)
            copytree(src='./draw', dst=utils)

        def get_audio_sequence(i, index, file):
            iFL = index[i]
            data_labels = [None] * 2
            with h5py.File(file, 'r') as f:
                data_labels[0] = f['input'][iFL: iFL + config['sequence']]
                data_labels[1] = f['input'][iFL + config['sequence']]
            return data_labels

        # TODO: transformed == True:
        if args.transformed:
            STEPS = 1
            file_list = glob.glob(os.path.join(args.folder_in, '*'))
            for file in file_list:
                with h5py.File(file, 'r') as f:
                    soundPath = str(np.array(f['sound_path']))[2:-1]
                    position = f['position']
                    audio_data = f['input']
                    current_lenght = audio_data.shape[0]
                    if config['sequence'] + config['sequence_out'] >= current_lenght:
                        logging.error('The lenght of the sequence is larger thant the lenght of the file...')
                        raise ValueError('')
                    max_size = current_lenght - (config['sequence'] + config['sequence_out'] + STEPS)
                    index = [x for x in np.arange(max_size)]

                sequence_predictions = []
                for i in range(max_size):
                    x_labels = get_audio_sequence(i, index, file)
                    x_labels[0] = np.expand_dims(x_labels[0], axis=0)
                    x_labels[0] = np.expand_dims(x_labels[0], axis=-1)
                    x_labels[1] = np.expand_dims(x_labels[1], axis=0)
                    x_labels[1] = np.expand_dims(x_labels[1], axis=-1)
                    prediction = model.predict(x_labels)
                    prediction = np.squeeze(prediction)
                    sequence_predictions.append(prediction[0])

                sequence_predictions = np.array(sequence_predictions)
                sequence_predictions = reverse_motion_transform(sequence_predictions, config)

                # This are entries for save.
                # Use previous 'pos = [start_position, end_position]' parameter.
                sound_name = soundPath.split('/')[-1]
                start_pos = config['sequence']
                end_pos = sequence_predictions.shape[0] + config['sequence_out']

                os.makedirs(os.path.join(prefix, 'DANCE_' + sound_name))
                save(sequence_predictions, start_pos, end_pos, path=os.path.join(prefix, 'DANCE_' + sound_name))


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
    parser.add_argument('--mode', '-md', type=int,
                        help='model and dataset', default=1)

    args = parser.parse_args()
    mode = args.mode

    with open(args.configuration, 'rb') as f:
        config = pickle.load(f)

    print(config)

    main()
