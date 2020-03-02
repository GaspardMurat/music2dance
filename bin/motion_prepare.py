#!/usr/bin/env python3
import argparse
import os
import logging
import glob
import h5py
import pickle
import numpy as np

import sys

module_utils = os.getcwd().replace('bin', 'utils')
sys.path.append(module_utils)

from utils.motion_loader import output_loader


def calculate_minmax(fileslist):
    logging.info('MinMax file not found...')
    logging.info('Creating MinMax file...')
    if len(fileslist) < 1:
        logging.error('No files were found in the folder ...')
        raise ValueError()
    init = 0
    for item in fileslist:
        with h5py.File(item, 'r') as f:
            motion = np.array(f['motion'])
            # TODO
            init_trans = np.mean(motion[0:26, :], axis=0)
            motion -= init_trans
            tmp_minmax = np.concatenate((np.amin(motion, axis=0)[:, None],
                                         np.amax(motion, axis=0)[:, None]), axis=1)
            # TODO
            tmp_minmax = tmp_minmax.T
        if 'pos_minmax' not in locals():
            pos_minmax = np.zeros((tmp_minmax.shape), dtype=np.float32)
            pos_minmax[0, :] = tmp_minmax[0, :]
            pos_minmax[1, :] = tmp_minmax[1, :]
        pos_minmax[0, :] = np.amin([tmp_minmax[0, :], pos_minmax[0, :]], axis=0)  # minimum
        pos_minmax[1, :] = np.amax([tmp_minmax[1, :], pos_minmax[1, :]], axis=0)  # maximum
        init += init_trans
    init /= len(fileslist)
    with h5py.File(configuration['file_pos_minmax'], 'a') as f:
        f.create_dataset('minmax', data=pos_minmax.T)
        f.create_dataset('init', data=init)


def main():
    if args.type == 'train':
        print('====== Creating train dataset ======')
        preprefix = os.path.join(args.save, 'data')
        prefix = os.path.join(preprefix, args.type)
        if not os.path.exists(args.save):
            os.makedirs(args.save)
            os.makedirs(preprefix)
            os.makedirs(prefix)
        join = os.path.join(args.folder, 'DANCE_*')
        folders = glob.glob(join)
        configuration[args.type] = prefix
        for i in range(len(folders)):
            path = folders[i]
            print('Using ', path)
            motion, start_position, end_position= output_loader(path)
            h5file = '{}f{:03d}.h5'.format(os.path.join(prefix, args.type), i)
            list_path = np.string_(path)
            with h5py.File(h5file, 'a') as f:
                f.create_dataset('song_path', data=list_path)
                f.create_dataset('motion', data=motion)
                f.create_dataset('position', data=[start_position, end_position])
            print('Making ', h5file)

        configuration['file_pos_minmax'] = os.path.join(preprefix, 'pos_minmax.h5')

        if not os.path.exists(configuration['file_pos_minmax']):
            file_list = glob.glob(os.path.join(configuration[args.type], '*'))
            calculate_minmax(file_list)

        with open(os.path.join(args.save, "configuration.pickle"), "wb") as f:
            pickle.dump(configuration, f)

        with h5py.File(configuration['file_pos_minmax'], 'r') as f:
            pos_min = f['minmax'][0, :][None, :]
            pos_max = f['minmax'][1, :][None, :]

    elif args.type == 'test':
        print('====== Creating test dataset ======')
        preprefix = os.path.join(args.save, 'data')
        prefix = os.path.join(preprefix, args.type)
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        if not os.path.exists(preprefix):
            os.makedirs(preprefix)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        join = os.path.join(args.folder, 'test', 'DANCE_*')
        folders = glob.glob(join)
        if os.path.exists(os.path.join(args.save, 'configuration.pickle')):
            with open(os.path.join(args.save, 'configuration.pickle'), 'rb') as f:
                config = pickle.load(f)
        else:
            logging.warning('Run stage 0 for train before running stage 0 for test.')

        config[args.type] = prefix
        for i in range(len(folders)):
            path = folders[i]
            print('Using ', path)
            motion, start_position, end_position = output_loader(path)
            h5file = '{}f{:03d}.h5'.format(os.path.join(prefix, args.type), i)
            list_path = np.string_(path)
            with h5py.File(h5file, 'a') as f:
                f.create_dataset('song_path', data=list_path)
                f.create_dataset('motion', data=motion)
                f.create_dataset('position', data=[start_position, end_position])
            print('Making ', h5file)
            with open(os.path.join(args.save, "configuration.pickle"), "wb") as f:
                pickle.dump(config, f)
    else:
        logging.warning('Only train or test are acceptable arguments.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', '-f', type=int,
                        help='fps of the motion data', default=25)
    parser.add_argument('--sampling', '-sr', type=int,
                        help='sampling rate for acoustic data', default=44100)
    parser.add_argument('--hop_length', '-hl', type=int,
                        help='hop lenght', default=1764)
    parser.add_argument('--wlen', '-w', type=int,
                        help='STFT Window size', default=256)
    parser.add_argument('--snr', '-r', nargs='+', type=int,
                        help='List of SNR', default=[0])
    parser.add_argument('--folder', '-d', type=str,
                        help='Specify dataset master folder')
    parser.add_argument('--save', '-o', type=str,
                        help='Specify out folder')
    parser.add_argument('--type', '-t', type=str,
                        help='train or test')

    args = parser.parse_args()

    configuration = {'step': 0, 'fps': args.fps, 'sampling_rate': args.sampling,
                     'hop_length': args.hop_length, 'window_length': args.wlen, 'snr': args.snr}

    main()
