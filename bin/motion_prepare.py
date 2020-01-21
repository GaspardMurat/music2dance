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

from motion_loader import output_loader

def calculate_minmax(fileslist):
    logging.info('MinMax file not found...')
    logging.info('Creating MinMax file...')
    if len(fileslist) < 1:
        logging.error('No files were found in the folder ...')
        raise ValueError()
    for item in fileslist:
        with h5py.File(item, 'r') as f:
            motion = np.array(f['motion'])
            tmp_minmax = np.array(f['MinMax'])
        if 'pos_minmax' not in locals():
            pos_minmax = np.zeros((tmp_minmax.shape), dtype=np.float32)
            pos_minmax[0, :] = tmp_minmax[0, :]
            pos_minmax[1, :] = tmp_minmax[1, :]
        pos_minmax[0, :] = np.amin([tmp_minmax[0, :], pos_minmax[0, :]], axis=0)  # minimum
        pos_minmax[1, :] = np.amax([tmp_minmax[1, :], pos_minmax[1, :]], axis=0)  # maximum
    with h5py.File(configuration['file_pos_minmax'], 'a') as f:
        f.create_dataset('minmax', data=pos_minmax.T)
    return

def main():

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
        motion , start_position, end_position, pos_min, pos_max = output_loader(path)
        h5file = '{}f{:03d}.h5'.format(os.path.join(prefix, args.type), i)
        list_path = np.string_(path)
        with h5py.File(h5file, 'a') as f:
            f.create_dataset('song_path', data=list_path)
            f.create_dataset('motion', data=motion)
            f.create_dataset('position', data=[start_position, end_position])
            f.create_dataset('MinMax', data=[pos_min, pos_max])

    configuration['file_pos_minmax'] = os.path.join(preprefix, 'pos_minmax.h5')

    if not os.path.exists(configuration['file_pos_minmax']):
        file_list = glob.glob(os.path.join(configuration[args.type], '*'))
        calculate_minmax(file_list)

    with open(os.path.join(args.save, "configuration.pickle"), "wb") as f:
        pickle.dump(configuration, f)

    with h5py.File(configuration['file_pos_minmax'], 'r') as f:
        pos_min = f['minmax'][0, :][None, :]
        pos_max = f['minmax'][1, :][None, :]

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

    # TODO : suppress this after test
    args.save = os.getcwd().replace('bin', 'exp')
    args.folder = os.getcwd().replace('bin', 'dataset_master')
    args.type = 'train'


    configuration = {'step': 0, 'type': args.type, 'fps': args.fps, 'sampling_rate': args.sampling,
                     'hop_length': args.hop_length, 'window_length': args.wlen, 'snr': args.snr}

    main()
