#!/usr/bin/env python3
import argparse
import os
import pickle
import logging
import glob
import h5py
import numpy as np

import sys

module_utils = os.getcwd().replace('bin', 'utils')
sys.path.append(module_utils)

from utils.audio_transform import load_audio
from utils.audio_transform import audio_augmontation
from utils.audio_transform import input_loader
from utils.audio_transform import audio_transform

from utils.motion_transform import motion_transform


def main():
    prefix = config[args.type]
    file_list = glob.glob(os.path.join(prefix, '*'))
    logging.info('Preparing dataset...')
    for i in range(len(file_list)):
        item = file_list[i]
        with h5py.File(item, 'r') as f:
            motion = np.array(f['motion'])
            pos = np.array(f['position'])
            start_pos = pos[0]
            end_pos = pos[1]
            soundPath = str(np.array(f['song_path']))[2:-1]

        # TODO: finish motion transform (add silence)
        motion = motion_transform(motion, config)
        motion = np.squeeze(motion
                            )
        samplingRate = config['sampling_rate']
        audio_wave, sr = load_audio(soundPath, samplingRate)
        for snr in config['snr']:
            # TODO: finish  data_augmontation (add silence)
            transformed_audio_wave = audio_augmontation(audio_wave, snr)
            audiodata, bool = input_loader(transformed_audio_wave, start_pos, end_pos, config)
            if not bool:
                print('discarded')
                os.remove(item)
            else:
                audiodata = audio_transform(audiodata, config)

                os.remove(item)

                for snr in config['snr']:
                    h5file = '{}f{:03d}snr{:03d}.h5'.format(os.path.join(prefix, args.type + '_'), i, snr)
                    with h5py.File(h5file, 'a') as f:
                        f.create_dataset('song_path', data=item)
                        f.create_dataset('motion', data=motion)
                        f.create_dataset('input', data=audiodata)
                        f.create_dataset('snr', data=snr)
                        f.create_dataset('position', data=pos)
                print('ok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', '-t', type=str,
                        help='train or test')
    parser.add_argument('--folder', '-f', type=str,
                        help='path to folders')

    args = parser.parse_args()

    # TODO : suppress this after test
    args.folder = os.getcwd().replace('bin', 'exp')
    args.type = 'train'

    with open(os.path.join(args.folder, 'configuration.pickle'), 'rb') as f:
        config = pickle.load(f)

    config['rng_pos'] = [-0.9, 0.9]
    config['rng_wav'] = [-0.9, 0.9]
    main()
    print(config)
