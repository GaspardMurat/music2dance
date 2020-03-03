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
from utils.audio_transform import audio_silence

from utils.motion_transform import motion_transform
from utils.motion_transform import motion_silence


def main():
    prefix = config[args.type]
    file_list = glob.glob(os.path.join(prefix, '*'))
    logging.info('Preparing dataset...')
    for i in range(len(file_list)):
        item = file_list[i]
        print('treatment of {}'.format(item))
        with h5py.File(item, 'r') as f:
            motion = np.array(f['motion'])
            pos = np.array(f['position'])
            start_pos = pos[0]
            end_pos = pos[1]
            soundPath = str(np.array(f['song_path']))[2:-1]

        motion, motion_mean = motion_transform(motion, config)
        motion = np.squeeze(motion)

        samplingRate = config['sampling_rate']
        audio_wave, sr = load_audio(soundPath, samplingRate)
        _, bool = input_loader(audio_wave, start_pos, end_pos, config)
        if not bool:
            print('discarded')
            if os.path.exists(item):
                os.remove(item)
            else:
                pass
        else:
            for snr in config['snr']:
                transformed_audio_wave = audio_augmontation(audio_wave, snr)
                audiodata, _ = input_loader(transformed_audio_wave, start_pos, end_pos, config)

                silence_audio = audio_silence(config)
                audiodata = np.concatenate((silence_audio, audiodata, silence_audio))
                audiodata = audio_transform(audiodata, config)

                silence_pos_0 = motion_silence(motion[0], silence_audio.shape[0])
                silence_pos_last = motion_silence(motion[-1], silence_audio.shape[0])
                motion = np.concatenate((silence_pos_0, motion, silence_pos_last))

                h5file = '{}f{:03d}snr{:03d}.h5'.format(os.path.join(prefix, args.type + '_'), i, snr)
                with h5py.File(h5file, 'a') as f:

                    f.create_dataset('sound_path', data=soundPath)
                    f.create_dataset('motion', data=motion)
                    f.create_dataset('input', data=audiodata)
                    f.create_dataset('snr', data=snr)
                    f.create_dataset('position', data=pos)

            if os.path.exists(item):
                os.remove(item)
            else:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', '-t', type=str,
                        help='train or test')
    parser.add_argument('--folder', '-f', type=str,
                        help='path to folders')
    parser.add_argument('--silence', '-s', type=int,
                        help='snr value', default=1)
    parser.add_argument('--normalisation', '-n', type=str,
                        help='motion normalisation', default='none')

    args = parser.parse_args()

    with open(os.path.join(args.folder, 'configuration.pickle'), 'rb') as f:
        config = pickle.load(f)

    config['silence'] = args.silence
    config['normalisation'] = args.normalisation
    config['rng_pos'] = [-0.9, 0.9]
    config['rng_wav'] = [-0.9, 0.9]

    main()

    with open(os.path.join(args.folder, 'configuration.pickle'), "wb") as f:
        pickle.dump(config, f)
    print(config)
