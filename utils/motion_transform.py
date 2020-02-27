import h5py
import numpy as np

import os
import pickle


def motion_transform(position_data, config):
    with h5py.File(config['file_pos_minmax'], 'r') as f:
        minmax = np.array(f['minmax']).T
        pos_min = minmax[0, :][None, :]
        pos_max = minmax[1, :][None, :]

    div = pos_max - pos_min
    div[div == 0] = 1
    config['slope_pos'] = (config['rng_pos'][1] - config['rng_pos'][0]) / div
    config['intersec_pos'] = config['rng_pos'][1] - config['slope_pos'] * pos_max
    init_trans = np.mean(position_data[0:25, :], axis=0)
    position_data[:, :] -= init_trans
    position_data = position_data * config['slope_pos'] + config['intersec_pos']

    return position_data, init_trans


def motion_untransform(position_data, config):
    with h5py.File(config['file_pos_minmax'], 'r') as f:
        minmax = np.array(f['minmax']).T
        pos_min = minmax[0, :][None, :]
        pos_max = minmax[1, :][None, :]


if __name__ == '__main__':
    prefix = os.getcwd().replace('utils', 'exp/data/train/trainf000.h5')
    with h5py.File(prefix, 'r') as f:
        motion = np.array(f['motion'])
    prefix = os.getcwd().replace('utils', 'exp')
    with open(os.path.join(prefix, 'configuration.pickle'), 'rb') as f:
        config = pickle.load(f)
    config['rng_pos'] = [-0.9, 0.9]
    config['rng_wav'] = [-0.9, 0.9]
    print(motion.shape)
    quit()
    motion_transform(motion, config)
