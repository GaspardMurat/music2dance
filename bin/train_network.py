#!/usr/bin/env python3
import os
import argparse
import glob
import logging
import pickle
import matplotlib.pyplot as plt

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, LearningRateScheduler

import sys

module_network = os.getcwd().replace('bin', 'network')
sys.path.append(module_network)
from network.convlstm import convlstm

module_utils = os.getcwd().replace('bin', 'utils')
sys.path.append(module_utils)
from utils.dataset import DataGenerator


def main():
    print('enter main')
    path = config[args.type]
    file_list = glob.glob(os.path.join(path, '*'))
    logging.info('number of h5 files: {}'.format(len(file_list)))

    dataset = DataGenerator(path, args.batch, args.sequence, args.type, args.init_step, shuffle=True)
    batch_0 = dataset[0]
    input_shape = batch_0[0].shape[1:]
    output_shape = batch_0[1].shape[2]  # output_shape = batch_0[1].shape[1:]
    print('input shape: ', input_shape)
    print('output shape: ', output_shape)

    folder_models = os.path.join(args.out, 'models')

    if not os.path.exists(folder_models):
        os.makedirs(folder_models)

    model = convlstm(input_shape, output_shape, args.base_lr)

    model_saver = ModelCheckpoint(filepath=os.path.join(folder_models, 'model.ckpt.{epoch:04d}.hdf5'),
                                  verbose=1,
                                  save_best_only=False,
                                  period=10)

    def lr_scheduler(epoch, lr):
        decay_rate = 0.90
        decay_step = 20
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr

    callbacks_list = [model_saver,
                      TerminateOnNaN(),
                      LearningRateScheduler(lr_scheduler, verbose=1)]

    history = model.fit_generator(dataset,
                                  epochs=args.epochs,
                                  use_multiprocessing=args.multiprocessing,
                                  workers=args.workers,
                                  callbacks=callbacks_list)

    plot_model(model, show_layer_names=True, show_shapes=True, to_file=os.path.join(args.out, 'model.png'))

    def plot_loss(hist, save):
        # Plot training & validation loss values
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(save, 'loss_values.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', '-t', type=str,
                        help='train or test')
    parser.add_argument('--folder', '-f', type=str,
                        help='path to folders')
    parser.add_argument('--out', '-o', type=str,
                        help='path to out')
    parser.add_argument('--verbose', '-v', type=int,
                        help='verbose', default=1)
    parser.add_argument('--base_lr', '-lr', type=int,
                        help='lr used first')
    parser.add_argument('--epochs', '-e', type=int,
                        help='nb of epochs')
    parser.add_argument('--batch', '-b', type=int,
                        help='Minibatch size', default=50)
    parser.add_argument('--checkpoint', '-c', type=int,
                        help='use checkpoint', default=1)
    parser.add_argument('--checkpoint_occurrence', '-co', type=int,
                        help='number of epochs per checkpoint', default=10)
    parser.add_argument('--init_step', '-is', type=int,
                        help='nb of epochs', default=0)
    parser.add_argument('--sequence', '-q', type=int,
                        help='Training sequence', default=1)
    parser.add_argument('--validation', '-val', type=str,
                        help='use validation data', default=True)
    parser.add_argument('--validation_prop', '-vp', type=float,
                        help='proporsion of data use for validation', default=0.2)
    parser.add_argument('--multiprocessing', '-m', type=str,
                        help='use_multiprocessing', default=False)
    parser.add_argument('--workers', '-w', type=int,
                        help='nb of workers', default=1)

    # TODO: add validation data
    args = parser.parse_args()

    '''
    # TODO: suppress this after test
    args.folder = os.getcwd().replace('bin', 'exp')
    args.type = 'train'
    args.out = os.path.join(os.getcwd().replace('bin', 'exp'), 'trained')
    args.verbose = 1
    args.base_lr = 1.10e-4
    args.epochs = 10
    args.checkpoint = 1
    args.checkpoint_occurrence = 5
    args.init_step = 1
    args.sequence = 150
    args.batch = 32
    args.multiprocessing = True
    args.workers = 4
    '''

    with open(os.path.join(args.folder, 'configuration.pickle'), 'rb') as f:
        config = pickle.load(f)

    main()
