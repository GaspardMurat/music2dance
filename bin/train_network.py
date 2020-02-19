#!/usr/bin/env python3
import os
import argparse
import glob
import logging
import pickle
import matplotlib.pyplot as plt

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, LearningRateScheduler, TensorBoard
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import sys

module_network = os.getcwd().replace('bin', 'network')
sys.path.append(module_network)
from network.convLSTM2dMoldel import ConvLSTM2dModel

module_utils = os.getcwd().replace('bin', 'utils')
sys.path.append(module_utils)
from utils.dataset2 import DataGenerator2


def main():
    if 'test' not in config:
        logging.warning('Path to validation set does not exist')
        quit()

    path = config['train']
    file_list = glob.glob(os.path.join(path, '*'))
    logging.info('number of h5 files: {}'.format(len(file_list)))

    train_dataset = DataGenerator2(path, args.batch, args.sequence, args.sequence_out, 'train', args.init_step, shuffle=True)
    batch_0 = train_dataset[270]
    input_encoder_shape = batch_0[0][0].shape[1:]
    input_decoder_shape = batch_0[0][0].shape[1:]
    output_shape = batch_0[1].shape[1:]
    folder_models = os.path.join(args.out, 'models')

    if not os.path.exists(folder_models):
        os.makedirs(folder_models)

    model = ConvLSTM2dModel(input_encoder_shape, output_shape, args.base_lr)

    model_saver = ModelCheckpoint(filepath=os.path.join(folder_models, 'model.ckpt.{epoch:04d}.hdf5'),
                                  verbose=1,
                                  save_best_only=False,
                                  period=10)
    print(model.summary())

    def lr_scheduler(epoch, lr):
        decay_rate = 0.90
        decay_step = 20
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr

    callbacks_list = [model_saver,
                      TerminateOnNaN(),
                      LearningRateScheduler(lr_scheduler, verbose=1)]

    if args.validation_set:
        validation_path = config['test']
        test_dataset = DataGenerator2(validation_path, args.batch, args.sequence, args.sequence_out,'test', args.init_step,
                                     shuffle=True)

        history = model.fit_generator(train_dataset,
                                      validation_data=test_dataset,
                                      epochs=args.epochs,
                                      use_multiprocessing=args.multiprocessing,
                                      workers=args.workers,
                                      callbacks=callbacks_list,
                                      verbose=args.verbose)
    else:
        history = model.fit_generator(train_dataset,
                                      epochs=args.epochs,
                                      use_multiprocessing=args.multiprocessing,
                                      workers=args.workers,
                                      callbacks=callbacks_list,
                                      verbose=args.verbose)

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

    plot_loss(history, args.out)

    model.save(os.path.join(args.out, 'models', 'model.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', '-f', type=str,
                        help='path to folders')
    parser.add_argument('--out', '-o', type=str,
                        help='path to out')
    parser.add_argument('--verbose', '-v', type=int,
                        help='verbose', default=1.10e-4)
    parser.add_argument('--base_lr', '-lr', type=float,
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
    parser.add_argument('--sequence_out', '-p', type=int,
                        help='Training out sequence', default=1)
    parser.add_argument('--validation', '-val', type=str,
                        help='use validation data', default=True)
    parser.add_argument('--validation_prop', '-vp', type=float,
                        help='proporsion of data use for validation', default=0.2)
    parser.add_argument('--multiprocessing', '-m', type=str,
                        help='use_multiprocessing', default=False)
    parser.add_argument('--workers', '-w', type=int,
                        help='nb of workers', default=1)
    parser.add_argument('--validation_set', '-vs', type=str,
                        help='Use of validation set or not', default=False)

    args = parser.parse_args()

    with open(os.path.join(args.folder, 'configuration.pickle'), 'rb') as f:
        config = pickle.load(f)

    main()
