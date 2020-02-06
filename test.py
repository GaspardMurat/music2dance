import argparse
import os
import sys
import logging
import glob
import numpy as np
import h5py
import tensorflow as tf

'''
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
'''
