import os
import sys
import logging
import glob

# TODO: Failed to run script from another python scripts...

prefix = './bin/'
print('########### Music2Dance ###########')

# general configuration
stage = 0
dataset_master_folder = './dataset_master'
exp = './exp'

type = 'train'

fps = 25
sampling = 44100
hop_length = 1764
wlen = 256


print("stage 0: preparing motion file...")
os.system('chmod u+x ' + prefix + "motion_prepare.py ")

print('starting',
      prefix,
      "motion_prepare.py",
      "-fps", str(fps),
      '-sampling', str(sampling),
      '-hop_length', str(hop_length),
      '-wlen', str(wlen),
      '-snr', str(0),
      '-folder', str(dataset_master_folder),
      '-save', str(exp),
      '-type', type)
os.system(prefix +
          "motion_prepare.py " +
          'starting ' +
          prefix +
          "motion_prepare.py " +
          " -fps " + str(fps) +
          ' -sampling ' + str(sampling) +
          ' -hop_length ' + str(hop_length) +
          ' -wlen ' +  str(wlen) +
          ' -snr ' + str(0) +
          ' -folder ' + str(dataset_master_folder) +
          ' -save ' + str(exp) +
          ' -type ' + type)

print("----- End-to-End stage")
