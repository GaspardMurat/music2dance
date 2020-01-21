#!/bin/bash

. ./path.sh
. ./cmd.sh

# general configuration
stage=0
dataset_master_folder=./dataset_master
exp=./exp

type='train'

fps=25
sampling=44100
hop_length=1764
wlen=256

if [ ${stage} -eq 0 ]; then
  echo "stage 0: preparing motion file..."
  motion_prepare.py -fps ${fps} \
                    -sampling ${sampling} \
                    -hop_length ${hop_length} \
                    -wlen ${wlen} \
                    -snr 0 \
                    -folder ${dataset_master_folder} \
                    -save ${exp} \
                    -type ${train} || exit 1;
fi
echo "----- End-to-End stage"

if [ ${stage} -eq 1 ]; then
  echo "stage 0: preparing motion file..."
  motion_audio_treatment.py -fps ${fps} \
                            -sampling ${sampling} \
                            -hop_length ${hop_length} \
                            -wlen ${wlen} \
                            -snr 0 \
                            -folder ${dataset_master_folder} \
                            -save ${exp} \
                            -type ${train} || exit 1;
fi
echo "----- End-to-End stage"