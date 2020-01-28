#!/bin/bash
#!/usr/bin/env python3
#!/usr/bin/env python2

echo "running path.sh"
. ./path.sh

# general configuration

stage=-1
dataset_master_folder=./dataset_master
exp=./exp

type='train'

lr=1.10e-4
epochs=10
batch=32
checkpoints=1
checkpoints_occurence=10

multiprocessing=True
workers=4

init_step=1
sequence=150

. ./local/parse_options.sh || exit 1;

fps=25
sampling=44100
hop_length=1764
wlen=256

train_folder=${exp}/trained

echo "============================================================"
echo "                        Music2Dance"
echo "============================================================"

if [ ${stage} -eq -1 ]; then
  echo "......Test......"
fi

if [ ${stage} -eq 0 ]; then
  echo "stage 0: preparing motion file..."
  motion_prepare.py -f ${fps} \
                    -sr ${sampling} \
                    -hl ${hop_length} \
                    -w ${wlen} \
                    -r 0 \
                    -d ${dataset_master_folder} \
                    -o ${exp} \
                    -t ${type} || exit 1;
  echo "----- End-to-End stage"
fi


if [ ${stage} -eq 1 ]; then
  echo "stage 1: making h5 data files..."
  motion_audio_treatment.py -t ${type} \
                            -f ${exp} || exit 1;
  echo "----- End-to-End stage"
fi


if [ ${stage} -eq 2 ]; then
  echo "stage 2: Training network "
  train_network.py -t ${type} \
                   -f ${exp} \
                   -o ${train_folder} \
                   -v 1 \
                   -lr ${lr} \
                   -e ${epochs} \
                   -b ${batch} \
                   -c ${checkpoints} \
                   -co ${checkpoints_occurence} \
                   -is ${init_step} \
                   -q ${sequence} \
                   -m ${multiprocessing} \
                   -w ${workers} || exit 1;
  echo "----- End-to-End stage"
fi


echo "`basename $0` Done."
