#!/bin/bash
#!/usr/bin/env python3
#!/usr/bin/env python2

echo "running path.sh"
. ./path.sh

# general configuration

stage=0
dataset_master_folder=./dataset_master
exp=./exp

type='train'

. local/parse_options.sh || exit 1;

fps=25
sampling=44100
hop_length=1764
wlen=256

echo "============================================================"
echo "                        Music2Dance"
echo "============================================================"

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
fi
echo "----- End-to-End stage"

if [ ${stage} -eq 1 ]; then
  echo "stage 1: whatever..."

fi
echo "----- End-to-End stage"

echo "`basename $0` Done."