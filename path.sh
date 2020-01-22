#!/bin/bash

MAIN_ROOT=$PWD

if [ -e $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    . $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    . $MAIN_ROOT/tools/venv/bin/activate
fi

export PATH=${MAIN_ROOT}:${MAIN_ROOT}/bin:$PATH