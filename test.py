import argparse
import os
import sys
import logging
import glob

prefix = 'exemple/of/path'
_type = 'train'
i=1
snr=5
h5file = '{}f{:03d}snr{:03d}.h5'.format(os.path.join(prefix, _type + '_'), i, snr)

print(h5file)
