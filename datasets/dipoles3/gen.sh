#!/bin/bash

OUT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SEMGEN=$( realpath ${OUT}/../../semgen/semgen.py )

N_PROC=16
N=4000  # number of images to generate
W=640
H=640

# dipole
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} ${OUT}dipoles3 dipole \
  --dipole-n 10 50 \
  --enable-gradient

# dipole labels
python ${SEMGEN} -p -t ${N_PROC} generate -o \
  -n ${N} -d ${W} ${H} ${OUT}dipoles3_labels dipole-labels \
  --params ${OUT}dipoles3

# sem noise
# python ${SEMGEN} -p -t ${N_PROC} distort -o \
#  ${OUT}dipoles3 ${OUT}dipoles3_noise semnoise \
#  --gaussian-size 7 \
#  --scan-passes 1
