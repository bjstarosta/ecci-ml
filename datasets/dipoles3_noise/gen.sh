#!/bin/bash

OUT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SEMGEN=$( realpath ${OUT}/../../semgen/semgen.py )

N_PROC=16
N=4000  # number of images to generate
W=640
H=640

mkdir -p ${OUT}/dipoles3_clean
mkdir -p ${OUT}/dipoles3_labels

# dipole
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} ${OUT}/dipoles3_clean dipole \
  --dipole-n 10 50 \
  --enable-gradient

# sem noise
python ${SEMGEN} -p -t ${N_PROC} distort -o \
  ${OUT}dipoles3_clean ${OUT}/dipoles3 semnoise \
  --gaussian-size 7 \
  --scan-passes 1

# clean out the temporary images
rm -f ${OUT}/dipoles3_clean/semgen-*.tif

# dipole labels
python ${SEMGEN} -p -t ${N_PROC} generate -o \
  -n ${N} -d ${W} ${H} ${OUT}/dipoles3_labels dipole-labels \
  --params ${OUT}/dipoles3_clean
