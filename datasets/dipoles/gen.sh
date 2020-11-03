#!/bin/bash

OUT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SEMGEN=$( realpath ${OUT}/../../semgen/semgen.py )

N_PROC=16
N=4000  # number of images to generate per folder
W=52
H=52

# dipole
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} ${OUT}/dipoles_vlc dipole \
  --dipole-n 1 1 \
  --dipole-offset 0 0 \
  --dipole-contrast 0.1 0.25 \
  --dipole-mask-size 2 4 \
  --enable-gradient \
  --gradient-limit 0.25 0.75 \
  --gradient-range 0.2 0.4
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} ${OUT}/dipoles_lc dipole \
  --dipole-n 1 1 \
  --dipole-offset 0 0 \
  --dipole-contrast 0.25 0.4 \
  --dipole-mask-size 2 4 \
  --enable-gradient \
  --gradient-limit 0.25 0.75 \
  --gradient-range 0.2 0.4
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} ${OUT}/dipoles_hc dipole \
  --dipole-n 1 1 \
  --dipole-offset 0 0 \
  --dipole-contrast 0.4 0.7 \
  --dipole-mask-size 2 4 \
  --enable-gradient \
  --gradient-limit 0.15 0.85 \
  --gradient-range 0.6 0.65
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} ${OUT}/constants constant \
  --grey-limit 0.15 0.85

# sem noise
python ${SEMGEN} -p -t ${N_PROC} distort -o \
 ${OUT}/dipoles_vlc ${OUT}/dipoles_vlc_noise semnoise
python ${SEMGEN} -p -t ${N_PROC} distort -o \
 ${OUT}/dipoles_lc ${OUT}/dipoles_lc_noise semnoise
python ${SEMGEN} -p -t ${N_PROC} distort -o \
 ${OUT}/dipoles_hc ${OUT}/dipoles_hc_noise semnoise
python ${SEMGEN} -p -t ${N_PROC} distort -o \
 ${OUT}/constants ${OUT}/constants_noise semnoise
