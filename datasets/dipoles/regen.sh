#!/bin/bash

OUT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SEMGEN=$( realpath ${OUT}/../../semgen/semgen.py )

N_PROC=16
N=4000  # number of images to generate per folder
W=52
H=52

# dipole
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} -u ${OUT}/dipoles_vlc ${OUT}/dipoles_vlc dipole
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} -u ${OUT}/dipoles_lc ${OUT}/dipoles_lc dipole
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} -u ${OUT}/dipoles_hc ${OUT}/dipoles_hc dipole
python ${SEMGEN} -p -t ${N_PROC} generate -o -l \
  -n ${N} -d ${W} ${H} -u ${OUT}/constants ${OUT}/constants constant

# sem noise
python ${SEMGEN} -p -t ${N_PROC} distort -o \
 ${OUT}/dipoles_vlc ${OUT}/dipoles_vlc_noise semnoise
python ${SEMGEN} -p -t ${N_PROC} distort -o \
 ${OUT}/dipoles_lc ${OUT}/dipoles_lc_noise semnoise
python ${SEMGEN} -p -t ${N_PROC} distort -o \
 ${OUT}/dipoles_hc ${OUT}/dipoles_hc_noise semnoise
python ${SEMGEN} -p -t ${N_PROC} distort -o \
 ${OUT}/constants ${OUT}/constants_noise semnoise
