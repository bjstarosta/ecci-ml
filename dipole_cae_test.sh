#!/bin/bash

# HOW TO USE:
# Download semgen (https://github.com/bjstarosta/semgen) into this directory,
# then run this script. It will automatically generate synthetic examples and
# compare them against the current trained autoencoder model. Samples will
# regenerate on each script run.

OUT1=datasets/dipoles_test/
OUT2=datasets/dipoles_test_noise/

if [ ! -d "${OUT1}" ]; then
    mkdir -p "${OUT1}";
fi;
if [ ! -d "${OUT2}" ]; then
    mkdir -p "${OUT2}";
fi;
rm ${OUT1}*
rm ${OUT2}*

python semgen/semgen.py -p generate -n 3 -d 52 52 -l ${OUT1} dipole --grey-range 0.1 --grey-limit 0.25 0.75 --clip 0.75 1
python semgen/semgen.py -p generate -n 3 -d 52 52 -l ${OUT1} dipole --grey-range 0.25 --grey-limit 0.25 0.75 --clip 0.75 1
python semgen/semgen.py -p generate -n 3 -d 52 52 -l ${OUT1} dipole --grey-range 0.6 --grey-limit 0.15 0.85 --clip 0.7 1
python semgen/semgen.py -p generate -n 3 -d 52 52 -l ${OUT1} constant --grey-limit 0.15 0.85
python semgen/semgen.py -p distort -o -l ${OUT1} ${OUT2} semnoise
python cae_test.py -v comparison
