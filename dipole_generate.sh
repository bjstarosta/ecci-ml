#!/bin/bash

OUT=datasets/

python semgen/semgen.py -p generate -o -n 4000 -d 52 52 -l ${OUT}dipoles_vlc dipole --grey-range 0.1 --grey-limit 0.25 0.75 --clip 0.75 1
python semgen/semgen.py -p generate -o -n 4000 -d 52 52 -l ${OUT}dipoles_lc dipole --grey-range 0.25 --grey-limit 0.25 0.75 --clip 0.75 1
python semgen/semgen.py -p generate -o -n 4000 -d 52 52 -l ${OUT}dipoles_hc dipole --grey-range 0.6 --grey-limit 0.15 0.85 --clip 0.7 1
python semgen/semgen.py -p generate -o -n 4000 -d 52 52 -l ${OUT}constants constant --grey-limit 0.15 0.85

python semgen/semgen.py -p distort -o -l ${OUT}dipoles_vlc ${OUT}dipoles_vlc_noise semnoise &
python semgen/semgen.py -p distort -o -l ${OUT}dipoles_lc ${OUT}dipoles_lc_noise semnoise &
python semgen/semgen.py -p distort -o -l ${OUT}dipoles_hc ${OUT}dipoles_hc_noise semnoise &
python semgen/semgen.py -p distort -o -l ${OUT}constants ${OUT}constants_noise semnoise &
