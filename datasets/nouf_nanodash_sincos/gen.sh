#!/bin/bash

# This requires the nouf_nanodash_pos dataset

OUT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IN=$( realpath ${OUT}/../nouf_nanodash_pos )
FILTERS=$( realpath ${OUT}/../ )
TOOLS=$( realpath ${OUT}/../tools.py )
TOOLS_TDGROUND=$( realpath ${OUT}/../tools_tdground.py )

#FLAGS='-v'
FLAGS=''

CHUNK_W=512
CHUNK_H=512
CHUNK_STRIDE=0.25
PAD_X=0 #64
PAD_Y=0 #64
BLOB_MIN_R=5
BLOB_MAX_R=14
KERNEL_SIZE=20

mkdir -p ${OUT}/exp
mkdir -p ${OUT}/ground

# filter
python ${FILTERS}/filter_sincos.py ${FLAGS} -ks ${KERNEL_SIZE} \
-i ${IN}/P3.TIF \
-o ${OUT}
python ${FILTERS}/filter_sincos.py ${FLAGS} -ks ${KERNEL_SIZE} \
-i ${IN}/P3B.TIF \
-o ${OUT}
python ${FILTERS}/filter_sincos.py ${FLAGS} -ks ${KERNEL_SIZE} \
-i ${IN}/P3C.TIF \
-o ${OUT}

# augment
mkdir -p ${OUT}/tmp_exp
mkdir -p ${OUT}/tmp_ground

python ${TOOLS} ${FLAGS} augm \
-i ${OUT}/P3.TIF \
-o ${OUT}/tmp_exp/
python ${TOOLS} ${FLAGS} augm \
-i ${OUT}/P3B.TIF \
-o ${OUT}/tmp_exp/
python ${TOOLS} ${FLAGS} augm \
-i ${OUT}/P3C.TIF \
-o ${OUT}/tmp_exp/
python ${TOOLS} ${FLAGS} augm \
-i ${IN}/P3.png \
-o ${OUT}/tmp_ground/
python ${TOOLS} ${FLAGS} augm \
-i ${IN}/P3B.png \
-o ${OUT}/tmp_ground/
python ${TOOLS} ${FLAGS} augm \
-i ${IN}/P3C.png \
-o ${OUT}/tmp_ground/

# detect ground truth blobs
mkdir -p ${OUT}/tmp_ground_csv

python ${TOOLS_TDGROUND} ${FLAGS} blob-coords \
-i ${OUT}/tmp_ground/ \
-o ${OUT}/tmp_ground_csv/ \
--min_r ${BLOB_MIN_R} --max_r ${BLOB_MAX_R}

# generate ground truth images
mkdir -p ${OUT}/tmp_ground_out

python ${TOOLS_TDGROUND} ${FLAGS} make-circles \
-i ${OUT}/tmp_ground_csv/ \
-o ${OUT}/tmp_ground_out/ \
--width ${CHUNK_W} --height ${CHUNK_H} --stride ${CHUNK_STRIDE} \
--type square --diskradius 10

# pad
python ${TOOLS} ${FLAGS} pad \
-i ${OUT}/tmp_ground_out/ \
-o ${OUT}/ground/ \
--mode reflect --pad-width ${PAD_X} ${PAD_Y}

# split
python ${TOOLS} ${FLAGS} split \
-i ${OUT}/tmp_exp/ \
-o ${OUT}/exp/ \
--width ${CHUNK_W} --height ${CHUNK_H} --stride ${CHUNK_STRIDE}

# clean up
rm -r ${OUT}/tmp_exp
rm -r ${OUT}/tmp_ground
rm -r ${OUT}/tmp_ground_csv
rm -r ${OUT}/tmp_ground_out
