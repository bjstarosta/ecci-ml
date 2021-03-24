#!/bin/bash

OUT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
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

mkdir -p ${OUT}/exp
mkdir -p ${OUT}/ground

# augment
mkdir -p ${OUT}/tmp_exp
mkdir -p ${OUT}/tmp_ground

python ${TOOLS} ${FLAGS} augm \
-i ${OUT}/M2_5K01_original.tif \
-o ${OUT}/tmp_exp/
python ${TOOLS} ${FLAGS} augm \
-i ${OUT}/M6_5K_original.tif \
-o ${OUT}/tmp_exp/
python ${TOOLS} ${FLAGS} augm \
-i ${OUT}/naresh_slide3_original.tif \
-o ${OUT}/tmp_exp/
python ${TOOLS} ${FLAGS} augm \
-i ${OUT}/M2_5K01_ground.tif \
-o ${OUT}/tmp_ground/
python ${TOOLS} ${FLAGS} augm \
-i ${OUT}/M6_5K_ground.png \
-o ${OUT}/tmp_ground/
python ${TOOLS} ${FLAGS} augm \
-i ${OUT}/naresh_slide3_ground.tif \
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
--diskradius 5

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
