#!/bin/bash

OUT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TOOLS=$( realpath ${OUT}/../tools.py )

CHUNK_W=512
CHUNK_H=512
CHUNK_STRIDE=0.25
PAD_W=128
PAD_H=128

mkdir -p ${OUT}/exp
mkdir -p ${OUT}/ground

# augm
mkdir -p ${OUT}/tmp1a
mkdir -p ${OUT}/tmp2a
python ${TOOLS} -v augm \
  -i ${OUT}/M2_5K01_original.tif \
  -o ${OUT}/tmp1a/
python ${TOOLS} -v augm \
  -i ${OUT}/M2_5K01_ground.tif \
  -o ${OUT}/tmp2a/
python ${TOOLS} -v augm \
  -i ${OUT}/naresh_slide3_original.tif \
  -o ${OUT}/tmp1a/
python ${TOOLS} -v augm \
  -i ${OUT}/naresh_slide3_ground.tif \
  -o ${OUT}/tmp2a/

# split
mkdir -p ${OUT}/tmp1b
mkdir -p ${OUT}/tmp2b
python ${TOOLS} -v split \
  -i ${OUT}/tmp1a/ \
  -o ${OUT}/exp/ \
  --width ${CHUNK_W} --height ${CHUNK_H} --stride ${CHUNK_STRIDE}
python ${TOOLS} -v split \
  -i ${OUT}/tmp2a/ \
  -o ${OUT}/ground/ \
  --width ${CHUNK_W} --height ${CHUNK_H} --stride ${CHUNK_STRIDE}

# pad
# python ${TOOLS} -v pad \
#   -i ${OUT}/tmp1b/ \
#   -o ${OUT}/exp/ \
#   --mode symmetric --pad-width ${PAD_W} ${PAD_H}
# python ${TOOLS} -v pad \
#   -i ${OUT}/tmp2b/ \
#   -o ${OUT}/ground/ \
#   --mode symmetric --pad-width ${PAD_W} ${PAD_H}

# clean up
rm -r ${OUT}/tmp1a
rm -r ${OUT}/tmp2a
rm -r ${OUT}/tmp1b
rm -r ${OUT}/tmp2b
