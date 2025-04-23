#!/bin/bash -v

set -e -o pipefail -x

export DATA=mustc
export LANG=de
export PREP_PATH=/path/to/cmsp/data/st/s2t_raw
export MUSTC_ROOT=/path/to/cmsp/data/st/dataset/MuST-C
export SPM_INFO=/path/to/cmsp/data/st/s2t_raw/spminfo/${DATA}
export SPM_DATA=/path/to/cmsp/data/st/s2t_raw/spmdata/${DATA}

mkdir -p ${SPM_INFO}
mkdir -p ${SPM_DATA}

python ${PREP_PATH}/prep_mustc_data.py \
	--task st \
	--tgt-lang ${LANG} \
	--data-root ${MUSTC_ROOT} \
	--spm-info ${SPM_INFO} \
	--spm-data ${SPM_DATA} \
	--vocab-type unigram \
	--vocab-size 10000 \
	--min_n_frames 1000 --max_n_frames 480000 \
	--use-audio-input \