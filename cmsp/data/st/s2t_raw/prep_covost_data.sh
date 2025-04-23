#!/bin/bash -v

set -e -o pipefail -x

export DATA=covost
export SRC_LANG=de
export TGT_LANG=en
export PREP_PATH=/path/to/cmsp/data/st/s2t_raw
export COVOST_ROOT=/path/to/cmsp/data/st/dataset/CoVoST/cv-corpus-x.x
export SPM_INFO=/path/to/cmsp/data/st/s2t_raw/spminfo/${DATA}
export SPM_DATA=/path/to/cmsp/data/st/s2t_raw/spmdata/${DATA}

mkdir -p ${SPM_INFO}
mkdir -p ${SPM_DATA}

python ${PREP_PATH}/prep_covost_data.py \
	--src-lang ${SRC_LANG} --tgt-lang ${TGT_LANG} \
	--data-root ${COVOST_ROOT} \
	--spm-info ${SPM_INFO} \
	--spm-data ${SPM_DATA} \
	--vocab-type unigram \
	--vocab-size 10000 \
	--min_n_frames 1000 --max_n_frames 480000 \
	--use-audio-input \