#!/bin/bash -v

set -e -o pipefail -x

export SRC_LANG=de
export TGT_LANG=en
export MODE=pretrain
export TYPE=s2t_covost
export TASK=${SRC_LANG}-${TGT_LANG}
export BRAN=exp_mtl_mt

export SCRIPTS=/path/to/fairseq/scripts
export SAVE_DIR=/path/to/cmsp/scripts/saveckp/${MODE}/${TYPE}/${TASK}/${BRAN}

python ${SCRIPTS}/average_checkpoints.py \
	--inputs ${SAVE_DIR} \
	--num-epoch-checkpoints 10 \
	--output ${SAVE_DIR}/avg_last_10_checkpoint.pt