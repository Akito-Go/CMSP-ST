#!/bin/bash -v

set -e -o pipefail -x

export CUDA_VISIBLE_DEVICES=0

export SRC_LANG=en
export TGT_LANG=de
export MODE=pretrain
export TYPE=s2t_mustc
export TASK=${SRC_LANG}-${TGT_LANG}
export BRAN=exp_mtl_mt

export MT_ROOT=/path/to/cmsp/data/mt/s2t_raw/spminfo/mustc/en-de/mtl_mt
export CKP=/path/to/cmsp/scripts/saveckp/pretrain/s2t_mustc/en-de/exp_mt/avg_last_10_checkpoint.pt
export SAVE_DIR=/path/to/cmsp/scripts/saveckp/${MODE}/${TYPE}/${TASK}/${BRAN}
export LOG_DIR=/path/to/cmsp/scripts/${MODE}/${TYPE}/${TASK}/${BRAN}_log

mkdir -p ${SAVE_DIR}
mkdir -p ${LOG_DIR}

fairseq-train ${MT_ROOT} \
	--arch transformer --save-dir ${SAVE_DIR} --restore-file ${CKP} --share-decoder-input-output-embed \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
	--max-tokens 8192 --max-update 500000 --num-workers 4 \
	--optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--dropout 0.1 --weight-decay 0.0 \
	--seed 1 --update-freq 1 \
	--log-interval 10 \
	--validate-interval 1 --save-interval 1 \
	--keep-last-epochs 10 --patience 10 \
	--skip-invalid-size-inputs-valid-test \
	--fp16 --memory-efficient-fp16 \
	--ddp-backend no_c10d --distributed-world-size 1 \
	--source-lang ${SRC_LANG} --target-lang ${TGT_LANG} \
	2>&1 | tee ${LOG_DIR}/train.log