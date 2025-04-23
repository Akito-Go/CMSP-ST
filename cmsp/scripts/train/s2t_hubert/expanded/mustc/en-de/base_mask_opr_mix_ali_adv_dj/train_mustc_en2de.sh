#!/bin/bash -v

set -e -o pipefail -x

export CUDA_VISIBLE_DEVICES=0

export LANG=de
export MODE=train
export TYPE=s2t_hubert
export TASK=expanded
export DATA=mustc
export PAIR=en-${LANG}
export BRAN=base_mask_opr_mix_ali_adv_dj

export ARCH=s2t_hubert_mtlip_mix_ali_adv_post
export CRITERION=label_smoothed_cross_entropy_mtlip_mix_ali_adv_dj

export ROOT=/home/jlou/CMSP-ST/cmsp
export USER=${ROOT}/arch
export MUSTC_ROOT=${ROOT}/data/st/dataset/MuST-C/en-${LANG}
export PREMODEL=${ROOT}/arch/models/pretrain
export MTMODEL=${ROOT}/scripts/saveckp/pretrain/s2t_mustc/en-${LANG}/exp_mtl_mt/avg_last_10_checkpoint.pt
export SPM_INFO=${ROOT}/data/st/s2t_raw/spminfo/mustc/en-${LANG}
export SAVE_DIR=${ROOT}/scripts/saveckp/${MODE}/${TYPE}/${TASK}/${BRAN}
export LOG_DIR=${ROOT}/scripts/${MODE}/${TYPE}/${TASK}/${BRAN}/${BRAN}_log

mkdir -p ${SAVE_DIR}
mkdir -p ${LOG_DIR}

fairseq-train ${MUSTC_ROOT} \
	--task speech_and_text \
	--user-dir ${USER} \
	--arch ${ARCH} --save-dir ${SAVE_DIR} \
	--use-mix --use-adv \
	--hubert-model-path ${PREMODEL}/hubert_base_ls960.pt --mt-model-path ${MTMODEL} \
	--config-yaml ${SPM_INFO}/config_st.yaml --train-subset train_st_raw --valid-subset dev_st_raw \
	--max-tokens 3300000 --max-source-positions 900000 --max-target-positions 1024 --batch-size 48 \
	--max-epoch 96 --max-update 80000 --num-workers 4 \
	--log-interval 10 --keep-last-epochs 10 --patience 5 \
	--criterion ${CRITERION} --label-smoothing 0.1 --report-accuracy \
	--layernorm-embedding --optimizer adam --adam-betas "(0.9, 0.98)" \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 25000 \
	--clip-norm 0.0 --seed 1 --update-freq 8 \
	--skip-invalid-size-inputs-valid-test \
	--fp16 --memory-efficient-fp16 \
	--ddp-backend no_c10d --distributed-world-size 1 \
	2>&1 | tee ${LOG_DIR}/train.log