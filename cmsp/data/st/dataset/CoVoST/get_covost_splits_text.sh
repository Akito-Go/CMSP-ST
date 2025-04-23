#!/bin/bash -v

set -e -o pipefail -x

export src_lang=de
export tgt_lang=en
export root_path=/path/to/cmsp/data/st/dataset/CoVoST/cv-corpus-x.x/de/
export tsv_path=/path/to/cmsp/data/st/dataset/CoVoST/cv-corpus-x.x/de/validated.tsv

python get_covost_splits.py \
	--version 2 --src-lang ${src_lang} --tgt-lang ${tgt_lang} \
	--root ${root_path} \
	--cv-tsv ${tsv_path}

python get_text.py \
	--path ${root_path} \
	--src-lang ${src_lang} --tgt-lang ${tgt_lang} \
	--mode train

python get_text.py \
	--path ${root_path} \
	--src-lang ${src_lang} --tgt-lang ${tgt_lang} \
	--mode dev

python get_text.py \
	--path ${root_path} \
	--src-lang ${src_lang} --tgt-lang ${tgt_lang} \
	--mode test