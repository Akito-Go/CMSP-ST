# CMSP-ST
This is a PyTorch implementation for INTERSPEECH 2025 main conference paper "CMSP-ST: Cross-modal Mixup with Speech Purification for End-to-End Speech Translation".

### **Dependencies**

* Python version >= 3.8
* [Pytorch](https://pytorch.org/)
* To install [fairseq](https://github.com/facebookresearch/fairseq) version 0.12.2 and develop locally:
    <pre>cd fairseq
  pip install --editable ./</pre>

### **Train your CMSP-ST model**

#### **1.Data Preparation**

* MuST-C: Download [MuST-C](https://mt.fbk.eu/must-c/) v1.0 dataset. Place the dataset in `./st/dataset/MuST-C/`.
  
* CoVoST-2: Download [CoVoST-2](https://commonvoice.mozilla.org/en/datasets) dataset. Place the dataset in `./st/dataset/CoVoST/`.
  
* HuBERT Model: Download [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) model. Place the model in `./models/pretrain/`.

* WMT: Download WMT [14](https://www.statmt.org/wmt14/translation-task.html) / [16](https://www.statmt.org/wmt16/translation-task.html) dataset. Place the dataset in `./mt/dataset/WMT/`.

#### **2.Preprocess**

###### **1) st vocab construction**
* cd ./data/st/s2t_raw/
* bash `prep_mustc_data.sh` or `prep_covost_data.sh`

###### **2) mt vocab construction**
* cd ./data/mt/s2t_raw/
* bash `prep_mtl_mustc_mt.sh` or `prep_mtl_covost_mt.sh` (for multi-task learning)
* bash `prep_exp_mustc_mt.sh` or `prep_exp_covost_mt.sh` (for expanded data)

#### **3.MT Pretraining**

###### **1) for multi-task learning**
* cd ./scripts/pretrain/
* bash `train_mtl_mt.sh` and `average_cpt.sh`

###### **2) for expanded data**
* bash `train_exp_mt.sh` and `average_cpt.sh`
* bash `train_exp_mtl_mt.sh` and `average_cpt.sh`

#### **4.Training and Inference**
* cd ./scripts/train/
* bash `train_xxxxx_xx2xx.sh` and `evaluation.sh`
