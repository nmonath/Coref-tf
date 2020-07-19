# CorefQA: Coreference Resolution as Query-based Span Prediction

The repository contains the code of the recent research advances in [Shannon.AI](http://www.shannonai.com). 


**CorefQA: Coreference Resolution as Query-based Span Prediction** <br>
Wei Wu, Fei Wang, Arianna Yuan, Fei Wu and Jiwei Li<br>
In ACL 2020. [paper](https://arxiv.org/abs/1911.01746)<br>
If you find this repo helpful, please cite the following:
```latex
@article{wu2019coreference,
  title={Coreference Resolution as Query-based Span Prediction},
  author={Wu, Wei and Wang, Fei and Yuan, Arianna and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1911.01746},
  year={2019}
}
```

## Contents 
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Data Preprocess](#data-preprocess)
- [Training](#replicate-experimental-results)
    - [Install Package Dependencies](#install-package-dependencies)
    - [Load Pretrained Models]
    - [Train CorefQA Model](#train-corefqa-model)
    - [Prediction](#prediction)
- [Evaluation](#evaluating-the-trained-model)
- [Descriptions of Directories](#descriptions-of-directories)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)


## Overview 
The model introduces +3.5 (83.1) F1 performance boost over previous SOTA coreference models on the CoNLL benchmark. The current codebase is written in Tensorflow. We plan to release the PyTorch version soon.  The current code version only supports training on TPUs and testing on GPUs (due to the annoying nature of TPUs). You thus have to bear the trouble of transferring all saved checkpoints from TPUs to GPU for evaluation (we plan fix this soon). Please follow the parameter setting in the log directionary to reproduce the performance.  

Please post github issues or email xiaoya_li@shannonai.com for any pertinent questions.

| Model          | F1 (%) |
| -------------- |:------:|
| Previous SOTA  (Joshi et al., 2019a)  | 79.6  |
| CorefQA + SpanBERT-large | 83.1   |

## Hardware Requirements
(xiaoya todo)
TPU for training: Cloud TPU v3-8 device (128G memory) with Tensorflow 1.15 Python 3.5 
GPU for evaluation: 

## Data Preprocess 
1. Download the [Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) dataset.
2. Split train/dev/test datasets and preprocess the official release `ontonotes-release-5.0` for coreference resolution annotations. <br>
Run `./scripts/preprocess_conll_data.sh  <PATH-TO-ontonotes-release-5.0> <PATH-TO-SAVE-CoNLL-FORMAT-DATASETS> <PATH-TO-CorefQA-REPOSITORY>`. <br>
E.g.: `./scripts/preprocess_conll_data.sh /home/shannon/ontonotes-release-5.0 /home/shannnon/conll12_coreference_data /home/shannon/CorefQA`
3. Generate and save training datasets to TFRecord files. <br>
Run `./scripts/generate_train_data.sh <PATH-TO-SAVE-CoNLL-FORMAT-DATASETS> <LANGUAGE> <NUMBER-of-SLIDING-WINDOW-SIZE>`<br>
E.g.: `./scripts/generate_train_data.sh /home/shannon/conll12_coreference_data english 384`


## Training 

### Install Package Dependencies 

* Install packages dependencies via : `pip install -r requirements.txt`
* Cloud TPU v3-8 device with Tensorflow 1.15 Python 3.5. <br> 

### Load Pretrained Models
Follow the pipeline described in the paper, you need to (1) load a pretrained SpanBERT model; (2) finetune the SpanBERT model on the combination of Squad and Quoref datasets; (3) pretrain the mention proposal model on the coref dataset; and (4) jointly train the mention proposal model and the mention linking model. We provide the options of both pretraining these models yourself and loading the pretrained models for (2) and (3). 

1. Download Data Augmentation Models  on Squad and Quoref<br>
Run `./scripts/download_qauad2_finetune_model.sh <model-scale> <path-to-save-model>` to download finetuned SpanBERT on SQuAD2.0. <br>
The `<model-scale>` should take the value of `[base, large]`. <br>
The `<path-to-save-model>` is the path to save finetuned spanbert on SQuAD2.0 datasets. <br>

2. Download the Pretrained Mention Proposal Model 
xiaoya todo

### Train CorefQA Model
1. Pretrain the mention proposal model on CoNLL-12

2. Jointly train the mention proposal model and linking model in CoNLL-12. <br> 


### Evaluation



## Descriptions of Directories

Name | Descriptions 
----------- | ------------- 
bert | 
conll-2012 | 
data_utils | 
func_builders | 
logs | The logs for experiments. 
models | An implementation of CorefQA/MentionProposal models based on TF.
run | Train / Evaluate MRC-NER models.
scripts/data | 
scripts/models | 
tests | 
utils | 




## Acknowledgement

Many thanks to `Yuxian Meng` and the previous work `https://github.com/mandarjoshi90/coref`.

## Contact

Feel free to discuss papers/code with us through issues/emails!
