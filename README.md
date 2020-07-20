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
- [Install Package Dependencies](#install-package-dependencies)
- [Data Preprocess](#data-preprocess)
- [Download Pretrained MLM](#download-pretrained-mlm)
- [Training](#training)
    - [Load Pretrained Models](#load-pretrained-models)
    - [Train CorefQA Model](#train-corefqa-model)
    - [Prediction](#prediction)
- [Evaluation](#evaluating-the-trained-model)
- [Descriptions of Directories](#descriptions-of-directories)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)


## Overview 
The model introduces +3.5 (83.1) F1 performance boost over previous SOTA coreference models on the CoNLL benchmark. The current codebase is written in Tensorflow. We plan to release the PyTorch version soon.  The current code version only supports training on TPUs and testing on GPUs (due to the annoying nature of TPUs). You thus have to bear the trouble of transferring all saved checkpoints from TPUs to GPU for evaluation (we plan fix this soon). Please follow the parameter setting in the log directionary to reproduce the performance.  

Please post github issues for any pertinent questions.

| Model          | F1 (%) |
| -------------- |:------:|
| Previous SOTA  (Joshi et al., 2019a)  | 79.6  |
| CorefQA + SpanBERT-large | 83.1   |


## Hardware Requirements
TPU for training: Cloud TPU v3-8 device (128G memory) with Tensorflow 1.15 Python 3.5 
GPU for evaluation: with CUDA 10.0 Tensorflow 1.15 Python 3.5

## Install Package Dependencies
 
```shell
$ python3 -m pip install --user virtualenv
$ virtualenv --python=python3.5 ~/corefqa_venv
$ source ~/corefqa_venv/bin/activate
$ cd coref-tf
$ pip install -r requirements.txt
# If you are using TPU, please run the following commands:
$ pip install cloud-tpu-client
$ pip install google-cloud-storage
```

## Data Preprocess 

1) Download the offical released [Ontonotes 5.0 (LDC2013T19)](https://catalog.ldc.upenn.edu/LDC2013T19). <br> 
2) Preprocess Ontonotes5 annotations files for the CoNLL-2012 coreference resolution task. <br> 
Run the command with **Python 2**
`bash ./scripts/data/preprocess_ontonotes_annfiles.sh  /path_to_LDC2013T19-ontonotes5.0_directory  /path_to_save_CoNLL12_coreference_resolution_directory <language>`<br> 
and it will create `{train/dev/test}.{language}.v4_gold_conll` files in the directory `/path_to_save_CoNLL12_coreference_resolution_directory`. <br> 
`<language>` can be `english`, `arabic` or `chinese`. In this paper, we set `<language>` to `english`. <br>
If you want to use **Python 3**, please refer to the
[guideline](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/train/training.md#get-the-data) <br> 
3) Generate TFRecord files for experiments. <br> 
Run the command with **Python 3** `bash ./scripts/data/generate_tfrecord_dataset.sh /path_to_save_CoNLL12_coreference_resolution_directory  /path_to_save_tfrecord_directory /path_to_vocab_file`
and it will create `{train/dev/test}.overlap.corefqa.{language}.tfrecord` files in the directory `/path_to_save_CoNLL12_coreference_resolution_directory`. <br> 

## Download Pretrained MLM
In our experiments, we used pretrained mask language models to initialize the mention_proposal and corefqa models. 

1) Download the pretrained models. <br> 
Run `bash ./scripts/data/download_pretrained_mlm.sh /path_to_save_pretrained_mlm <model_sign>` to download and unzip the pretrained mlm models. <br> 
`<model_sign>` shoule take the value of `[bert_base, bert_large, spanbert_base, spanbert_large, bert_tiny]`.

- `bert_base, bert_large, spanbert_base, spanbert_large` are trained with a cased(upppercase and lowercase tokens) vocabulary. Should use the cased train/dev/test coreference datasets. 
- `bert_tiny` is trained with a uncased(lowercase tokens) vocabulary. We use the tinyBERT model for fast debugging. Should use the uncased train/dev/test coreference datasets. <br> 

2) Transform SpanBERT from `Pytorch` to `Tensorflow`. <br> 
We need to tranform the SpanBERT checkpoints from Pytorch to TF because the offical relased models were trained with Pytorch. 
Run `bash ./scripts/data/transform_ckpt_pytorch_to_tf.sh <model_name>  /path_to_spanbert_<scale>_pytorch_dir /path_to_bert_<scale>_tf_dir  /path_to_save_spanbert_tf_checkpoint_dir` 
and the `<model_name>` in TF will be saved in `/path_to_save_spanbert_tf_checkpoint_dir`.

- `<model_name>` should take the value of `[spanbert_base, spanbert_large]`. 
- `<scale>` indicates that the `bert_model.ckpt` in the `/path_to_bert_<scale>_tf_dir` should have the same scale(base, large) to the `bert_model.bin` in `/path_to_spanbert_<scale>_pytorch_dir`.


## Training 

### Load Pretrained Models
Follow the pipeline described in the paper, you need to: <br> 
1) load a pretrained SpanBERT model. <br> 
2) finetune the SpanBERT model on the combination of Squad and Quoref datasets. <br> 
3) pretrain the mention proposal model on the coref dataset. <br>
4) jointly train the mention proposal model and the mention linking model. <br> 
**Notice:**: We provide the options of both pretraining these models yourself and loading the pretrained models for 2) and 3). <br> 


1. Download Data Augmentation Models on Squad and Quoref<br>
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
bert | BERT modules (model,tokenizer,optimization) ref to the `google-research/bert` repository. 
conll-2012 | offical evaluation scripts for CoNLL2012 shared task.
data_utils | modules for processing training data.  
func_builders | the input dataloader and model constructor for CorefQA.
logs | the log files in our experiments. 
models | an implementation of CorefQA/MentionProposal models based on TF.
run | modules for data preparation and training models.
scripts/data | scripts for data preparation and loading pretrained models.
scripts/models | scripts for {train/evaluate} {mention_proposal/corefqa} models on {TPU/GPU}. 
utils | modules including metrics、optimizers. 




## Acknowledgement

Many thanks to `Yuxian Meng` and the previous work `https://github.com/mandarjoshi90/coref`.

## Contact

Feel free to discuss papers/code with us through issues/emails!
