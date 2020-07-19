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
For any question, please feel free to post Github issues.

## Contents 
- [Overview](#overview)
- [Experimental Results](#experimental-results)
- [Data Preprocess](#data-preprocess)
- [Replicate Experimental Results](#replicate-experimental-results)
    - [Install Package Dependencies](#install-package-dependencies)
    - [Train CorefQA Model](#train-corefqa-model)
    - [Prediction](#prediction)
- [Evaluating the Trained Model](#evaluating-the-trained-model)
- [Descriptions of Directories](#descriptions-of-directories)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)


## Overview 


## Experimental Results 

| Model          | F1 (%) |
| -------------- |:------:|
| CorefQA + SpanBERT-base  | 79.9  |
| CorefQA + SpanBERT-large | 83.1   |



## Data Preprocess 
1. Download the [Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) dataset.
2. Split train/dev/test datasets and preprocess the official release `ontonotes-release-5.0` for coreference resolution annotations. <br>
Run `./scripts/preprocess_conll_data.sh  <PATH-TO-ontonotes-release-5.0> <PATH-TO-SAVE-CoNLL-FORMAT-DATASETS> <PATH-TO-CorefQA-REPOSITORY>`. <br>
E.g.: `./scripts/preprocess_conll_data.sh /home/shannon/ontonotes-release-5.0 /home/shannnon/conll12_coreference_data /home/shannon/CorefQA`
3. Generate and save training datasets to TFRecord files. <br>
Run `./scripts/generate_train_data.sh <PATH-TO-SAVE-CoNLL-FORMAT-DATASETS> <LANGUAGE> <NUMBER-of-SLIDING-WINDOW-SIZE>`<br>
E.g.: `./scripts/generate_train_data.sh /home/shannon/conll12_coreference_data english 384`


## Replicate Experimental Results 

### Install Package Dependencies 

* Install packages dependencies via : `pip install -r requirements.txt`
* Cloud TPU v3-8 device with Tensorflow 1.15 Python 3.5. <br> 


### Train CorefQA Model

1. Download Data Augmentation Models <br>
Run `./scripts/download_qauad2_finetune_model.sh <model-scale> <path-to-save-model>` to download finetuned SpanBERT on SQuAD2.0. <br>
The `<model-scale>` should take the value of `[base, large]`. <br>
The `<path-to-save-model>` is the path to save finetuned spanbert on SQuAD2.0 datasets. <br>
2. Train CoNLL-12 Coreference Resolution Model. <br> 
If using TPU, please run `./scripts/models/train_tpu.sh`<br>


### Prediction



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