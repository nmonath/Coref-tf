#!/usr/bin/env bash 
# -*- coding: utf-8 -*-



# author: xiaoy li
# description:
# transform trained spanbert language model from pytorch(.bin) to tensorflow(.ckpt). 
# PLEASE NOTICE: the same scale(Base/Large) BERT(TF) Models are also necessary. 



REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}


# for spanbert large 
echo "Transform SpanBERT Cased Base from Pytorch To TF"
python3 ${REPO_PATH}/run/transform_spanbert_pytorch_to_tf.py \
--spanbert_config_path /xiaoya/pretrain_ckpt/spanbert_large_cased/config.json \
--bert_tf_ckpt_path /xiaoya/pretrain_ckpt/cased_L-24_H-1024_A-16/bert_model.ckpt \
--spanbert_pytorch_bin_path /xiaoya/pretrain_ckpt/spanbert_large_cased/pytorch_model.bin \
--output_spanbert_tf_dir /xiaoya/pretrain_ckpt/tf_spanbert_large_cased



# for spanbert base 
echo "Transform SpanBERT Cased Large from Pytorch To TF"
python3 ${REPO_PATH}/run/transform_spanbert_pytorch_to_tf.py \
--spanbert_config_path /xiaoya/pretrain_ckpt/spanbert_base_cased/config.json \
--bert_tf_ckpt_path /xiaoya/pretrain_ckpt/cased_L-12_H-768_A-12/bert_model.ckpt \
--spanbert_pytorch_bin_path /xiaoya/pretrain_ckpt/spanbert_base_cased/pytorch_model.bin \
--output_spanbert_tf_dir /xiaoya/pretrain_ckpt/tf_spanbert_base_cased