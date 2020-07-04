#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 


REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}

OUTPUT_DIR=/xiaoya/export_dir


CUDA_VISIBLE_DEVICES=0 python3 ${REPO_PATH}/run/train_corefqa.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--use_tpu=False \
--iterations_per_loop=50
