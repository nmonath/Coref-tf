#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 


REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}

OUTPUT_DIR=/dev/shm/xiaoya/export_dir_corefqa

rm -rf ${OUTPUT_DIR}

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1 python3 ${REPO_PATH}/run/train_corefqa.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--use_tpu=False \
--iterations_per_loop=50
