#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# REPO_PATH=/home/lixiaoya/xiaoy_tf
REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}

OUTPUT_DIR=/xiaoya/export_dir_mention

mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=0 python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--do_eval=True \
--use_tpu=False \
--concat_only=True \
--iterations_per_loop=500