#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# REPO_PATH=/home/lixiaoya/xiaoy_tf
REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}

OUTPUT_DIR=/xiaoya/export_dir_mention
# config_filename=config/experiments.conf 
config_filename=config/experiments_tinybert.conf

rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=1 python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--do_eval=False \
--use_tpu=False \
--config_filename=${config_filename} \
--concat_only=True \
--iterations_per_loop=5000