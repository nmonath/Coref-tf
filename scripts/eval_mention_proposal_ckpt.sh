#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# evaluate trained ckpt 


REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}

OUTPUT_DIR=/xiaoya/export_dir_mention_test

CONFIG_FILENAME=config/experiments_tinybert.conf


CUDA_VISIBLE_DEVICES=3 python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${OUTPUT_DIR} \
--do_train=False \
--do_eval=False \
--do_predict=True \
--use_tpu=False \
--concat_only=True \
--iterations_per_loop=500 \
--config_filename=${CONFIG_FILENAME} 