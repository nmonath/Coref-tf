#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 


REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}
OUTPUT_DIR=/xiaoya/export_dir_corefqa
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
EVAL_DIR=/xiaoya/export_dir_corefqa


CONFIG_FILENAME=config/gpu_corefqa.conf 
config_params=train_tinybert
logfile_path=/home/lixiaoya/xiaoy_tf/logs/corefqa/${config_params}.log


CUDA_VISIBLE_DEVICES=3 python3 ${REPO_PATH}/run/train_corefqa.py \
--output_dir=${OUTPUT_DIR} \
--eval_dir=${EVAL_DIR} \
--do_train=True \
--do_eval=True \
--do_predict=False \
--use_tpu=False \
--logfile_path=${logfile_path} \
--config_filename=${CONFIG_FILENAME} \
--config_params=${config_params} \
--config_filename=${CONFIG_FILENAME} \
--iterations_per_loop=5604 \
--config_filename=${CONFIG_FILENAME}



