#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}

OUTPUT_DIR=/xiaoya/export_dir_mention
config_filename=config/experiments_tinybert.conf

rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

logfile_path=/home/lixiaoya/xiaoy_tf/logs/mention_proposal/test_log.log
config_params=train_tinybert


CUDA_VISIBLE_DEVICES=3 python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--do_eval=True \
--use_tpu=False \
--logfile_path=${logfile_path} \
--config_filename=${config_filename} \
--config_params=${config_params} \
--concat_only=True \
--iterations_per_loop=5000