#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# evaluate trained ckpt 


REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}

OUTPUT_DIR=/xiaoya/export_dir_mention
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}


CONFIG_FILENAME=config/eval_ckpt.conf


CUDA_VISIBLE_DEVICES=2 python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${OUTPUT_DIR} \
--do_train=False \
--do_eval=True \
--use_tpu=False \
--concat_only=True \
--iterations_per_loop=500 \
--config_filename=${CONFIG_FILENAME} 