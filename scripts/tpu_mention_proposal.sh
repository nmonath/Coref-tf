#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# REPO_PATH=/home/lixiaoya/xiaoy_tf
REPO_PATH=/home/xiaoyli1110/xiaoya/Coref-tf
export PYTHONPATH=${REPO_PATH}
export PYTHONPATH="$PYTHONPATH:/home/xiaoyli1110/xiaoya/Coref-tf"
export TPU_NAME=tensorflow-tpu
GCP_PROJECT=xiaoyli-20-04-274510
OUTPUT_DIR=gs://mention_proposal/output_30

# CUDA_VISIBLE_DEVICES=0 
python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--use_tpu=True \
--iterations_per_loop=500 \
--tpu_name=${TPU_NAME} \
--tpu_zone=us-central1-f \
--gcp_project=${GCP_PROJECT} \
--master=10.242.10.122 \
--num_tpu_cores=1