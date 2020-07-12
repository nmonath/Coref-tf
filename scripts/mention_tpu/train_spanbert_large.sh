#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



REPO_PATH=/home/xiaoyli1110/xiaoya/Coref-tf
export PYTHONPATH="$PYTHONPATH:/home/xiaoyli1110/xiaoya/Coref-tf"
export TPU_NAME=tf-tpu-2
# export TPU_NAME=
GCP_PROJECT=xiaoyli-20-04-274510
OUTPUT_DIR=gs://mention_proposal/spanbert_large_overlap_128_5_output_2e-5


CONFIG_FILENAME=config/tpu_mention_128_5_spanbert_large.conf 


python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--do_eval=True \
--do_predict=False \
--use_tpu=True \
--iterations_per_loop=500 \
--config_filename=${CONFIG_FILENAME} \
--tpu_name=${TPU_NAME} \
--tpu_zone=us-central1-f \
--gcp_project=${GCP_PROJECT} \
--num_tpu_cores=1 # > ${OUTPUT_DIR}/log.txt