#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 




# author: xiaoy li 
# 


REPO_PATH=/home/lixiaoya/xiaoy_tf
export PYTHONPATH=${REPO_PATH}

output_dir=/xiaoya/corefqa_data_output
input_file=/xiaoya/data/test.english.v4_gold_conll


python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${output_dir} \
--do_eval=True \
--output_dir=${output_dir} \
--input_dir=${input_file}

