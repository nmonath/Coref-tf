#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# transform pytorch .bin models to tensorflow ckpt 


import os 
import shutil 
from bert import modeling 
import tensorflow as tf  
from operation_funcs.load_pytorch_to_tf import load_from_pytorch_checkpoint 


def load_models(bert_config_path, ):
    bert_config = modeling.BertConfig.from_json_file(bert_config_path)
    input_ids = tf.ones((8, 128), tf.int32)

    model = modeling.BertModel(
        config=bert_config,
        is_training=False, 
        input_ids=input_ids,
        use_one_hot_embeddings=False, 
        scope="bert")

    return model, bert_config 


def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)


def main(bert_config_path, bert_ckpt_path, pytorch_init_checkpoint, output_tf_dir):
    # saver = tf.train.Saver()

    with tf.Session() as session:
        model, bert_config = load_models(bert_config_path)
        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, bert_ckpt_path)
        session.run(tf.global_variables_initializer())
        init_from_checkpoint = load_from_pytorch_checkpoint
        init_from_checkpoint(pytorch_init_checkpoint, assignment_map)

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                print("name = %s, shape = %s%s" % (var.name, var.shape, init_string))
        
        saver = tf.train.Saver()
        saver.save(session, os.path.join(output_tf_dir, "model"), global_step=100)
        copy_checkpoint(os.path.join(output_tf_dir, "model-{}".format(str(100))), os.path.join(output_tf_dir, "bert_model.ckpt"))
        print("=*="*30)
        print("save models : {}".format(output_tf_dir))
        print("=*="*30)


if __name__ == "__main__":
    # python3 transform_pytorch_ckpt.py 
    # spanbert base 
    bert_config_path = "/xiaoya/pretrain_ckpt/spanbert_base_cased/config.json"
    bert_ckpt_path = "/xiaoya/pretrain_ckpt/cased_L-12_H-768_A-12/bert_model.ckpt"
    pytorch_init_checkpoint = "/xiaoya/pretrain_ckpt/spanbert_base_cased/pytorch_model.bin"
    output_tf_dir = "/xiaoya/pretrain_ckpt/pytorch_to_tf/spanbert_base"
    os.makedirs(output_tf_dir, exist_ok=True)
    # main(bert_config_path, bert_ckpt_path, pytorch_init_checkpoint, output_tf_dir)

    # spanbert large 
    bert_config_path = "/xiaoya/pretrain_ckpt/spanbert_large_cased/config.json"
    bert_ckpt_path = "/xiaoya/pretrain_ckpt/cased_L-24_H-1024_A-16/bert_model.ckpt"
    pytorch_init_checkpoint = "/xiaoya/pretrain_ckpt/spanbert_large_cased/pytorch_model.bin"
    output_tf_dir = "/xiaoya/pretrain_ckpt/pytorch_to_tf/spanbert_large"
    os.makedirs(output_tf_dir, exist_ok=True)
    main(bert_config_path, bert_ckpt_path, pytorch_init_checkpoint, output_tf_dir)















