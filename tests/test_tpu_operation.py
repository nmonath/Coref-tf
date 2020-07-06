#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# descripiton:
# test math operations in tpu 


import os 
import sys 
import tensorflow as tf 


TPU_NAME = ""
TPU_ZONE = ""
GCP_PROJECT = ""



if __name__ == "__main__":
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_NAME, zone=TPU_ZONE, project=GCP_PROJECT)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

    with tf.compat.v1.InteractiveSession(tpu_cluster_resolver) as sess:
        sess.run(tpu.initialize_system())

        scores = tf.constant([1.0, 2.3, 3.2, 4.3, 1.5, 1.8, 98, 2.9])
        k = 2
        top_scores, top_index = tf.nn.top_k(scores, k)

        top_scores.eval()
        top_index.eval()

        sess.run(tpu.shutdown_system())