#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# author: https://github.com/tensorflow/tpu/blob/b9fd824ed36bd36e0a2c1507cc1c9f71fd9fc0e8/models/official/mask_rcnn/box_utils.py
# utils functions to manipulate boxes 


import numpy as np
import tensorflow as tf


BBOX_XFORM_CLIP = np.log(1000. / 16.)
NMS_TILE_SIZE = 512


def top_k(scores, k, boxes_list):
  """A wrapper that returns top-k scores and correponding boxes.
  This functions selects the top-k scores and boxes as follows.
  indices = argsort(scores)[:k]
  scores = scores[indices]
  outputs = []
  for boxes in boxes_list:
    outputs.append(boxes[indices, :])
  return scores, outputs
  Args:
    scores: a tensor with a shape of [batch_size, N]. N is the number of scores.
    k: an integer for selecting the top-k elements.
    boxes_list: a list containing at least one element. Each element has a shape
      of [batch_size, N, 4].
  Returns:
    scores: the selected top-k scores with a shape of [batch_size, k].
    outputs: the list containing the corresponding boxes in the order of the
      input `boxes_list`.
  """
  assert isinstance(boxes_list, list)

  batch_size, _ = scores.get_shape().as_list()
  with tf.name_scope('top_k_wrapper'):
    scores, top_k_indices = tf.nn.top_k(scores, k=k)
    outputs = []
    for boxes in boxes_list:
      if batch_size == 1:
        boxes = tf.squeeze(tf.gather(boxes, top_k_indices, axis=1), axis=1)
      else:
        boxes_index_offsets = tf.range(batch_size) * tf.shape(boxes)[1]
        boxes_indices = tf.reshape(
            top_k_indices + tf.expand_dims(boxes_index_offsets, 1), [-1])
        boxes = tf.reshape(
            tf.gather(tf.reshape(boxes, [-1, 4]), boxes_indices),
            [batch_size, -1, 4])
      outputs.append(boxes)
    return scores, outputs



