#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


import os
import sys 


repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)


import tensorflow as tf
from bert import modeling



class MentionProposalModel(object):
    def __init__(self, config):
        self.config = config 
        self.bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)
        self.bert_config.hidden_dropout_prob = config.dropout_rate

    def get_mention_proposal_and_loss(self, instance, is_training, use_tpu=False):
        self.use_tpu = use_tpu 
        self.dropout = self.get_dropout(self.config.dropout_rate, is_training)

        flat_input_ids, flat_doc_overlap_input_mask, flat_sentence_map, text_len, speaker_ids, gold_starts, gold_ends, cluster_ids = instance

        flat_input_ids = tf.math.maximum(flat_input_ids, tf.zeros_like(flat_input_ids, tf.int32))
        
        flat_doc_overlap_input_mask = tf.where(tf.math.greater_equal(flat_doc_overlap_input_mask, 0), x=tf.constant(1), y=tf.constant(0))
        # flat_doc_overlap_input_mask = tf.math.maximum(flat_doc_overlap_input_mask, tf.zeros_like(flat_doc_overlap_input_mask, tf.int32))
        flat_sentence_map = tf.math.maximum(flat_sentence_map, tf.zeros_like(flat_sentence_map, tf.int32))
        
        gold_start_end_mask = tf.cast(tf.math.greater_equal(gold_starts, tf.zeros_like(gold_starts, tf.int32)), tf.bool)
        gold_start_index_labels = self.boolean_mask_1d(gold_starts, gold_start_end_mask, scope="gold_starts", use_tpu=self.use_tpu)
        gold_end_index_labels = self.boolean_mask_1d(gold_ends, gold_start_end_mask, scope="gold_ends", use_tpu=self.use_tpu)

        text_len = tf.math.maximum(text_len, tf.zeros_like(text_len, tf.int32))
        num_words_in_doc = tf.math.reduce_sum(text_len)

        input_ids = tf.reshape(flat_input_ids, [-1, self.config.window_size])
        input_mask = tf.ones_like(input_ids, tf.int32)

        model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=input_ids, input_mask=input_mask, 
            use_one_hot_embeddings=False, scope='bert')

        doc_overlap_window_embs = model.get_sequence_output()
        doc_overlap_input_mask = tf.reshape(flat_doc_overlap_input_mask, [self.config.num_window, self.config.window_size])

        doc_flat_embs = self.transform_overlap_windows_to_original_doc(doc_overlap_window_embs, doc_overlap_input_mask)
        doc_flat_embs = tf.reshape(doc_flat_embs, [-1, self.config.hidden_size])

        expand_start_embs = tf.tile(tf.expand_dims(doc_flat_embs, 1), [1, num_words_in_doc, 1])
        expand_end_embs = tf.tile(tf.expand_dims(doc_flat_embs, 0), [num_words_in_doc, 1, 1])
        expand_mention_span_embs = tf.concat([expand_start_embs, expand_end_embs], axis=-1)
        expand_mention_span_embs = tf.reshape(expand_mention_span_embs, [-1, self.config.hidden_size*2])
        span_sequence_logits = self.ffnn(expand_mention_span_embs, self.config.hidden_size*2, 1, dropout=self.dropout, name_scope="mention_span")

        if self.config.start_end_share:
            start_end_sequence_logits = self.ffnn(doc_flat_embs, self.config.hidden_size, 2, dropout=self.dropout, name_scope="mention_start_end")
            start_sequence_logits, end_sequence_logits = tf.split(start_end_sequence_logits, axis=1)
        else:
            start_sequence_logits = self.ffnn(doc_flat_embs, self.config.hidden_size, 1, dropout=self.dropout, name_scope="mention_start")
            end_sequence_logits = self.ffnn(doc_flat_embs, self.config.hidden_size, 1, dropout=self.dropout, name_scope="mention_end")

        gold_start_sequence_labels = self.scatter_gold_index_to_label_sequence(gold_start_index_labels, num_words_in_doc)
        gold_end_sequence_labels = self.scatter_gold_index_to_label_sequence(gold_end_index_labels, num_words_in_doc)

        start_loss, start_sequence_scores = self.compute_score_and_loss(start_sequence_logits, gold_start_sequence_labels)
        end_loss, end_sequence_scores = self.compute_score_and_loss(end_sequence_logits, gold_end_sequence_labels)

        gold_span_sequence_labels = self.scatter_span_sequence_labels(gold_start_index_labels, gold_end_index_labels, num_words_in_doc)
        span_loss, span_sequence_scores = self.compute_score_and_loss(span_sequence_logits, gold_span_sequence_labels)

        if self.config.mention_proposal_only_concate:
            return span_loss, span_sequence_scores 
        else:
            total_loss = self.config.loss_start_ratio * start_loss + self.config.loss_end_ratio * end_loss + self.config.loss_span_ratio * span_loss 
            return total_loss, start_sequence_scores, end_sequence_scores, span_sequence_scores


    def scatter_gold_index_to_label_sequence(self, gold_index_labels, expect_length_of_labels):
        gold_labels_pos = tf.reshape(gold_index_labels, [-1, 1])
        gold_value = tf.reshape(tf.ones_like(gold_index_labels), [-1])
        label_shape = tf.Variable(expect_length_of_labels, shape=[1])
        gold_label_sequence = tf.cast(tf.scatter_nd(gold_labels_pos, gold_value, label_shape), tf.int32)
        return gold_label_sequence 


    def scatter_span_sequence_labels(self, gold_start_index_labels, gold_end_index_labels, expect_length_of_labels):
        gold_span_index_labels = tf.stack([gold_start_index_labels, gold_end_index_labels], axis=1)
        gold_span_value = tf.reshape(tf.ones_like(gold_start_index_labels, tf.int32), [-1])
        gold_span_label_shape = tf.reshape(tf.Variable([expect_length_of_labels, expect_length_of_labels], shape=[2, 1]), [-1])

        gold_span_sequence_labels = tf.cast(tf.scatter_nd(gold_span_index_labels, gold_span_value, gold_span_label_shape), tf.int32)
        return gold_span_sequence_labels


    def compute_score_and_loss(self, pred_sequence_scores, gold_sequence_labels, loss_mask=None):
        pred_sequence_scores = tf.cast(tf.reshape(tf.sigmoid(pred_sequence_scores), [-1]),tf.float32)
        expand_pred_sequence_scores = tf.stack([(1 - pred_sequence_scores), pred_sequence_scores], axis=-1) 
        expand_gold_sequence_labels = tf.cast(tf.one_hot(tf.reshape(gold_sequence_labels, [-1]), 2, axis=-1), tf.float32)

        loss = tf.keras.losses.binary_crossentropy(expand_gold_sequence_labels, expand_pred_sequence_scores)

        if loss_mask is not None:
            loss = tf.multiply(loss, tf.cast(loss_mask, tf.float32))
        total_loss = tf.reduce_mean(loss)

        return total_loss, pred_sequence_scores 


    def transform_overlap_windows_to_original_doc(self, doc_overlap_window_embs, doc_overlap_input_mask):
        ones_input_mask = tf.ones_like(doc_overlap_input_mask, tf.int32)
        cumsum_input_mask = tf.math.cumsum(ones_input_mask, axis=1)
        offset_input_mask = tf.tile(tf.expand_dims(tf.range(self.config.num_window) * self.config.window_size, 1), [1, self.config.window_size])
        offset_cumsum_input_mask = offset_input_mask + cumsum_input_mask
        global_input_mask = tf.math.multiply(ones_input_mask, offset_cumsum_input_mask)
        global_input_mask_index = self.boolean_mask_1d(global_input_mask, tf.math.greater(global_input_mask, tf.zeros_like(global_input_mask, tf.int32)))

        doc_overlap_window_embs = tf.reshape(doc_overlap_window_embs, [-1, self.config.hidden_size])
        original_doc_embs = tf.gather(doc_overlap_window_embs, global_input_mask_index)
        return original_doc_embs 


    def ffnn(self, inputs, hidden_size, output_size, dropout=None, name_scope="fully-conntected-neural-network",
        hidden_initializer=tf.truncated_normal_initializer(stddev=0.02) ):
        
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            hidden_weights = tf.get_variable("hidden_weights", [hidden_size, output_size],
                initializer=hidden_initializer)
            hidden_bias = tf.get_variable("hidden_bias", [output_size], initializer=tf.zeros_initializer())
            outputs = tf.nn.relu(tf.nn.xw_plus_b(inputs, hidden_weights, hidden_bias))

            if dropout is not None:
                outputs = tf.nn.dropout(outputs, dropout)

        return outputs 


    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)


    def get_shape(self, x, dim):
        return x.get_shape()[dim].value or tf.shape(x)[dim]


    def boolean_mask_1d(self, itemtensor, boolmask_indicator, scope="boolean_mask1d", use_tpu=False):
        """
        Desc:
            the same functionality of tf.boolean_mask. 
            The tf.boolean_mask operation is not allowed by the TPU. 
        Args:
            itemtensor : a Tensor contains [tf.int32, tf.float32] numbers. Should be 1-Rank.
            boolmask_indicator : a tf.bool Tensor. Should be 1-Rank. 
            scope : name scope for the operation. 
            use_tpu : if False, return tf.boolean_mask.  
        """
        with tf.name_scope(scope):
            if not use_tpu:
                return tf.boolean_mask(itemtensor, boolmask_indicator)

            boolmask_sum = tf.reduce_sum(tf.cast(boolmask_indicator, tf.int32))
            selected_positions = tf.cast(boolmask_indicator, dtype=tf.float32)
            indexed_positions = tf.cast(tf.multiply(tf.cumsum(selected_positions), selected_positions),dtype=tf.int32)
            one_hot_selector = tf.one_hot(indexed_positions - 1, boolmask_sum, dtype=tf.float32)
            sampled_indices = tf.cast(tf.tensordot(tf.cast(tf.range(tf.shape(boolmask_indicator)[0]), dtype=tf.float32),
                one_hot_selector,axes=[0, 0]),dtype=tf.int32)
            sampled_indices = tf.reshape(sampled_indices, [-1])
            mask_itemtensor = tf.gather(itemtensor, sampled_indices)

            return mask_itemtensor







