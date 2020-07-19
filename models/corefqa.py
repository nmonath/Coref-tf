#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


import os
import sys 

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import tensorflow as tf
from bert import modeling
from utils import metrics
from bert import tokenization

# TODO (xiaoya): mention linking from the previous mentions 


class CorefQAModel(object):
    def __init__(self, config):
        self.config = config 
        self.dropout = None
        self.pad_idx = 0 
        self.mention_start_idx = 37
        self.mention_end_idx = 42
        self.bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)
        self.bert_config.hidden_dropout_prob = config.dropout_rate

    def get_coreference_resolution_and_loss(self, instance, is_training, use_tpu=False):


        self.use_tpu = use_tpu 
        self.dropout = self.get_dropout(self.config.dropout_rate, is_training)

        flat_window_input_ids, flat_window_input_mask, flat_doc_sentence_map, window_text_len, speaker_ids, gold_starts, gold_ends, gold_cluster_ids = instance
        # flat_input_ids: (num_window, window_size)
        # flat_doc_overlap_input_mask: (num_window, window_size)
        # flat_sentence_map: (num_window, window_size)
        # text_len: dynamic length and is padded to fix length
        # gold_start: (max_num_mention), mention start index in the original (NON-OVERLAP) document. Pad with -1 to the fix length max_num_mention.
        # gold_end: (max_num_mention), mention end index in the original (NON-OVERLAP) document. Pad with -1 to the fix length max_num_mention.
        # cluster_ids/speaker_ids is not used in the mention proposal model.

        flat_window_input_ids = tf.math.maximum(flat_window_input_ids, tf.zeros_like(flat_window_input_ids, tf.int32)) # (num_window * window_size)
        
        flat_doc_overlap_input_mask = tf.where(tf.math.greater_equal(flat_window_input_mask, 0), 
            x=tf.ones_like(flat_window_input_mask, tf.int32), y=tf.zeros_like(flat_window_input_mask, tf.int32)) # (num_window * window_size)
        # flat_doc_overlap_input_mask = tf.math.maximum(flat_doc_overlap_input_mask, tf.zeros_like(flat_doc_overlap_input_mask, tf.int32))
        flat_doc_sentence_map = tf.math.maximum(flat_doc_sentence_map, tf.zeros_like(flat_doc_sentence_map, tf.int32)) # (num_window * window_size)
        
        gold_start_end_mask = tf.cast(tf.math.greater_equal(gold_starts, tf.zeros_like(gold_starts, tf.int32)), tf.bool) # (max_num_mention)
        gold_start_index_labels = self.boolean_mask_1d(gold_starts, gold_start_end_mask, name_scope="gold_starts", use_tpu=self.use_tpu) # (num_of_mention)
        gold_end_index_labels = self.boolean_mask_1d(gold_ends, gold_start_end_mask, name_scope="gold_ends", use_tpu=self.use_tpu) # (num_of_mention)

        gold_cluster_mask = tf.cast(tf.math.greater_equal(gold_cluster_ids, tf.zeros_like(gold_cluster_ids, tf.int32)), tf.bool) # (max_num_cluster)
        glod_cluster_ids = self.boolean_mask_1d(gold_cluster_ids, gold_cluster_mask, name_scope="gold_cluster", use_tpu=self.use_tpu)

        window_text_len = tf.math.maximum(window_text_len, tf.zeros_like(window_text_len, tf.int32)) # (num_of_non_empty_window)
        num_subtoken_in_doc = tf.math.reduce_sum(window_text_len) # the value should be num_subtoken_in_doc 

        mention_input_ids = tf.reshape(flat_window_input_ids, [-1, self.config.window_size]) # (num_window, window_size)
        mention_input_mask = tf.ones_like(mention_input_ids, tf.int32) # (num_window, window_size)
        mention_model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=mention_input_ids, input_mask=mention_input_mask, use_one_hot_embeddings=False, scope='bert')

        mention_doc_overlap_window_embs = mention_model.get_sequence_output() # (num_window, window_size, hidden_size)
        doc_overlap_input_mask = tf.reshape(flat_doc_overlap_input_mask, [self.config.num_window, self.config.window_size]) # (num_window, window_size)

        mention_doc_flat_embs = self.transform_overlap_sliding_windows_to_original_document(mention_doc_overlap_window_embs, doc_overlap_input_mask) 
        mention_doc_flat_embs = tf.reshape(mention_doc_flat_embs, [-1, self.config.hidden_size]) # (num_subtoken_in_doc, hidden_size) 

        candidate_mention_starts = tf.tile(tf.expand_dims(tf.range(num_subtoken_in_doc), 1), [1, self.config.max_span_width])
        candidate_mention_ends = tf.math.add(candidate_mention_starts, tf.expand_dims(tf.range(self.config.max_span_width), 0))
        
        candidate_mention_sentence_start_idx = tf.gather(flat_doc_sentence_map, candidate_mention_starts)
        candidate_mention_sentence_end_idx = tf.gather(flat_doc_sentence_map, candidate_mention_ends)
        
        candidate_mention_mask = tf.logical_and(candidate_mention_ends < num_subtoken_in_doc, tf.equal(candidate_mention_sentence_start_idx, candidate_mention_sentence_end_idx))
        candidate_mention_mask = tf.reshape(candidate_mention_mask, [-1])

        candidate_mention_starts = self.boolean_mask_1d(tf.reshape(candidate_mention_starts, [-1]), candidate_mention_mask, name_scope="candidate_mention_starts", use_tpu=self.use_tpu)
        candidate_mention_ends = self.boolean_mask_1d(tf.reshape(candidate_mention_ends, [-1]), candidate_mention_mask, name_scope="candidate_mention_ends", use_tpu=self.use_tpu)

        candidate_cluster_idx_labels = self.get_candidate_cluster_labels(candidate_mention_starts, candidate_mention_ends, gold_starts, gold_ends, gold_cluster_ids)


        candidate_mention_span_embs, candidate_mention_start_embs, candidate_mention_end_embs = self.get_candidate_span_embedding(
            mention_doc_flat_embs, candidate_mention_starts, candidate_mention_ends) 

        gold_label_candidate_mention_spans, gold_label_candidate_mention_starts, gold_label_candidate_mention_ends = self.get_candidate_mention_gold_sequence_label(
            mention_doc_flat_embs, candidate_mention_starts, candidate_mention_ends, gold_starts, gold_ends, num_subtoken_in_doc)

        mention_proposal_loss, candidate_mention_start_prob, candidate_mention_end_prob, candidate_mention_span_prob, candidate_mention_span_scores = self.get_mention_proposal_score_and_loss(
            candidate_mention_span_embs, candidate_mention_start_embs, candidate_mention_end_embs, gold_label_candidate_mention_spans, 
            gold_label_candidate_mention_starts, gold_label_candidate_mention_ends)

        self.k = tf.minimum(self.config.max_candidate_mentions, tf.to_int32(tf.floor(tf.to_float(num_subtoken_in_doc) * self.config.top_span_ratio)))
        self.c = tf.to_int32(tf.minimum(self.config.max_top_antecedents, k))

        candidate_mention_span_scores = tf.reshape(candidate_mention_span_scores, [-1])
        topk_mention_span_scores, topk_mention_span_indices = tf.nn.top_k(candidate_mention_span_scores, self.k)
        topk_mention_span_indices = tf.reshape(topk_mention_span_indices, [-1])

        topk_mention_start_indices = tf.gather(candidate_mention_starts, topk_mention_span_indices) 
        topk_mention_end_indices = tf.gather(candidate_mention_ends, topk_mention_span_indices)
        topk_mention_span_cluster_ids = tf.gather(candidate_cluster_idx_labels, topk_mention_span_indices)
        topk_mention_span_scores = tf.gather(candidate_mention_span_scores, topk_mention_span_indices)


        i0 = tf.constant(0)
        forward_qa_input_ids = tf.zeros((1, self.config.num_window, self.config.window_size + self.config.max_query_len), dtype=tf.int32)
        forward_qa_input_mask = tf.zeros((1, self.config.num_window, self.config.window_size + self.config.max_query_len), dtype=tf.int32)
        forward_qa_input_token_type_mask = tf.zeros((1, self.config.num_window, self.config.window_size + self.config.max_query_len), dtype=tf.int32)

        # prepare for non-overlap input token ids 
        nonoverlap_doc_input_ids = self.transform_overlap_sliding_windows_to_original_document(flat_window_input_ids, flat_doc_overlap_input_mask)
        overlap_window_input_ids = tf.reshape(flat_window_input_ids, [self.config.num_window, self.config.window_size]) 

        @tf.function
        def forward_qa_mention_linking(i, batch_qa_input_ids, batch_qa_input_mask, batch_qa_input_token_type_mask):
            tmp_mention_start_idx = tf.gather(topk_mention_start_indices, i)
            tmp_mention_end_idx = tf.gather(topk_mention_end_indices, i)

            query_input_token_ids, mention_start_idx_in_sent, mention_end_idx_in_sent = self.get_query_token_ids(
                nonoverlap_doc_input_ids, flat_doc_sentence_map, tmp_mention_start_idx, tmp_mention_end_idx)

            query_pad_token_ids = tf.zeros([self.config.max_query_len - self.get_shape(query_input_token_ids, 0)], dtype=tf.int32)

            pad_query_input_token_ids = tf.concat([query_input_token_ids, query_pad_token_ids], axis=0)
            pad_query_input_token_mask = tf.ones_like(pad_query_input_token_ids, tf.int32)
            pad_query_input_token_type_mask = tf.zeros_like(pad_query_input_token_ids, tf.int32)


            expand_pad_query_input_token_ids = tf.tile(tf.expand_dims(pad_query_input_token_ids, 0), [self.config.num_window, 1])
            expand_pad_query_input_token_mask = tf.tile(tf.expand_dims(pad_query_input_token_mask, 0), [self.config.num_window, 1])
            expand_pad_query_input_token_type_mask = tf.tile(tf.expand_dims(pad_query_input_token_type_mask, 0), [self.config.num_window, 1])


            query_context_input_token_ids = tf.concat([expand_pad_query_input_token_ids, overlap_window_input_ids], axis=1)
            query_context_input_token_mask = tf.concat([expand_pad_query_input_token_mask, tf.ones_like(overlap_window_input_ids, tf.int32)], axis=1)
            query_context_input_token_type_mask = tf.concat([expand_pad_query_input_token_type_mask, tf.ones_like(overlap_window_input_ids, tf.int32)], axis=1)


            return [tf.math.add(i, 1), tf.concat([batch_qa_input_ids, query_context_input_token_ids], 0), 
                    tf.concat([batch_qa_input_mask, query_context_input_token_mask], 0), 
                    tf.concat([batch_qa_input_token_type_mask, query_context_input_token_type_mask], 0)]



        _, stack_forward_qa_input_ids, stack_forward_qa_input_mask, stack_forward_qa_input_type_mask = tf.while_loop(
            cond=lambda i, o1, o2, o3 : i < self.k,
            body=forward_qa_mention_linking, 
            loop_vars=[i0, forward_qa_input_ids, forward_qa_input_mask, forward_qa_input_token_type_mask], 
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, None, None]), 
                tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None])])


        batch_forward_qa_input_ids = tf.reshape(stack_forward_qa_input_ids, [-1, self.config.max_query_len+self.config.window_size])
        batch_forward_qa_input_mask = tf.reshape(stack_forward_qa_input_mask, [-1, self.config.max_query_len+self.config.window_size])
        batch_forward_qa_input_type_mask = tf.reshape(stack_forward_qa_input_type_mask, [-1, self.config.max_query_len+self.config.window_size])

        forward_qa_linking_model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=batch_forward_qa_input_ids, input_mask=batch_forward_qa_input_mask, 
            token_type_ids=batch_forward_qa_input_type_mask, use_one_hot_embeddings=False, 
            scope="bert")

        forward_qa_overlap_window_embs = forward_qa_linking_model.get_sequence_output()
        
        


    def get_query_token_ids(self, nonoverlap_doc_input_ids, sentence_map, mention_start_idx, mention_end_idx, paddding=True):
        """
        Desc:
            pass
        Args:
            pass  
        """
        nonoverlap_doc_input_ids = tf.reshape(nonoverlap_doc_input_ids, [-1])

        sentence_idx_for_mention = tf.gather(sentence_map, mention_start_idx)
        sentence_mask_for_mention = tf.math.equal(sentence_map, sentence_idx_for_mention)
        query_token_input_ids = self.boolean_mask_1d(nonoverlap_doc_input_ids, sentence_mask_for_mention, name_scope="query_mention", use_tpu=self.use_tpu)

        mention_start_in_sent = None 
        mention_end_in_sent = None 

        return query_token_input_ids, mention_start_in_sent, mention_end_in_sent 



    def get_mention_proposal_score_and_loss(self, candidate_mention_span_embs, candidate_mention_start_embs, candidate_mention_end_embs, 
        gold_label_candidate_mention_spans, gold_label_candidate_mention_starts, gold_label_candidate_mention_ends, expect_length_of_labels):

        candidate_mention_span_logits = self.ffnn(candidate_mention_span_embs, self.config.hidden_size*2, 1, dropout=self.dropout, name_scope="mention_span")
        candidate_mention_start_logits = self.ffnn(candidate_mention_start_embs, self.config.hidden_size, 1, dropout=self.dropout, name_scope="mention_start")
        candidate_mention_end_logits = self.ffnn(candidate_mention_end_embs, self.config.hidden_size, 1, dropout=self.dropout, name_scope="mention_end")

        start_loss, candidate_mention_start_probability = self.compute_mention_score_and_loss(candidate_mention_start_logits, gold_label_candidate_mention_starts)
        end_loss, candidate_mention_end_probability = self.compute_mention_score_and_loss(candidate_mention_end_logits, gold_label_candidate_mention_ends)
        span_loss, candidate_mention_span_probability = self.compute_mention_score_and_loss(candidate_mention_span_logits, gold_label_candidate_mention_spans)

        if self.config.mention_proposal_only_concate:
            return span_loss, candidate_mention_span_probability, candidate_mention_span_probability
        else:
            total_loss = start_loss + end_loss + span_loss
            candidate_mention_span_scores = (candidate_mention_start_probability + candidate_mention_end_probability + candidate_mention_span_probability) / 3.0 

            return total_loss, candidate_mention_start_probability, candidate_mention_end_probability, candidate_mention_span_probability, candidate_mention_span_scores


    def compute_mention_score_and_loss(self, pred_sequence_logits, gold_sequence_labels, loss_mask=None):
        """
        Desc:
            compute the unifrom start/end loss and probabilities. 
        Args:
            pred_sequence_logits: (input_shape, 1) 
            gold_sequence_labels: (input_shape, 1)
            loss_mask: [optional] if is not None, it should be (input_shape). should be tf.int32 0/1 tensor. 
            FOR start/end score and loss, input_shape should be num_subtoken_in_doc.
            FOR span score and loss, input_shape should be num_subtoken_in_doc * num_subtoken_in_doc. 
        """
        pred_sequence_probabilities = tf.cast(tf.reshape(tf.sigmoid(pred_sequence_logits), [-1]),tf.float32) # (input_shape)
        expand_pred_sequence_scores = tf.stack([(1 - pred_sequence_logits), pred_sequence_probabilities], axis=-1) # (input_shape, 2)
        expand_gold_sequence_labels = tf.cast(tf.one_hot(tf.reshape(gold_sequence_labels, [-1]), 2, axis=-1), tf.float32) # (input_shape, 2)

        loss = tf.keras.losses.binary_crossentropy(expand_gold_sequence_labels, expand_pred_sequence_scores)
        # loss -> shape is (input_shape)

        if loss_mask is not None:
            loss = tf.multiply(loss, tf.cast(loss_mask, tf.float32))

        total_loss = tf.reduce_mean(loss)
        # total_loss -> a scalar 

        return total_loss, pred_sequence_probabilities


    def get_candidate_span_embedding(self, doc_sequence_embeddings, candidate_span_starts, candidate_span_ends):
        doc_sequence_embeddings = tf.reshape(doc_sequence_embeddings, [-1, self.config.hidden_size])

        span_start_embedding = tf.gather(doc_sequence_embeddings, candidate_span_starts)
        span_end_embedding = tf.gather(doc_sequence_embeddings, candidate_span_ends)
        span_embedding = tf.concat([span_start_embedding, span_end_embedding], 1) 

        return span_embedding, span_start_embedding, span_end_embedding 

    def get_candidate_mention_gold_sequence_label(self, candidate_mention_starts, candidate_mention_ends, 
        gold_start_index_labels, gold_end_index_labels, expect_length_of_labels):

        gold_start_sequence_label = self.scatter_gold_index_to_label_sequence(gold_start_index_labels, expect_length_of_labels)
        gold_end_sequence_label = self.scatter_gold_index_to_label_sequence(gold_end_index_labels, expect_length_of_labels)

        gold_label_candidate_mention_starts = tf.gather(gold_start_sequence_label, candidate_mention_starts)
        gold_label_candidate_mention_ends = tf.gather(gold_end_sequence_label, candidate_mention_ends)

        gold_mention_sparse_label = tf.stack([gold_start_index_labels, gold_end_index_labels], axis=1)
        gold_span_value = tf.reshape(tf.ones_like(gold_start_index_labels, tf.int32), [-1])
        gold_span_shape = tf.constant([expect_length_of_labels, expect_length_of_labels])
        gold_span_label = tf.cast(tf.scatter_nd(gold_mention_sparse_label, gold_span_value, gold_span_shape), tf.int32)

        candidate_mention_spans = tf.stack([candidate_mention_starts, candidate_mention_ends], axis=1)
        gold_label_candidate_mention_spans = tf.gather_nd(gold_span_label, tf.expand_dims(candidate_mention_spans, 1))
 
        return gold_label_candidate_mention_spans, gold_label_candidate_mention_starts, gold_label_candidate_mention_ends


    def scatter_gold_index_to_label_sequence(self, gold_index_labels, expect_length_of_labels):
        """
        Desc:
            transform the mention start/end position index tf.int32 Tensor to a tf.int32 Tensor with 1/0 labels for the input subtoken sequences.
            1 denotes this subtoken is the start/end for a mention. 
        Args:
            gold_index_labels: a tf.int32 Tensor with mention start/end position index in the original document. 
            expect_length_of_labels: the number of subtokens in the original document. 
        """
        gold_labels_pos = tf.reshape(gold_index_labels, [-1, 1]) # (num_of_mention, 1)
        gold_value = tf.reshape(tf.ones_like(gold_index_labels), [-1]) # (num_of_mention)
        label_shape = tf.Variable(expect_length_of_labels) 
        label_shape = tf.reshape(label_shape, [1]) # [1]
        gold_label_sequence = tf.cast(tf.scatter_nd(gold_labels_pos, gold_value, label_shape), tf.int32) # (num_subtoken_in_doc)
        return gold_label_sequence


    def scatter_span_sequence_labels(self, gold_start_index_labels, gold_end_index_labels, expect_length_of_labels):
        """
        Desc:
            transform the mention (start, end) position pairs to a span matrix gold_span_sequence_labels. 
                matrix[i][j]: whether the subtokens between the position $i$ to $j$ can be a mention.  
                if matrix[i][j] == 0: from $i$ to $j$ is not a mention. 
                if matrix[i][j] == 1: from $i$ to $j$ is a mention.
        Args:
            gold_start_index_labels: a tf.int32 Tensor with mention start position index in the original document. 
            gold_end_index_labels: a tf.int32 Tensor with mention end position index in the original document. 
            expect_length_of_labels: a scalar, should be the same with num_subtoken_in_doc
        """ 
        gold_span_index_labels = tf.stack([gold_start_index_labels, gold_end_index_labels], axis=1) # (num_of_mention, 2)
        gold_span_value = tf.reshape(tf.ones_like(gold_start_index_labels, tf.int32), [-1]) # (num_of_mention)
        gold_span_label_shape = tf.Variable([expect_length_of_labels, expect_length_of_labels]) 
        gold_span_label_shape = tf.reshape(gold_span_label_shape, [-1])

        gold_span_sequence_labels = tf.cast(tf.scatter_nd(gold_span_index_labels, gold_span_value, gold_span_label_shape), tf.int32) # (num_subtoken_in_doc, num_subtoken_in_doc)
        return gold_span_sequence_labels 


    def get_candidate_cluster_labels(self, candidate_mention_starts, candidate_mention_ends, 
            gold_mention_starts, gold_mention_ends, gold_cluster_ids):
        """
        Desc:
            pass 
        Args:
            pass 
        """
        same_mention_start = tf.equal(tf.expand_dims(gold_mention_starts, 1), tf.expand_dims(candidate_mention_starts, 0))
        same_mention_end = tf.equal(tf.expand_dims(gold_mention_ends, 1), tf.expand_dims(candidate_mention_ends, 0)) 
        same_mention_span = tf.logical_and(same_mention_start, same_mention_end)
        
        candidate_cluster_idx_labels = tf.matmul(tf.expand_dims(gold_cluster_ids, 0), tf.to_int32(same_mention_span))  # [1, num_candidates]
        candidate_cluster_idx_labels = tf.squeeze(candidate_cluster_idx_labels, 0)  # [num_candidates]

        return candidate_cluster_idx_labels 


    def transform_overlap_sliding_windows_to_original_document(self, overlap_window_inputs, overlap_window_mask):
        """
        Desc:
            hidden_size should be equal to embeddding_size. 
        Args:
            doc_overlap_window_embs: (num_window, window_size, hidden_size). 
                the output of (num_window, window_size) input_ids forward into BERT model. 
            doc_overlap_input_mask: (num_window, window_size). A tf.int32 Tensor contains 0/1. 
                0 represents token in this position should be neglected. 1 represents token in this position should be reserved. 
        """
        ones_input_mask = tf.ones_like(overlap_window_mask, tf.int32) # (num_window, window_size)
        cumsum_input_mask = tf.math.cumsum(ones_input_mask, axis=1) # (num_window, window_size)
        offset_input_mask = tf.tile(tf.expand_dims(tf.range(self.config.num_window) * self.config.window_size, 1), [1, self.config.window_size]) # (num_window, window_size)
        offset_cumsum_input_mask = offset_input_mask + cumsum_input_mask # (num_window, window_size)
        global_input_mask = tf.math.multiply(ones_input_mask, offset_cumsum_input_mask) # (num_window, window_size)
        global_input_mask = tf.reshape(global_input_mask, [-1]) # (num_window * window_size)
        global_input_mask_index = self.boolean_mask_1d(global_input_mask, tf.math.greater(global_input_mask, tf.zeros_like(global_input_mask, tf.int32))) # (num_subtoken_in_doc)

        overlap_window_inputs = tf.reshape(overlap_window_inputs, [self.config.num_window * self.config.window_size, -1]) # (num_window * window_size, hidden_size)
        original_doc_inputs = tf.gather(overlap_window_inputs, global_input_mask_index)  # (num_subtoken_in_doc, hidden_size)

        return original_doc_inputs


    def ffnn(self, inputs, hidden_size, output_size, dropout=None, name_scope="fully-conntected-neural-network",
        hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
        """
        Desc:
            fully-connected neural network. 
            transform non-linearly the [input] tensor with [hidden_size] to a fix [output_size] size.  
        Args: 
            hidden_size: should be the size of last dimension of [inputs]. 
        """
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            hidden_weights = tf.get_variable("hidden_weights", [hidden_size, output_size],
                initializer=hidden_initializer)
            hidden_bias = tf.get_variable("hidden_bias", [output_size], initializer=tf.zeros_initializer())
            outputs = tf.nn.relu(tf.nn.xw_plus_b(inputs, hidden_weights, hidden_bias))

            if dropout is not None:
                outputs = tf.nn.dropout(outputs, dropout)
        return outputs


    def boolean_mask_1d(self, itemtensor, boolmask_indicator, name_scope="boolean_mask1d", use_tpu=False):
        """
        Desc:
            the same functionality of tf.boolean_mask. 
            The tf.boolean_mask operation is not available on the cloud TPU. 
        Args:
            itemtensor : a Tensor contains [tf.int32, tf.float32] numbers. Should be 1-Rank.
            boolmask_indicator : a tf.bool Tensor. Should be 1-Rank. 
            scope : name scope for the operation. 
            use_tpu : if False, return tf.boolean_mask.  
        """
        with tf.name_scope(name_scope):
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


    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)


    def get_shape(self, x, dim):
        """
        Desc:
            return the size of input x in DIM. 
        """ 
        return x.get_shape()[dim].value or tf.shape(x)[dim]













