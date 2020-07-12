#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


import json
import math
import os
import sys 
import random
import threading

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import metrics
import tensorflow as tf
from bert import modeling
from bert import tokenization

# TODO (xiaoya): mention linking from the previous mentions 


class CorefModel(object):
    def __init__(self, config):
        self.config = config
        self.max_segment_len = config['max_segment_len']
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.subtoken_maps = {}
        self.gold = {}
        self.eval_data = None  # Load eval data lazily.
        self.dropout = None
        self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
        self.tokenizer = tokenization.FullTokenizer(vocab_file=config['vocab_file'], do_lower_case=False)

        self.pad_idx = 0 
        self.mention_start_idx = 37
        self.mention_end_idx = 42

        self.coref_evaluator = metrics.CorefEvaluator()

    def get_predictions_and_loss(self, input_ids, input_mask, text_len, speaker_ids, 
            genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, span_mention):
        """
        Desc:
            input_mask: (max_sent_len, max_seg_len), 如果input_mask[i] > 0, 说明了当前位置的token是组成最终doc里面的一部分。
                如果input_mask[i] < 0, 说明了当前位置的token是overlap里面的词语，或者是speaker的补充词。
            e.g.: [[-3, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
            [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3]]
        """

        input_ids = tf.where(tf.cast(tf.math.greater_equal(input_ids, tf.zeros_like(input_ids)),tf.bool), x=input_ids, y=tf.zeros_like(input_ids)) 
        input_mask = tf.where(tf.cast(tf.math.greater_equal(input_mask, tf.zeros_like(input_mask)), tf.bool), x=input_mask, y=tf.zeros_like(input_mask)) 
        text_len = tf.where(tf.cast(tf.math.greater_equal(text_len, tf.zeros_like(text_len)), tf.bool), x= text_len, y=tf.zeros_like(text_len)) 
        speaker_ids = tf.where(tf.cast(tf.math.greater_equal(speaker_ids, tf.zeros_like(speaker_ids)),tf.bool), x=speaker_ids, y=tf.zeros_like(speaker_ids)) 
        gold_starts = tf.where(tf.cast(tf.math.greater_equal(gold_starts, tf.zeros_like(gold_starts)),tf.bool), x=gold_starts, y=tf.zeros_like(gold_starts)) 
        gold_ends = tf.where(tf.cast(tf.math.greater_equal(gold_ends, tf.zeros_like(gold_ends)),tf.bool), x=gold_ends, y=tf.zeros_like(gold_ends) ) 
        cluster_ids = tf.where(tf.cast(tf.math.greater_equal(cluster_ids, tf.zeros_like(cluster_ids)),tf.bool), x=cluster_ids, y=tf.zeros_like(cluster_ids)) 


        input_ids = tf.reshape(input_ids, [-1, self.config["max_segment_len"]])
        input_mask  = tf.reshape(input_mask, [-1, self.config["max_segment_len"]])
        text_len = tf.reshape(text_len, [-1])
        speaker_ids = tf.reshape(speaker_ids, [-1, self.config["max_segment_len"]])
        sentence_map = tf.reshape(sentence_map, [-1])
        cluster_ids = tf.reshape(cluster_ids, [-1]) 
        gold_starts = tf.reshape(gold_starts, [-1]) 
        gold_ends = tf.reshape(gold_ends, [-1]) 
        span_mention = tf.reshape(span_mention, [self.config["max_training_sentences"], self.config["max_segment_len"] * self.config["max_segment_len"]])

        self.input_ids = input_ids # (max_sent_len, max_seg_len)
        self.input_mask = input_mask  # (max_sent_len, max_seg_len)
        self.sentence_map = sentence_map 
        flat_input_mask = tf.ones_like(input_ids, tf.int32)

        model = modeling.BertModel(
            config=self.bert_config, 
            is_training=is_training, 
            input_ids = input_ids, 
            input_mask = flat_input_mask, 
            use_one_hot_embeddings=False, 
            scope="mention_proposal") # original is bert 
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

        doc_seq_emb = model.get_sequence_output() # (max_sentence_len, max_seg_len)
        doc_seq_emb, doc_overlap_mask = self.flatten_emb_by_sentence(doc_seq_emb, input_mask) # (max_sent_len * )
        # why add boolean mask 
        # doc_seq_emb = self.boolean_mask_2d(doc_seq_emb, input_mask)
        # doc_seq_emb = self.boolean_mask_1d(doc_seq_emb, input_mask)

        doc_seq_emb = tf.reshape(doc_seq_emb, [-1, self.config["hidden_size"]])


        num_words = self.shape(doc_seq_emb, 0) # true words in one document  # senten_map 
        # num_words is smaller than the max_sentence_len * max_segment_len
        # candidate_span: 
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width])
        candidate_ends = tf.math.add(candidate_starts, tf.expand_dims(tf.range(self.max_span_width), 0))

        ######### sentence_map = self.boolean_mask_1d(tf.reshape(sentence_map, [-1]), doc_overlap_mask, use_tpu=self.config["tpu"])
        sentence_map = tf.reshape(sentence_map, [-1])

        candidate_start_sentence_indices = tf.gather(sentence_map, candidate_starts)
        candidate_end_sentence_indices = tf.gather(sentence_map, tf.minimum(candidate_ends, num_words - 2))

        # [num_words, max_span_width], 合法的span需要满足start/end不能越界；start/end必须在同一个句子里
        candidate_mask = tf.logical_and(candidate_ends < num_words,
                                        tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices))

        flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
        candidate_starts = self.boolean_mask_1d(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask, use_tpu=self.config["tpu"] )
        candidate_ends = self.boolean_mask_1d(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask, use_tpu=self.config["tpu"] )
        # add an assert that the length of candidate_starts must equal to the length of candidate_ends 
        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids)

        # candidate_binary_labels = candidate_cluster_ids > 0 
        # [num_candidates, emb] -> 候选答案的向量表示
        # [num_candidates, ] -> 候选答案的得分

        candidate_span_emb, candidate_start_emb, candidate_end_emb = self.get_span_emb(doc_seq_emb, candidate_starts, candidate_ends) # (candidate_mention, embedding)
        gold_candidate_mention_label = self.get_candidate_span_label(gold_starts, gold_ends, candidate_starts, candidate_ends, num_words)

        if self.config["mention_proposal_only_concate"]:
            candidate_mention_scores = self.get_mention_scores(span_emb = candidate_span_emb, span_name = "mention_proposal")
            mention_proposal_loss = self.get_mention_proposal_loss(candidate_mention_scores, gold_candidate_mention_label, self.config["mention_proposal_only_concate"])
        else:
            candidate_mention_scores, candidate_start_scores, candidate_end_scores = self.get_mention_scores(span_emb = candidate_span_emb, span_name = "mention_proposal", \
                                                           start_emb = candidate_start_emb, start_name = "mention_starts", \
                                                           end_emb = candidate_end_emb, end_name = "mention_ends")
            candidate_gold_starts = tf.gather(gold_starts, candidate_starts)
            candidate_gold_ends = tf.gather(gold_ends, candidate_ends)
            mention_proposal_loss = self.get_mention_proposal_loss(candidate_mention_scores, gold_candidate_mention_label, self.config["mention_proposal_only_concate"],
                candidate_start_scores=candidate_start_scores, candidate_end_scores=candidate_end_scores,
                gold_starts=candidate_gold_starts, gold_ends=candidate_gold_ends)

        # beam size 所有span的数量小于num_words * top_span_ratio
        k = tf.minimum(self.config["max_candidate_mentions"], tf.to_int32(tf.floor(tf.to_float(num_words) * self.config["top_span_ratio"])))
        c = tf.to_int32(tf.minimum(self.config["max_top_antecedents"], k))  # 初筛挑出0.4*500=200个候选，细筛再挑出50个候选
        candidate_mention_scores = tf.reshape(candidate_mention_scores, [-1])

        top_span_scores, top_span_indices = tf.nn.top_k(candidate_mention_scores, k)

        top_span_indices = tf.reshape(top_span_indices, [-1]) # k
        top_span_starts = tf.gather(candidate_starts, top_span_indices) # k 
        top_span_ends = tf.gather(candidate_ends, top_span_indices)
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        # top_span_emb = tf.gather(candidate_span_emb, top_span_indices)
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k] 

        self.topk_span_starts = top_span_starts 
        self.topk_span_ends = top_span_ends 

        i0 = tf.constant(0)
        num_forward_question = k 
        batch_qa_input_ids = tf.zeros((1, self.config["max_training_sentences"], tf.math.add(self.config["max_segment_len"], self.config["max_query_len"])), dtype=tf.int32)
        batch_qa_input_mask = tf.zeros((1, self.config["max_training_sentences"], tf.math.add(self.config["max_segment_len"], self.config["max_query_len"])), dtype=tf.int32)
        batch_qa_input_token_type_mask = tf.zeros((1, self.config["max_training_sentences"],  tf.math.add(self.config["max_segment_len"], self.config["max_query_len"])), dtype=tf.int32)
        # i0, batch_qa_input_ids, batch_qa_input_mask, batch_qa_input_token_type_mask
        batch_query_ids = tf.zeros((1, self.config["max_query_len"]), dtype=tf.int32)

        @tf.function
        def forward_qa_loop(i, link_qa_input_ids, link_qa_input_mask, link_qa_input_type_mask, link_qa_query_ids):
            ######tmp_context_input_ids = tf.reshape(self.input_ids,[-1, self.config["max_segment_len"]])  
            # (max_train_sent, max_segment_len)
            ######tmp_context_input_mask = tf.reshape(self.input_mask, [-1, self.config["max_segment_len"]]) 
            # tf.ones_like(tmp_context_input_ids) 
            ######actual_mask = tf.cast(tf.not_equal(self.input_mask, self.pad_idx), tf.int32)  
            # (max_train_sent, max_segment_len) 
            # def get_question_token_ids(self, input_ids, input_mask, sentence_map, top_start, top_end, special=True)
            question_tokens, start_in_sentence, end_in_sentence = self.get_question_token_ids(
                self.input_ids, self.input_mask, self.sentence_map, tf.gather(top_span_starts, i), tf.gather(top_span_ends, i))

            pad_tokens = tf.zeros([self.config["max_query_len"] - self.shape(question_tokens, 0)], dtype=tf.int32)
            pad_query_tokens = tf.concat([question_tokens, pad_tokens], axis=0)
            pad_query_token_mask = tf.ones_like(pad_query_tokens, dtype=tf.int32)
       
            batch_query_tokens = tf.tile(tf.expand_dims(pad_query_tokens, 0), tf.constant([self.config["max_training_sentences"], 1])) 
            # batch_pad_question_tokens: (max_training_sentences, max_query_len)
            batch_query_token_type_mask = tf.zeros_like(batch_query_tokens)
            batch_query_token_mask = tf.tile(tf.expand_dims(pad_query_token_mask, 0), tf.constant([self.config["max_training_sentences"], 1])) 

            batch_context_tokens = tf.reshape(self.input_ids, [self.config["max_training_sentences"], self.config["max_segment_len"]])
            batch_context_token_type_mask = tf.ones_like(batch_context_tokens) # max_train_sent, max_segment_len 
            batch_context_token_mask = tf.ones_like(batch_context_tokens)


            batch_qa_input_ids = tf.concat([batch_query_tokens, batch_context_tokens], -1)
            batch_qa_input_token_type_mask = tf.concat([batch_query_token_type_mask, batch_context_token_type_mask], -1)
            batch_qa_input_mask = tf.concat([batch_query_token_mask, batch_context_token_mask], -1)

            batch_qa_input_ids = tf.cast(tf.reshape(batch_qa_input_ids, [1, self.config["max_training_sentences"], tf.math.add(self.config["max_segment_len"], self.config["max_query_len"])]), tf.int32)
            batch_qa_input_token_type_mask = tf.cast(tf.reshape(batch_qa_input_token_type_mask, [1, self.config["max_training_sentences"], tf.math.add(self.config["max_segment_len"] ,self.config["max_query_len"])]), tf.int32)
            batch_qa_input_mask = tf.cast(tf.reshape(batch_qa_input_mask, [1, self.config["max_training_sentences"], tf.math.add(self.config["max_segment_len"], self.config["max_query_len"])]), tf.int32)
            pad_query_tokens = tf.cast(tf.reshape(pad_query_tokens, [1, self.config["max_query_len"]]), tf.int32)

            # link_qa_input_ids, link_qa_input_mask, link_qa_input_type_mask
            return [tf.math.add(i, 1), tf.concat([link_qa_input_ids, batch_qa_input_ids], 0), 
                tf.concat([link_qa_input_mask, batch_qa_input_mask], 0), 
                tf.concat([link_qa_input_type_mask, batch_qa_input_token_type_mask], 0), 
                tf.concat([link_qa_query_ids, pad_query_tokens], 0)]


        _, forward_qa_input_ids, forward_qa_input_mask, forward_qa_input_token_type_mask, qa_topk_query_tokens = tf.while_loop(
            cond=lambda i, o1, o2, o3, o4 : i < k, 
            body=forward_qa_loop, 
            loop_vars=[i0, batch_qa_input_ids, batch_qa_input_mask, batch_qa_input_token_type_mask, batch_query_ids], 
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, None, None]), 
                tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None]), 
                tf.TensorShape([None, None])])

        # forward_qa_input_ids -> (k, max_train_sent, max_query_len + max_segment_len) -> (k * max_train_sent, max_query_len + max_segment_len)
        forward_qa_input_ids = tf.reshape(forward_qa_input_ids, [-1, tf.math.add(self.config["max_query_len"], self.config["max_segment_len"])]) 
        forward_qa_input_mask = tf.reshape(forward_qa_input_mask, [-1, tf.math.add(self.config["max_query_len"], self.config["max_segment_len"])]) 
        forward_qa_input_token_type_mask = tf.reshape(forward_qa_input_token_type_mask, [-1]) # self.config["max_query_len"] + self.config["max_segment_len"]]) 

        forward_bert_qa_model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=forward_qa_input_ids, input_mask=forward_qa_input_mask, 
            token_type_ids=forward_qa_input_token_type_mask, use_one_hot_embeddings=False, 
            scope="forward_qa_linking")

        forward_qa_emb = forward_bert_qa_model.get_sequence_output() # (k * max_train_sent, max_query_len + max_segment_len, hidden_size)
        forward_qa_input_token_type_mask_bool = tf.cast(tf.reshape(forward_qa_input_token_type_mask, [-1, tf.math.add(self.config["max_query_len"] , self.config["max_segment_len"])]), tf.bool)
        
        forward_qa_emb = tf.reshape(forward_qa_emb, [-1, tf.math.add(self.config["max_query_len"], self.config["max_segment_len"]), self.config["hidden_size"]])
        forward_qa_input_token_type_mask_bool = tf.reshape(forward_qa_input_token_type_mask_bool, [-1, tf.math.add(self.config["max_query_len"], self.config["max_segment_len"])])

        forward_doc_emb = self.boolean_mask_2d(forward_qa_emb, forward_qa_input_token_type_mask_bool,  dims=2)
        # (k * max_train_sent, max_segment_len, hidden_size)
        forward_doc_emb = tf.reshape(forward_doc_emb, [-1, self.config["hidden_size"]]) 
        # (k * max_train_sent * max_segment_len, hidden_size)
        flat_sentence_map = tf.tile(tf.expand_dims(sentence_map, 0), [k, 1]) # (k, max_sent * max_segment) 
        flat_sentence_map = tf.reshape(flat_sentence_map, [-1])
        flat_sentence_map = tf.where(tf.cast(tf.math.greater_equal(flat_sentence_map, tf.zeros_like(flat_sentence_map)),tf.bool), x=flat_sentence_map, y=tf.zeros_like(flat_sentence_map)) 

        flat_forward_doc_emb = self.boolean_mask_1d(forward_doc_emb, flat_sentence_map, use_tpu=self.config["tpu"])
        # flat_forward_doc_emb -> (k * non_overlap_doc_len * hidden_size)
        flat_forward_doc_emb = tf.reshape(flat_forward_doc_emb, [k, -1, self.config["hidden_size"]])
        # flat_forward_doc_emb -> (k, non_overlap_doc_len, hidden_size)
        non_overlap_doc_len = self.shape(flat_forward_doc_emb, 1)
        top_span_starts = tf.reshape(top_span_starts, [-1])
        top_span_ends = tf.reshape(top_span_ends, [-1])

        forward_candidate_starts = tf.reshape(tf.tile(tf.expand_dims(candidate_starts, 0), [k, 1]), [k, -1]) # (k, num_candidates)
        forward_candidate_ends = tf.reshape(tf.tile(tf.expand_dims(candidate_ends, 0), [k, 1]), [k, -1]) # (k, num_candidates)
        num_candidate = self.shape(candidate_starts, 0)
        forward_pos_offset = tf.cast(tf.tile(tf.reshape(tf.range(0, num_candidate) * non_overlap_doc_len, [-1, 1]), [1, k]), tf.int32)
        forward_pos_offset = tf.reshape(forward_pos_offset, [k, -1])

        forward_candidate_starts = tf.reshape(tf.cast(tf.math.add(forward_candidate_starts, forward_pos_offset), tf.int32), [-1])
        forward_candidate_ends = tf.reshape(tf.cast(tf.math.add(forward_candidate_ends, forward_pos_offset), tf.int32), [-1])

        forward_mention_start_emb = tf.gather(tf.reshape(flat_forward_doc_emb, [-1, self.config["hidden_size"]]), forward_candidate_starts) # (k, k, emb)
        forward_mention_end_emb = tf.gather(tf.reshape(flat_forward_doc_emb, [-1, self.config["hidden_size"]]), forward_candidate_ends)

        forward_mention_start_emb = tf.reshape(forward_mention_start_emb, [-1, self.config["hidden_size"]])
        forward_mention_end_emb = tf.reshape(forward_mention_end_emb, [-1, self.config["hidden_size"]])
        forward_mention_span_emb = tf.concat([forward_mention_start_emb, forward_mention_end_emb], 1) # (k, k emb * 2) 

        forward_mention_span_emb = tf.reshape(forward_mention_span_emb, [-1, self.config["hidden_size"]*2])
        ### with tf.variable_scope("forward_qa"):
        forward_mention_ij_score = self.get_mention_scores(forward_mention_span_emb, "forward")
        #get_mention_scores(self, span_emb, span_name, start_emb=None,start_name=None , end_emb=None , end_name=None):


        #forward_mention_ij_score = self.ffnn(forward_mention_span_emb, 1, self.config["hidden_size"]*2, 1, self.dropout)
        forward_mention_ij_score = tf.reshape(forward_mention_ij_score, [k, -1])

        topc_forward_scores, topc_forward_indices = tf.nn.top_k(forward_mention_ij_score, c, sorted=False)
        # topc_forward_scores, topc_forward_indices : [k, c]

        flat_topc_forward_indices = tf.reshape(topc_forward_indices, [-1])

        topc_start_index_doc = tf.gather(top_span_starts, flat_topc_forward_indices) # (k * c)
        topc_end_index_doc = tf.gather(top_span_ends, flat_topc_forward_indices) # (k * c)
        topc_span_scores = tf.gather(top_span_mention_scores, flat_topc_forward_indices) # (k*c)
        topc_span_cluster_ids = tf.gather(top_span_cluster_ids, flat_topc_forward_indices)

        # link_qa_input_ids, link_qa_input_mask, link_qa_input_type_mask, link_qa_query_ids
        backward_qa_input_ids = tf.zeros((1, tf.math.add(self.config["max_query_len"],self.config["max_context_len"])), dtype=tf.int32)
        backward_qa_input_mask = tf.zeros((1, tf.math.add(self.config["max_query_len"], self.config["max_context_len"])), dtype=tf.int32)
        backward_qa_input_token_type = tf.zeros((1, tf.math.add(self.config["max_query_len"], self.config["max_context_len"])), dtype=tf.int32)
        backward_start_in_sent = tf.zeros((1), dtype=tf.int32)
        backward_end_in_sent = tf.zeros((1), dtype=tf.int32)

        tile_top_span_starts = tf.reshape(tf.tile(tf.reshape(self.topk_span_starts, [1, -1]), [c, 1]), [-1])
        tile_top_span_ends = tf.reshape(tf.tile(tf.reshape(self.topk_span_ends, [1, -1]), [c, 1]), [-1])

        # backward_qa_input_ids, backward_qa_input_mask, backward_qa_input_token_type, backward_start_in_sent, backward_end_in_sent
        backward_qa_input_ids = tf.zeros([1, tf.math.add(self.config["max_query_len"], self.config["max_context_len"])], dtype=tf.int32)
        backward_qa_input_mask = tf.zeros([1, tf.math.add(self.config["max_query_len"], self.config["max_context_len"])], dtype=tf.int32)
        backward_qa_input_token_type = tf.zeros([1, tf.math.add(self.config["max_query_len"], self.config["max_context_len"])], dtype=tf.int32)
        tmp_start_in_sent = tf.convert_to_tensor(tf.constant([0]), dtype=tf.int32)
        tmp_end_in_sent = tf.convert_to_tensor(tf.constant([0]), dtype=tf.int32)
        i0 = tf.constant(0)
        

        @tf.function
        def backward_qa_loop(i, rank_qa_input_ids, rank_qa_input_mask, rank_qa_input_type_mask, start_in_sent, end_in_sent,):
            
            query_tokens, t_start_in_sent, t_end_in_sent = self.get_question_token_ids(
                self.input_ids, self.input_mask, self.sentence_map, tf.gather(topc_start_index_doc, i), tf.gather(topc_end_index_doc, i))
        
            pad_tokens = tf.zeros([self.config["max_query_len"] - self.shape(query_tokens, 0)], dtype=tf.int32)
            pad_query_tokens = tf.concat([query_tokens, pad_tokens], axis=0)
            pad_query_tokens = tf.cast(pad_query_tokens, tf.int32) 

            query_input_token_type_mask = tf.zeros_like(pad_query_tokens, dtype=tf.int32)
            query_input_mask = tf.ones_like(pad_query_tokens, dtype=tf.int32)

            context_tokens, k_start_in_sent, k_end_in_sent = self.get_question_token_ids(
                self.input_ids, self.input_mask, self.sentence_map, tf.gather(tile_top_span_starts, i), tf.gather(tile_top_span_ends, i), special=False)

            pad_tokens = tf.zeros([self.config["max_query_len"] - self.shape(context_tokens, 0)], dtype=tf.int32)
            pad_context_tokens = tf.concat([context_tokens, pad_tokens], axis=0)

            pad_context_tokens = tf.cast(pad_context_tokens, tf.int32)
            context_input_mask = tf.ones_like(pad_context_tokens, dtype=tf.int32)
            context_input_token_type_mask = tf.ones_like(pad_context_tokens, dtype=tf.int32)

            qa_input_tokens = tf.concat([pad_query_tokens, pad_context_tokens], axis=-1)
            qa_input_mask = tf.concat([query_input_mask, context_input_mask], axis=-1)
            qa_input_token_type_mask = tf.concat([query_input_token_type_mask, context_input_token_type_mask], -1)

            qa_input_tokens = tf.cast(tf.reshape(qa_input_tokens, [1, tf.math.add(self.config["max_query_len"], self.config["max_context_len"])]), tf.int32)
            qa_input_mask = tf.cast(tf.reshape(qa_input_mask, [1, tf.math.add(self.config["max_query_len"], self.config["max_context_len"])]), tf.int32)
            qa_input_token_type_mask = tf.cast(tf.reshape(qa_input_token_type_mask, [1, tf.math.add(self.config["max_query_len"], self.config["max_context_len"])]), tf.int32)
            k_start_in_sent = tf.convert_to_tensor(tf.cast(k_start_in_sent, tf.int32), dtype=tf.int32) 
            k_end_in_sent = tf.convert_to_tensor(tf.cast(k_end_in_sent, tf.int32), dtype=tf.int32) 

            rank_qa_input_ids = tf.cast(rank_qa_input_ids, tf.int32)
            rank_qa_input_mask = tf.cast(rank_qa_input_mask, tf.int32)
            rank_qa_input_type_mask = tf.cast(rank_qa_input_type_mask, tf.int32)
            start_in_sent = tf.cast(start_in_sent, tf.int32)
            end_in_sent = tf.cast(end_in_sent, tf.int32)

            return [tf.math.add(i, 1), tf.concat([rank_qa_input_ids, qa_input_tokens],axis=0),
                tf.concat([rank_qa_input_mask, qa_input_mask], axis=0), 
                tf.concat([rank_qa_input_type_mask, qa_input_token_type_mask], axis=0), 
                tf.concat([start_in_sent, k_start_in_sent], axis=0), 
                tf.concat([end_in_sent, k_end_in_sent], axis=0)]

        _, batch_backward_input_ids, batch_backward_input_mask, batch_backward_token_type_mask, batch_backward_start_sent, batch_backward_end_sent = tf.while_loop(
            cond = lambda i, o1, o2, o3, o4, o5: i < k * c,
            body=backward_qa_loop, 
            loop_vars=[i0, backward_qa_input_ids, backward_qa_input_mask, backward_qa_input_token_type, tmp_start_in_sent, tmp_end_in_sent], 
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
            tf.TensorShape([None]), tf.TensorShape([None])])

        self.batch_backward_start_sent = tf.gather(batch_backward_start_sent, tf.range(1, tf.math.add(k * c, 1)))
        self.batch_backward_end_sent = tf.gather(batch_backward_end_sent, tf.range(1, tf.math.add(k*c, 1)) )

        backward_bert_qa_model = modeling.BertModel(config=self.bert_config, is_training=is_training, 
            input_ids=batch_backward_input_ids, input_mask=batch_backward_input_mask, 
            token_type_ids=batch_backward_token_type_mask, use_one_hot_embeddings=False, 
            scope="backward_qa_linking")

        backward_qa_emb = backward_bert_qa_model.get_sequence_output() # (c*k, num_ques_token+ max_context_len, embedding)
        # 1. (c*k
        backward_qa_input_token_type_mask_bool = tf.cast(batch_backward_token_type_mask ,tf.bool)
        backward_k_sent_emb = self.boolean_mask_2d(backward_qa_emb, backward_qa_input_token_type_mask_bool,  dims=2)
        # backward_k_sent_emb -> (c*k, max_context_len, embedding)

        backward_k_sent_emb =  tf.reshape(backward_k_sent_emb, [-1, self.config["hidden_size"]]) 
        backward_pos_offset = tf.cast(tf.reshape(tf.range(0, k*c) * self.config["max_context_len"], [-1, 1]), tf.int32) 

        batch_backward_start_sent = tf.math.add(tf.reshape(self.batch_backward_start_sent, [-1]) , tf.reshape(backward_pos_offset, [-1]))
        batch_backward_end_sent = tf.math.add(tf.reshape(self.batch_backward_end_sent, [-1]) , tf.reshape(backward_pos_offset, [-1]))


        backward_qa_start_emb = tf.gather(backward_k_sent_emb, tf.reshape(batch_backward_start_sent, [-1])) # (c*k, emb)
        backward_qa_end_emb = tf.gather(backward_k_sent_emb, tf.reshape(batch_backward_end_sent, [-1]))  # (c*k, emb)
        backward_qa_span_emb = tf.concat([backward_qa_start_emb,backward_qa_end_emb], axis=1) # (c*k, 2*emb)

        # with tf.variable_scope("backward_qa"):
        backard_mention_ji_score = self.get_mention_scores(backward_qa_span_emb, "backward")
        #backard_mention_ji_score = self.ffnn(tf.reshape(backward_qa_span_emb, [-1, self.config["hidden_size"]*2]), 1, self.config["hidden_size"]*2, 1, self.dropout)
        # inputs, num_hidden_layers, hidden_size, output_size, dropout,
        # s(j) topc_span_scores # (k*c)
        # s(i) top_span_mention_scores # k
        # topc_forward_scores # (k*c)
        # backard_mention_ji_score # (k*c) 
        tile_top_span_mention_scores = tf.tile(tf.expand_dims(tf.reshape(top_span_mention_scores, [-1]), 1), [1, c])
        
        # top_antecedent_scores = tf.math.add_n([tf.reshape(topc_forward_scores, [-1]), tf.reshape(backard_mention_ji_score, [-1]), \
        #     tf.reshape(topc_span_scores, [-1]),tf.reshape(tile_top_span_mention_scores, [-1])])

        top_antecedent_scores = tf.math.add_n([tf.reshape(topc_forward_scores, [-1]), tf.reshape(backard_mention_ji_score, [-1])]) * self.config["score_ratio"]
        top_antecedent_scores += tf.math.add_n([tf.reshape(topc_span_scores, [-1]), tf.reshape(tile_top_span_mention_scores, [-1])])

        
        top_antecedent_scores = tf.reshape(top_antecedent_scores, [k, c])

        dummy_scores = tf.zeros([k, 1])  # [k, 1]

        ###############################################################################################################################################
        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]
        # top_antecedent_cluster_ids [k, c] 每个mention每个antecedent的cluster_id
        # same_cluster_indicator [k, c] 每个mention跟每个预测的antecedent是否同一个cluster
        # pairwise_labels [k, c] 用pairwise的方法得到的label，非mention、非antecedent都是0，mention跟antecedent共指是1
        # top_antecedent_labels [k, c+1] 最终的标签，如果某个mention没有antecedent就是dummy_label为1

        topc_span_cluster_ids = tf.reshape(topc_span_cluster_ids, [k, c])
        same_cluster_indicator = tf.equal(topc_span_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # (k, c)
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]

        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
        
        loss = self.marginal_likelihood_loss(top_antecedent_scores, top_antecedent_labels)  # [k]

        loss += mention_proposal_loss * self.config["mention_proposal_loss_ratio"]

        return loss, self.topk_span_starts, self.topk_span_ends, top_antecedent_scores 

    def flatten_emb_by_sentence(self, emb, segment_overlap_mask):
        """
        Desc:
            flatten_embeddings_by_sentence_segment_mask
        Args:
            emb: [max_sentence_len, max_segment_len] 
            segment_overlap_mask:  [max_sentence_len, max_segment_len]
        """
        flattened_emb = tf.reshape(emb, [-1, self.config["hidden_size"]])
        flattened_overlap_mask = tf.reshape(segment_overlap_mask, [-1])
        segment_overlap_mask = tf.maximum(segment_overlap_mask, tf.zeros_like(segment_overlap_mask))
        segment_overlap_mask = tf.reshape(segment_overlap_mask, [-1])
        
        # flattened_emb = self.boolean_mask_1d(flattened_emb, segment_overlap_mask, use_tpu=self.config["tpu"])

        return flattened_emb, flattened_overlap_mask 


    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        """
        Desc:
            pass 
        Args:
            candidate_starts/candidate_ends: 
            labeled_starts/labeled_ends: 
            labels: 
        """
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0))
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) 
        same_span = tf.logical_and(same_start, same_end)
        # candidate_labels: [num_candidates] 预测对的candidate标上正确的cluster_id，预测错的标0
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels # 每个候选答案得到真实标注的cluster_id


    def get_span_emb(self, context_outputs, span_starts, span_ends):
        """
        一个span的表示由下面的组成
        span_start_embedding, span_end_embedding, span_with_embedding, head_attention_representation 
        """

        span_emb_list = []
        context_outputs = tf.reshape(context_outputs, [-1, self.config["hidden_size"]])

        span_end_emb = tf.gather(context_outputs, tf.reshape(span_ends, [-1]))
        span_start_emb = tf.gather(context_outputs, tf.reshape(span_starts, [-1])) # [k, emb]
        span_emb_list.append(span_start_emb)
        # span_end_emb = tf.gather(context_outputs, tf.reshape(span_ends, [-1]))
        span_emb_list.append(span_end_emb)
        
        span_width = tf.math.add(1, span_ends) - span_starts # [k]

        if self.config["use_features"]:
            span_width_index = span_width -1 # [k]
            with tf.variable_scope("span_features",):
                span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)), span_width_index)  # [k, emb]
                span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
                span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
            head_attn_reps = tf.matmul(mention_word_scores, context_outputs) # [k, t]
            span_emb_list.append(head_attn_reps)

        span_emb = tf.concat([span_start_emb, span_end_emb], 1) # [k, emb] origin span_emb_list 
        span_emb = tf.reshape(span_emb, [-1, self.config["hidden_size"]*2])
        return span_emb, span_start_emb, span_end_emb # [k, emb]

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = self.shape(encoded_doc, 0) # T 
        num_c = self.shape(span_starts, 0)
        doc_range = tf.tile(tf.expand_dims(tf.range(0, num_words), 0), [num_c, 1]) # [num_candidate, num_words]
        mention_mask = tf.logical_and(doc_range >= tf.expand_dims(span_starts, 1), 
            doc_range <= tf.expand_dims(span_ends, 1)) # [num_candidates, num_word]


        with tf.variable_scope("mention_word_attn",):
            word_attn = tf.squeeze(
                self.projection(encoded_doc, 1, initializer=tf.truncated_normal_initializer(stddev=0.02)), 1)

        word_attn = tf.cast(word_attn,tf.float32)
        mention_word_attn = tf.nn.softmax(tf.math.add(tf.log(tf.to_float(mention_mask)), tf.expand_dims(word_attn, 0)) )

        return mention_word_attn  # [num_candidates, num_words] 


    def get_mention_scores(self, span_emb, span_name, start_emb=None,start_name=None,\
        end_emb=None, end_name=None):
        # get scores for a span 
        mention_scores_lst = []

        if span_name is not None:
            with tf.variable_scope(span_name):
                # use specifc concate_scores
                concate_scores = self.ffnn(span_emb, 1, self.config["hidden_size"]*2, 1, self.dropout)
                mention_scores_lst.append(concate_scores)

        if start_name is not None:
            # we need to use whether a token is the start of a span for score computing
            with tf.variable_scope(start_name):
                start_scores = self.ffnn(start_emb, 1, self.config["hidden_size"], 1, self.dropout)
                mention_scores_lst.append(start_scores)


        if end_name is not None:
            # we need to use whether a token is the end of a span for score computing
            with tf.variable_scope(end_name):
                end_scores = self.ffnn(end_emb, 1, self.config["hidden_size"], 1, self.dropout)
                mention_scores_lst.append(end_scores)

        mention_scores = tf.math.add_n(mention_scores_lst)/len(mention_scores_lst)

        if len(mention_scores_lst) == 1:
            return  tf.squeeze(mention_scores, 1)
        else:
            return mention_scores, start_scores, end_scores 


    def get_question_token_ids(self, input_ids, flat_input_mask, sentence_map, top_start, top_end, special=True, pad=True):
        """
        Desc:
            construct question based on the selected mention 
        Args:
            sentence_map: original sentence_map 
            top_start: start index in non-overlap document 
            top_end: end index in non-overlap document 
        """

        nonoverlap_sentence = tf.where(tf.cast(tf.math.greater_equal(sentence_map, tf.zeros_like(sentence_map)),tf.bool), x=sentence_map, y=tf.zeros_like(sentence_map)) 

        nonoverlap_sentence = self.boolean_mask_1d(tf.reshape(input_ids, [-1]), tf.reshape(nonoverlap_sentence, [-1]), use_tpu=self.config["tpu"])

        flat_sentence_map = tf.reshape(sentence_map, [-1])

        sentence_idx = tf.gather(nonoverlap_sentence, top_start)

        query_sentence_mask = tf.math.equal(flat_sentence_map, sentence_idx)
        input_ids = tf.reshape(input_ids, [-1])
        query_sentence_tokens = self.boolean_mask_1d(input_ids, query_sentence_mask, use_tpu=self.config["tpu"])
        len_query_tokens = self.shape(query_sentence_tokens, 0)

        sentence_start = tf.where(tf.equal(nonoverlap_sentence, tf.gather(query_sentence_tokens, tf.constant(0))))
        # sentence_end = tf.where(tf.equal(nonoverlap_sentence, tf.gather(query_sentence_tokens, len_query_tokens -1 ))) 
        ############### mention_start = tf.where(tf.equal(nonoverlap_sentence, tf.gather(nonoverlap_sentence, top_start)))
        ############### mention_end = tf.where(tf.equal(nonoverlap_sentence, tf.gather(nonoverlap_sentence, top_end)))
        ##### should be flat_input_mask 
        mention_start = tf.cast(top_start, tf.int32) 
        mention_end = tf.cast(top_end, tf.int32) 

        original_tokens = query_sentence_tokens

        tmp_start = mention_start - tf.cast(sentence_start, tf.int32) 
        tmp_end = mention_end - tf.cast(sentence_start, tf.int32) 

        tmp_start = tf.gather(tmp_start, tf.constant(0))
        tmp_end = tf.gather(tmp_end, tf.constant(0))

        ## if pad:
        return original_tokens, tf.reshape(tmp_start, [-1]), tf.reshape(tmp_end, [-1]) 
        # tf.reshape(mention_start, [-1]),  tf.reshape(mention_end, [-1]) 

        sentence_end = tf.where(tf.equal(nonoverlap_sentence, tf.gather(query_sentence_tokens, len_query_tokens -1 ))) 
        mention_start = tf.where(tf.equal(nonoverlap_sentence, tf.gather(nonoverlap_sentence, top_start)))
        mention_end = tf.where(tf.equal(nonoverlap_sentence, tf.gather(nonoverlap_sentence, top_end)))
        mention_start_in_sentence = mention_start - sentence_start
        mention_end_in_sentence = mention_end - sentence_end

        mention_start_in_sentence = tf.reshape(tf.cast(mention_start_in_sentence, tf.int32), [-1])
        mention_end_in_sentence = tf.reshape(tf.cast(mention_end_in_sentence, tf.int32), [-1])

        if special:
            # 补充上special token， 注意start end应该按照这个向后移动一步
            # en_sent = self.shape(original_tokens, 0)
            # before_sent = tf.gather(original_tokens, tf.range(0, mention_start_in_sentence[0]))
            # mid_sent = tf.gather(original_tokens, tf.range(mention_start_in_sentence[0], mention_end_in_sentence[0] + 1))
            # end_sent = tf.gather(original_tokens, tf.range(mention_end_in_sentence[0] + 1, len_sent))

            question_token_ids = tf.concat([tf.cast(original_tokens[: mention_start_in_sentence], tf.int32),
                                        [tf.cast(self.mention_start_idx, tf.int32)],
                                         tf.cast(original_tokens[mention_start_in_sentence: mention_end_in_sentence + 1], tf.int32),
                                         [tf.cast(self.mention_end_idx, tf.int32)],
                                         tf.cast(original_tokens[mention_end_in_sentence + 1:], tf.int32),
                                         ], 0)

            # question_token_ids = tf.concat([before_sent,mid_sent,end_sent], 0)
            return question_token_ids, mention_start_in_sentence , mention_end_in_sentence + 1
        else:
            question_token_ids = original_tokens 
            return question_token_ids, mention_start_in_sentence, mention_end_in_sentence  

    def get_mention_proposal_loss(self, candidate_mention_scores, gold_span_mention, mention_proposal_only_concate,
                candidate_start_scores=None, candidate_end_scores=None,
                gold_starts=None, gold_ends=None):

        probs_span_mention = tf.reshape(tf.sigmoid(candidate_mention_scores), [-1])
        probs_span_mention = tf.stack([(1 - probs_span_mention), probs_span_mention], axis=-1)
        gold_span_mention = tf.reshape(gold_span_mention, [-1])
        gold_span_mention = tf.cast(tf.one_hot(tf.reshape(gold_span_mention, [-1]), 2, axis=-1), tf.float32)

        span_mention_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(gold_span_mention, probs_span_mention))

        if mention_proposal_only_concate:
            return span_mention_loss 
    
        probs_start_mention = tf.reshape(tf.sigmoid(candidate_start_scores), [-1])
        probs_end_mention = tf.reshape(tf.sigmoid(candidate_end_scores), [-1])
        probs_start_mention = tf.stack([(1 - probs_start_mention), probs_start_mention], axis=-1)
        probs_end_mention = tf.stack([(1 - probs_end_mention), probs_end_mention], axis=-1)
        gold_starts = tf.cast(tf.one_hot(tf.reshape(gold_starts, [-1]), 2, axis=-1), tf.float32)
        gold_ends = tf.cast(tf.one_hot(tf.reshape(gold_ends, [-1]), 2, axis=-1), tf.float32)

        start_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(gold_starts, probs_start_mention,))
        end_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(gold_ends, probs_end_mention))

        total_loss = start_loss + end_loss + span_mention_loss
        return total_loss 

    def marginal_likelihood_loss(self, antecedent_scores, antecedent_labels):
        """
        Desc:
            marginal likelihood of gold antecedent spans form coreference cluster 
        Args:
            antecedent_scores: [k, c+1] the predicted scores by the model
            antecedent_labels: [k, c+1] the gold-truth cluster labels
        Returns:
            a scalar of loss 
        """
        gold_scores = tf.math.add(antecedent_scores, tf.log(tf.to_float(antecedent_labels)))
        marginalized_gold_scores = tf.math.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.math.reduce_logsumexp(antecedent_scores, [1])  # [k]
        loss = log_norm - marginalized_gold_scores  # [k]
        return tf.math.reduce_sum(loss)

    def get_dropout(self, dropout_rate, is_training):  # is_training为True时keep=1-drop, 为False时keep=1
        
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def get_top_span_cluster_ids(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels, top_span_indices):

        """
        method to get top_span_cluster_ids
        :param candidate_starts: [num_candidates, ]
        :param candidate_ends: [num_candidates, ]
        :param labeled_starts: [num_mentions, ]
        :param labeled_ends: [num_mentions, ]
        :param labels: [num_mentions, ] gold truth cluster ids
        :param top_span_indices: [k, ]
        :return: [k, ] ground truth cluster ids for each proposed candidate span
        """
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0))
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0))
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates] predict_i == label_j

        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        top_span_cluster_ids = tf.gather(candidate_labels, top_span_indices)
        return top_span_cluster_ids


    def boolean_mask_1d(self, itemlist, indicator, fields=None, scope=None, 
        use_static_shapes=False, indicator_sum=None, use_tpu=True, dims=1):
        """Select boxes from BoxList according to indicator and return new BoxList.
        `boolean_mask` returns the subset of boxes that are marked as "True" by the
        indicator tensor. By default, `boolean_mask` returns boxes corresponding to
        the input index list, as well as all additional fields stored in the boxlist
        (indexing into the first dimension).  However one can optionally only draw
        from a subset of fields.
        Args:
            boxlist: BoxList holding N boxes
            indicator: a rank-1 boolean tensor
            fields: (optional) list of fields to also gather from.  If None (default),
                all fields are gathered from.  Pass an empty fields list to only gather
                the box coordinates.
            scope: name scope.
            use_static_shapes: Whether to use an implementation with static shape
                gurantees.
            indicator_sum: An integer containing the sum of `indicator` vector. Only
            required if `use_static_shape` is True.
        Returns:
        subboxlist: a BoxList corresponding to the subset of the input BoxList
        specified by indicator
        Raises:
        ValueError: if `indicator` is not a rank-1 boolean tensor.
        """

        # if not use_tpu:
        #  return tf.boolean_mask(itemlist, indicator)
        with tf.name_scope(scope, 'BooleanMask'):
            indicator_sum = tf.reduce_sum(tf.cast(indicator, tf.int32))

            selected_positions = tf.cast(indicator, dtype=tf.float32)
            indexed_positions = tf.cast(tf.multiply(tf.cumsum(selected_positions), selected_positions),dtype=tf.int32)
            one_hot_selector = tf.one_hot(indexed_positions - 1, indicator_sum, dtype=tf.float32)
            sampled_indices = tf.cast(tf.tensordot(tf.cast(tf.range(tf.shape(indicator)[0]), dtype=tf.float32),one_hot_selector,axes=[0, 0]),dtype=tf.int32)
            mask_itemlist = tf.gather(itemlist, sampled_indices)
            return mask_itemlist


    def boolean_mask_2d(self, itemlist, indicator, fields=None, scope=None, 
        use_static_shapes=False, indicator_sum=None, use_tpu=True, dims=1):
        """Select boxes from BoxList according to indicator and return new BoxList.
        `boolean_mask` returns the subset of boxes that are marked as "True" by the
        indicator tensor. By default, `boolean_mask` returns boxes corresponding to
        the input index list, as well as all additional fields stored in the boxlist
        (indexing into the first dimension).  However one can optionally only draw
        from a subset of fields.
        Args:
            boxlist: BoxList holding N boxes
            indicator: a rank-1 boolean tensor
            fields: (optional) list of fields to also gather from.  If None (default),
                all fields are gathered from.  Pass an empty fields list to only gather
                the box coordinates.
            scope: name scope.
            use_static_shapes: Whether to use an implementation with static shape
                gurantees.
            indicator_sum: An integer containing the sum of `indicator` vector. Only
            required if `use_static_shape` is True.
        Returns:
        subboxlist: a BoxList corresponding to the subset of the input BoxList
        specified by indicator
        Raises:
        ValueError: if `indicator` is not a rank-1 boolean tensor.
        """

        with tf.name_scope(scope, 'BooleanMask'):
            sum_idx = self.shape(itemlist, 0) 
            start_mask_lst = tf.cast(tf.zeros_like(tf.gather(itemlist, 0)), tf.float32) 
            i0 = tf.constant(0)

            @tf.function
            def mask_loop(i, stack_mask_itemlist):
                tmp_itemlist = tf.gather(itemlist, i) 
                tmp_indicator = tf.gather(indicator, i)
                tmp_mask_itemlist = self.boolean_mask_1d(tmp_itemlist, tmp_indicator, use_tpu=use_tpu, dims=1)
                return [tf.math.add(i, 1), tf.concat([stack_mask_itemlist, tmp_mask_itemlist], axis=0)]

            _, mask_itemlist_tensor = tf.while_loop(
                cond=lambda i, o1, : i < sum_idx,
                body=mask_loop, 
                loop_vars=[i0, start_mask_lst],
                shape_invariants=[i0.get_shape(), tf.TensorShape([None, None])],
                maximum_iterations=20)

            return mask_itemlist_tensor

    def ffnn(self, inputs, num_hidden_layers, hidden_size, output_size, dropout,
         output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
         hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
        if len(inputs.get_shape()) > 3:
            raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
        current_inputs = inputs

        hidden_weights = tf.get_variable("hidden_weights", [hidden_size, output_size],
                                         initializer=hidden_initializer)
        hidden_bias = tf.get_variable("hidden_bias", [output_size], initializer=tf.zeros_initializer())
        current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))
        current_inputs = current_outputs
        outputs = current_inputs
        return outputs

    def shape(self, x, dim):
        return x.get_shape()[dim].value or tf.shape(x)[dim]


    def projection(self, inputs, output_size, initializer=tf.truncated_normal_initializer(stddev=0.02)):
        
        return self.project_ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


    def project_ffnn(self, inputs, num_hidden_layers, hidden_size, output_size, dropout,
        output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
        hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
        if len(inputs.get_shape()) > 3:
            raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

        if len(inputs.get_shape()) == 3:
            batch_size = self.shape(inputs, 0)
            seqlen = self.shape(inputs, 1)
            emb_size = self.shape(inputs, 2)
            current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
        else:
            current_inputs = inputs

        for i in range(num_hidden_layers):
            hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [self.shape(current_inputs, 1), hidden_size],
                                         initializer=hidden_initializer)
            hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size], initializer=tf.zeros_initializer())
            current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

            if dropout is not None:
                current_outputs = tf.nn.dropout(current_outputs, dropout)
            current_inputs = current_outputs

        output_weights = tf.get_variable("output_weights", [self.shape(current_inputs, 1), output_size],
                                     initializer=output_weights_initializer)
        output_bias = tf.get_variable("output_bias", [output_size], initializer=tf.zeros_initializer())
        outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

        if len(inputs.get_shape()) == 3:
            outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
        return outputs

    def evaluate(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters):

        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    
        return predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold


    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
        assert i > predicted_index, (i, predicted_index)
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        if predicted_antecedent in mention_to_predicted:
            predicted_cluster = mention_to_predicted[predicted_antecedent]
        else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([predicted_antecedent])
            mention_to_predicted[predicted_antecedent] = predicted_cluster

        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        predicted_clusters[predicted_cluster].append(mention)
        mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

        return predicted_clusters, mention_to_predicted


    def get_candidate_span_label(self, gold_starts, gold_ends, candidate_starts, candidate_ends, doc_len):
        """
        Desc:
            according to the golden start/end mention position index, get the golden candidate starts/end 
            mention span labels. 
        Args:
            gold_starts: a tf.int32 tensor containing golden mention start position index in document, 
                tf.Tensor([2, 3, 4, 5])
            gold_ends: a tf.int32 tensor containing golden mention end position index in document, 
                tf.Tensor([3, 9, 10, 12])
            candidate_starts: a tf.int32 tensor containing candidate mention start position index in document, 
                tf.Tensor([3, 3, 10, 12, 16, 17])
            candidate_ends: a tf.int32 tensor containing candidate mention end position index in document,
            doc_len: an integer containing the length of the document. 
                tf.Tensor([3, 9, 14, 19, 19, 37])
        Return:
            a tf.int32 tensor containing the 0/1 label coresponds to every start-end 
            (candidate_starts, candidate_ends) span pairs. 
            tf.Tensor([1, 1, 0, 0, 0, 0])
        """
        gold_mention_sparse_label = tf.stack([gold_starts, gold_ends], axis=1)
        gold_span_value = tf.reshape(tf.ones_like(gold_starts, tf.int32), [-1])
        gold_span_shape = tf.constant([doc_len, doc_len])
        gold_span_label = tf.cast(tf.scatter_nd(gold_mention_sparse_label, gold_span_value, gold_span_shape), tf.int32)

        candidate_span = tf.stack([candidate_starts, candidate_ends], axis=1)
        gold_candidate_span_label = tf.gather_nd(gold_span_label, tf.expand_dims(candidate_span, 1))
        return gold_candidate_span_label 








