#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


import os
import sys 


repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)


import tensorflow as tf
from utils import util
from bert import modeling
from bert import tokenization




class MentionProposalModel(object):
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
        self.bert_config.hidden_dropout_prob = self.config["dropout_rate"]
        self.tokenizer = tokenization.FullTokenizer(vocab_file=config['vocab_file'], do_lower_case=False)


    def get_dropout(self, dropout_rate, is_training):  # is_training为True时keep=1-drop, 为False时keep=1
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def get_mention_proposal_and_loss(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training,
        gold_starts, gold_ends, cluster_ids, sentence_map, span_mention=None):
        """get mention proposals"""

        start_end_loss_mask = tf.cast(tf.where(tf.cast(tf.math.greater_equal(input_ids, tf.zeros_like(input_ids)),tf.bool), x=tf.ones_like(input_ids), y=tf.zeros_like(input_ids)), tf.float32) 
        input_ids = tf.where(tf.cast(tf.math.greater_equal(input_ids, tf.zeros_like(input_ids)),tf.bool), x=input_ids, y=tf.zeros_like(input_ids)) 
        input_mask = tf.where(tf.cast(tf.math.greater_equal(input_mask, tf.zeros_like(input_mask)), tf.bool), x=input_mask, y=tf.zeros_like(input_mask)) 
        text_len = tf.where(tf.cast(tf.math.greater_equal(text_len, tf.zeros_like(text_len)), tf.bool), x= text_len, y=tf.zeros_like(text_len)) 
        speaker_ids = tf.where(tf.cast(tf.math.greater_equal(speaker_ids, tf.zeros_like(speaker_ids)),tf.bool), x=speaker_ids, y=tf.zeros_like(speaker_ids)) 
        gold_starts = tf.where(tf.cast(tf.math.greater_equal(gold_starts, tf.zeros_like(gold_starts)),tf.bool), x=gold_starts, y=tf.zeros_like(gold_starts)) 
        gold_ends = tf.where(tf.cast(tf.math.greater_equal(gold_ends, tf.zeros_like(gold_ends)),tf.bool), x=gold_ends, y=tf.zeros_like(gold_ends) ) 
        cluster_ids = tf.where(tf.cast(tf.math.greater_equal(cluster_ids, tf.zeros_like(cluster_ids)),tf.bool), x=cluster_ids, y=tf.zeros_like(cluster_ids)) 
        sentence_map = tf.where(tf.cast(tf.math.greater_equal(sentence_map, tf.zeros_like(sentence_map)),tf.bool), x=sentence_map, y=tf.zeros_like(sentence_map)) 
        span_mention = tf.where(tf.cast(tf.math.greater_equal(span_mention, tf.zeros_like(span_mention)),tf.bool), x=span_mention, y=tf.zeros_like(span_mention)) 
        span_mention_loss_mask = tf.cast(tf.where(tf.cast(tf.math.greater_equal(span_mention, tf.zeros_like(span_mention)),tf.bool), x=tf.ones_like(span_mention), y=tf.zeros_like(span_mention)) , tf.float32)
        # span

        # gold_starts -> [1, 3, 5, 8, -1, -1, -1, -1] -> [1, 3, 5, 8, 0, 0, 0, 0]

        input_ids = tf.reshape(input_ids, [-1, self.config["max_segment_len"]])    # (max_train_sent, max_segment_len) 
        input_mask  = tf.reshape(input_mask, [-1, self.config["max_segment_len"]])   # (max_train_sent, max_segment_len)
        text_len = tf.reshape(text_len, [-1])  # (max_train_sent)
        speaker_ids = tf.reshape(speaker_ids, [-1, self.config["max_segment_len"]])  # (max_train_sent, max_segment_len) 
        sentence_map = tf.reshape(sentence_map, [-1])   # (max_train_sent * max_segment_len) 
        cluster_ids = tf.reshape(cluster_ids, [-1])     # (max_train_sent * max_segment_len) 
        gold_starts = tf.reshape(gold_starts, [-1])     # (max_train_sent * max_segment_len) 
        gold_ends = tf.reshape(gold_ends, [-1])         # (max_train_sent * max_segment_len) 
        span_mention = tf.reshape(span_mention, [self.config["max_training_sentences"], self.config["max_segment_len"] * self.config["max_segment_len"]])
        # span_mention : (max_train_sent, max_segment_len, max_segment_len)

        model = modeling.BertModel(config=self.bert_config, is_training=is_training, input_ids=input_ids,
            input_mask=input_mask, use_one_hot_embeddings=False, scope='bert')

        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        mention_doc = model.get_sequence_output()  # (max_train_sent, max_segment_len, hidden)
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)  # (max_train_sent, max_segment_len, emb) -> (max_train_sent * max_segment_len, e) 取出有效token的emb
        num_words = util.shape(mention_doc, 0)  # max_train_sent * max_segment_len

        seg_mention_doc = tf.reshape(mention_doc, [self.config["max_training_sentences"], self.config["max_segment_len"], -1]) # (max_train_sent, max_segment_len, embed)
        start_seg_mention_doc = tf.stack([seg_mention_doc] * self.config["max_segment_len"], axis=1) # (max_train_sent, 1, max_segment_len, embed) -> (max_train_sent, max_segment_len, max_segment_len, embed)
        end_seg_mention_doc = tf.stack([seg_mention_doc, ] * self.config["max_segment_len"], axis=2) # (max_train_sent, max_segment_len, 1, embed) -> (max_train_sent, max_segment_len, max_segment_len, embed)
        span_mention_doc = tf.concat([start_seg_mention_doc, end_seg_mention_doc], axis=-1) # (max_train_sent, max_segment_len, max_segment_len, embed * 2)
        span_mention_doc = tf.reshape(span_mention_doc, (self.config["max_training_sentences"]*self.config["max_segment_len"]*self.config["max_segment_len"], -1))
        # # (max_train_sent * max_segment_len * max_segment_len, embed * 2)

        with tf.variable_scope("span_scores", reuse=tf.AUTO_REUSE):  # [k, 1] 每个候选span的得分
            span_scores = util.ffnn(span_mention_doc, self.config["ffnn_depth"], self.config["ffnn_size"]*2, 1, self.dropout) # (max_train_sent, max_segment_len, 1)
        with tf.variable_scope("start_scores", reuse=tf.AUTO_REUSE):  # [k, 1] 每个候选span的得分
            start_scores = util.ffnn(mention_doc, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # (max_train_sent, max_segment_len, 1) 
        with tf.variable_scope("end_scores", reuse=tf.AUTO_REUSE):  # [k, 1] 每个候选span的得分
            end_scores = util.ffnn(mention_doc, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # (max_train_sent, max_segment_len, 1)

        gold_start_label = tf.reshape(gold_starts, [-1, 1])  
        # gold_starts -> [1, 3, 5, 8, -1, -1, -1, -1]
        start_value = tf.reshape(tf.ones_like(gold_starts), [-1])
        start_shape = tf.constant([self.config["max_training_sentences"] * self.config["max_segment_len"]])
        gold_start_label = tf.cast(tf.scatter_nd(gold_start_label, start_value, start_shape), tf.int32)
        # gold_start_label = tf.boolean_mask(gold_start_label, tf.reshape(input_mask, [-1]))

        gold_end_label = tf.reshape(gold_ends, [-1, 1])
        end_value = tf.reshape(tf.ones_like(gold_ends), [-1])
        end_shape = tf.constant([self.config["max_training_sentences"] * self.config["max_segment_len"]])
        gold_end_label = tf.cast(tf.scatter_nd(gold_end_label, end_value, end_shape), tf.int32)
        # gold_end_label = tf.boolean_mask(gold_end_label, tf.reshape(input_mask, [-1]))
        start_scores = tf.cast(tf.reshape(tf.sigmoid(start_scores), [-1]),tf.float32)
        end_scores = tf.cast(tf.reshape(tf.sigmoid(end_scores), [-1]),tf.float32)
        span_scores = tf.cast(tf.reshape(tf.sigmoid(span_scores), [-1]), tf.float32)

        start_scores = tf.stack([(1 - start_scores), start_scores], axis=-1) 
        end_scores = tf.stack([(1 - end_scores), end_scores], axis=-1) 
        span_scores = tf.stack([(1 - span_scores), span_scores], axis=-1)

        gold_start_label = tf.cast(tf.one_hot(tf.reshape(gold_start_label, [-1]), 2, axis=-1), tf.float32)
        gold_end_label = tf.cast(tf.one_hot(tf.reshape(gold_end_label, [-1]), 2, axis=-1), tf.float32)
        span_mention = tf.cast(tf.one_hot(tf.reshape(span_mention, [-1]), 2, axis=-1),tf.float32)

        start_end_loss_mask = tf.reshape(start_end_loss_mask, [-1])
        # true, pred 
        start_loss = tf.keras.losses.binary_crossentropy(gold_start_label, start_scores,)
        end_loss = tf.keras.losses.binary_crossentropy(gold_end_label, end_scores)
        span_loss = tf.keras.losses.binary_crossentropy(span_mention, span_scores,)

        start_loss = tf.reduce_mean(tf.multiply(start_loss, tf.cast(start_end_loss_mask, tf.float32))) 
        end_loss = tf.reduce_mean(tf.multiply(end_loss, tf.cast(start_end_loss_mask, tf.float32))) 
        span_loss = tf.reduce_mean(tf.multiply(span_loss, tf.cast(span_mention_loss_mask, tf.float32))) 


        if span_mention is None :
            loss = self.config["start_ratio"] * start_loss + self.config["end_ratio"] * end_loss 
            return loss, start_scores, end_scores
        else:
            loss = self.config["start_ratio"] * start_loss + self.config["end_ratio"] * end_loss +self.config["mention_ratio"] * span_loss 
            return loss, start_scores, end_scores, span_scores 


    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return flattened_emb 
        ##### return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))





