#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# author: xiaoy li
# description:



import os
import re
import conll
from typing import List, Tuple
from collections import defaultdict
import json  
import util 
import numpy as np 
import tensorflow as tf
from bert.tokenization import FullTokenizer


REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-2])

SPEAKER_START = '[unused19]'
SPEAKER_END = '[unused73]'
# MAX_LENGTH = 0

subtoken_maps = {}
gold = {}



"""
Desc:
a single training/test example for the squad dataset.
suppose origin input_tokens are :
['[unused19]', 'speaker', '#', '1', '[unused73]', '-', '-', 'basically', ',', 'it', 'was', 'unanimously', 'agreed', 'upon', 'by', 'the', 'various', 'relevant', 'parties', '.', 
'To', 'express', 'its', 'determination', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 
'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 
'de', '##viation', '##s', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '.', '[unused19]', 'Xu', '_', 'l', '##i', '[unused73]', 
'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 
'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shi', '##lin', '.', 'Good', '-', 'bye', ',', 'dear', 'viewers', '.'] 
IF sliding window size is 50. 
Args:
    doc_idx: a string: cctv/bn/0001
    sentence_map: 
        e.g. [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7]
    subtoken_map: 
        e.g. [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 53, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 97, 98, 99, 99, 99, 100, 101, 102, 103]
    flattened_window_input_ids: [num-window, window-size]
        e.g. before bert_tokenizer convert subtokens into ids:
        [['[CLS]', '[unused19]', 'speaker', '#', '1', '[unused73]', '-', '-', 'basically', ',', 'it', 'was', 'unanimously', 'agreed', 'upon', 'by', 'the', 'various', 'relevant', 'parties', '.', 'To', 'express', 'its', 'determination', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', '[SEP]'],
        ['[CLS]', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 'de', '##viation', '##s', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '[SEP]'],
        ['[CLS]', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 'de', '##viation', '##s', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '.', '[unused19]', 'Xu', '_', 'l', '##i', '[unused73]', 'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', '[SEP]'],
        ['[CLS]', '.', '[unused19]', 'Xu', '_', 'l', '##i', '[unused73]', 'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shi', '##lin', '.', 'Good', '-', 'bye', ',', 'dear', 'viewers', '[SEP]'],
        ['[CLS]', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shi', '##lin', '.', 'Good', '-', 'bye', ',', 'dear', 'viewers', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']] 
    flattened_window_masked_ids: [num-window, window-size]
        e.g.: before bert_tokenizer ids:
        [[-3, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
        [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
        [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, -1, -1, -1, -1, -1, -1, 68, 69, 70, 71, 72, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
        [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
        [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, -3, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]]
    span_start: 
        e.g.: mention start indices in the original document 
            [17, 20, 26, 43, 60, 85, 86]
    span_end:
        e.g.: mention end indices in the original document 
    cluster_ids: 
        e.g.: cluster ids for the (span_start, span_end) pairs
        [1, 1, 2, 2, 2, 3, 3] 
    check the mention in the subword list: 
        1. ['its']
        1. ['the', 'Chinese', 'securities', 'regulatory', 'department']
        2. ['this', 'stock', 'reform']
        2. ['the', 'stock', 'reform']
        2. ['the', 'stock', 'reform']
        3. ['you']
        3. ['everyone']
"""

def prepare_train_dataset(input_file, output_data_dir, output_filename, sliding_window_size, config, tokenizer=None,
    vocab_file=None, language="english", max_doc_length: int = None, is_training=True, demo=False, lowercase=False):
    if vocab_file is None:
        if not lowercase:
            vocab_file = os.path.join(REPO_PATH, "data_utils", "uppercase_vocab.txt")
        else:
            vocab_file = os.path.join(REPO_PATH, "data_utils", "lowercase_vocab.txt")

    if tokenizer is None:
        tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=lowercase)

    writer = tf.python_io.TFRecordWriter(os.path.join(output_data_dir, "{}.{}.tfrecord".format(output_filename, language)))
    doc_map = {}
    documents = read_conll_file(input_file)
    for doc_idx, document in enumerate(documents):
        doc_info = parse_document(document, language)
        tokenized_document = tokenize_document(config, doc_info, tokenizer, max_doc_length=max_doc_length)
        doc_key = tokenized_document['doc_key']
        token_windows, mask_windows, text_len = convert_to_sliding_window(tokenized_document, sliding_window_size)
        input_id_windows = [tokenizer.convert_tokens_to_ids(tokens) for tokens in token_windows]
        span_start, span_end, mention_span, cluster_ids = flatten_clusters(tokenized_document['clusters'])

        # {'sub_tokens': sub_tokens, 'sentence_map': sentence_map, 'subtoken_map': subtoken_map,
        # 'speakers': speakers, 'clusters': clusters, 'doc_key': doc_info['doc_key']}
        tmp_speaker_ids = tokenized_document["speakers"] 
        tmp_speaker_ids = [[0]*130]*config["max_training_sentences"]
        instance = (input_id_windows, mask_windows, text_len, tmp_speaker_ids, tokenized_document["genre"], is_training, span_start, span_end, cluster_ids, tokenized_document['sentence_map'])   
        write_instance_to_example_file(writer, instance, doc_key, config)
        doc_map[doc_idx] = doc_key
        if demo and doc_idx > 3:
            break 
    with open(os.path.join(output_data_dir, "{}.{}.map".format(output_filename, language)), 'w') as fo:
        json.dump(doc_map, fo, indent=2)



def write_instance_to_example_file(writer, instance, doc_key, config):
    input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map = instance 
    input_id_windows = input_ids 
    mask_windows = input_mask 
    flattened_input_ids = [i for j in input_id_windows for i in j]
    flattened_input_mask = [i for j in mask_windows for i in j]
    cluster_ids = [int(tmp) for tmp in cluster_ids]

    max_sequence_len = int(config["max_training_sentences"])
    max_seg_len = int(config["max_segment_len"])
    before_pad_start = gold_starts 
    before_pad_end = gold_ends 
    before_text_len = text_len 

    sentence_map = clip_or_pad(sentence_map, max_sequence_len*max_seg_len, pad_idx=-1)
    text_len = clip_or_pad(text_len, max_sequence_len, pad_idx=-1)
    tmp_subtoken_maps = clip_or_pad(subtoken_maps[doc_key], max_sequence_len*max_seg_len, pad_idx=-1)

    tmp_speaker_ids = clip_or_pad(speaker_ids[0], max_sequence_len*max_seg_len, pad_idx=-1)

    flattened_input_ids = clip_or_pad(flattened_input_ids, max_sequence_len*max_seg_len, pad_idx=-1)
    flattened_input_mask = clip_or_pad(flattened_input_mask, max_sequence_len*max_seg_len, pad_idx=-1)
    # genre = clip_or_pad(genre, )
    gold_starts = clip_or_pad(gold_starts, config["max_cluster_num"], pad_idx=-1)
    gold_ends = clip_or_pad(gold_ends, config["max_cluster_num"], pad_idx=-1)
    cluster_ids = clip_or_pad(cluster_ids, config["max_cluster_num"], pad_idx=-1)

    span_mention  = pad_span_mention(before_text_len, config, before_pad_start, before_pad_end)

    features = {
        'sentence_map': create_int_feature(sentence_map), 
        'text_len': create_int_feature(text_len), 
        'subtoken_map': create_int_feature(tmp_subtoken_maps), 
        'speaker_ids': create_int_feature(tmp_speaker_ids), 
        'flattened_input_ids': create_int_feature(flattened_input_ids),
        'flattened_input_mask': create_int_feature(flattened_input_mask),
        'genre': create_int_feature([genre]),
        'span_starts': create_int_feature(gold_starts), 
        'span_ends': create_int_feature(gold_ends), 
        'cluster_ids': create_int_feature(cluster_ids),
        'span_mention': create_int_feature(span_mention) 
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def pad_span_mention(text_len_lst, config, before_pad_start, before_pad_end):
    span_mention = np.zeros((config["max_training_sentences"], config["max_segment_len"], config["max_segment_len"]), dtype=int)

    for idx, (tmp_s, tmp_e) in enumerate(zip(before_pad_start, before_pad_end)):
        start_seg = int(tmp_s // config["max_segment_len"])
        end_seg = int(tmp_s // config["max_segment_len"])
        if start_seg != end_seg:
            continue 
        try:
            sent_idx = int(tmp_s // config["max_segment_len"]) + 1 if tmp_s % config["max_segment_len"] != 0 else int(tmp_s // config["max_segment_len"])
            start_offset = tmp_s % config["max_segment_len"] 
            end_offset = tmp_e % config["max_segment_len"]
            span_mention[sent_idx, start_offset, end_offset] = 1 
        except:
            continue 

    flatten_span_mention = np.reshape(span_mention, (1, -1))
    flatten_span_mention = flatten_span_mention.tolist()
    flatten_span_mention = [j for j in flatten_span_mention]

    return flatten_span_mention[0]


def clip_or_pad(var, max_var_len, pad_idx=-1):
    
    if len(var) >= max_var_len:
        return var[:max_var_len]
    else:
        pad_var  = (max_var_len - len(var)) * [pad_idx]
        var = list(var) + list(pad_var) 
        return var 


def flatten_clusters(clusters: List[List[Tuple[int, int]]]) -> Tuple[
    List[int], List[int], List[Tuple[int, int]], List[int]]:
    """
    flattern cluster information
    :param clusters:
    :return:
    """
    span_starts = []
    span_ends = []
    cluster_ids = []
    mention_span = []
    for cluster_id, cluster in enumerate(clusters):
        for start, end in cluster:
            span_starts.append(start)
            span_ends.append(end)
            mention_span.append((start, end))
            cluster_ids.append(cluster_id + 1)
    return span_starts, span_ends, mention_span, cluster_ids


def read_conll_file(conll_file_path):
    documents = []
    with open(conll_file_path, "r", encoding="utf-8") as fi:
        for line in fi:
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line.strip())
    return documents


def parse_document(document: Tuple[str, List], language: str) -> dict:
    """
    get basic information from one document annotation.
    :param document:
    :param language: english, chinese or arabic
    :return:
    """
    doc_key = document[0]
    sentences = [[]]
    speakers = []
    coreferences = []
    word_idx = -1
    last_speaker = ''
    for line_id, line in enumerate(document[1]):
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12
            word_idx += 1
            word = normalize_word(row[3], language)
            sentences[-1].append(word)
            speaker = row[9]
            if speaker != last_speaker:
                speakers.append((word_idx, speaker))
                last_speaker = speaker
            coreferences.append(row[-1])
        else:
            sentences.append([])
    clusters = coreference_annotations_to_clusters(coreferences)
    doc_info = {'doc_key': doc_key, 'sentences': sentences[: -1], 'speakers': speakers, 'clusters': clusters}
    return doc_info


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def coreference_annotations_to_clusters(annotations: List[str]) -> List[List[Tuple]]:
    """
    convert coreference information to clusters
    :param annotations:
    :return:
    """
    clusters = defaultdict(list)
    coref_stack = defaultdict(list)
    for word_idx, annotation in enumerate(annotations):
        if annotation == '-':
            continue
        for ann in annotation.split('|'):
            cluster_id = int(ann.replace('(', '').replace(')', ''))
            if ann[0] == '(' and ann[-1] == ')':
                clusters[cluster_id].append((word_idx, word_idx))
            elif ann[0] == '(':
                coref_stack[cluster_id].append(word_idx)
            elif ann[-1] == ')':
                span_start = coref_stack[cluster_id].pop()
                clusters[cluster_id].append((span_start, word_idx))
            else:
                raise NotImplementedError
    assert all([len(starts) == 0 for starts in coref_stack.values()])
    return list(clusters.values())


def checkout_clusters(doc_info):
    words = [i for j in doc_info['sentences'] for i in j]
    clusters = [[' '.join(words[start: end + 1]) for start, end in cluster] for cluster in doc_info['clusters']]
    print(clusters)


def tokenize_document(config, doc_info, tokenizer, max_doc_length):
    """
    tokenize into sub tokens
    :param doc_info:
    :param tokenizer:
    max_doc_length: pad to max_doc_length
    :return:
    """
    genres = {g: i for i, g in enumerate(config["genres"])}
    sub_tokens: List[str] = []  # all sub tokens of a document
    sentence_map: List[int] = []  # collected tokenized tokens -> sentence id
    subtoken_map: List[int] = []  # collected tokenized tokens -> original token id

    word_idx = -1

    for sentence_id, sentence in enumerate(doc_info['sentences']):
        for token in sentence:
            word_idx += 1
            word_tokens = tokenizer.tokenize(token)
            sub_tokens.extend(word_tokens)
            sentence_map.extend([sentence_id] * len(word_tokens))
            subtoken_map.extend([word_idx] * len(word_tokens))
    if max_doc_length:
        num_to_pad = max_doc_length - len(sub_tokens)
        sub_tokens.extend(["[PAD]"] * num_to_pad)
        sentence_map.extend([sentence_map[-1]+1] * num_to_pad)
        subtoken_map.extend(list(range(word_idx+1, num_to_pad+1+word_idx)))
    # global MAX_LENGTH
    # if len(sub_tokens) > MAX_LENGTH:
    #     print(len(sub_tokens))
    #     MAX_LENGTH = len(sub_tokens)
        # print(MAX_LENGTH)
    # todo(yuxian): need pad speakers?
    subtoken_maps[doc_info['doc_key']] = subtoken_map
    genre = genres.get(doc_info['doc_key'][:2], 0)
    speakers = {subtoken_map.index(word_index): tokenizer.tokenize(speaker)
                for word_index, speaker in doc_info['speakers']}
    clusters = [[(subtoken_map.index(start), len(subtoken_map) - 1 - subtoken_map[::-1].index(end))
                 for start, end in cluster] for cluster in doc_info['clusters']]
    tokenized_document = {'sub_tokens': sub_tokens, 'sentence_map': sentence_map, 'subtoken_map': subtoken_map,
                          'speakers': speakers, 'clusters': clusters, 'doc_key': doc_info['doc_key'], 
                          "genre": genre}
    return tokenized_document


def convert_to_sliding_window(tokenized_document: dict, sliding_window_size: int):
    """
    construct sliding windows, allocate tokens and masks into each window
    :param tokenized_document:
    :param sliding_window_size:
    :return:
    """
    expanded_tokens, expanded_masks = expand_with_speakers(tokenized_document)
    sliding_windows = construct_sliding_windows(len(expanded_tokens), sliding_window_size - 2)
    token_windows = []  # expanded tokens to sliding window
    mask_windows = []  # expanded masks to sliding window

    for window_start, window_end, window_mask in sliding_windows:
        original_tokens = expanded_tokens[window_start: window_end]
        original_masks = expanded_masks[window_start: window_end]
        window_masks = [-2 if w == 0 else o for w, o in zip(window_mask, original_masks)]
        one_window_token = ['[CLS]'] + original_tokens + ['[SEP]'] + ['[PAD]'] * (
                sliding_window_size - 2 - len(original_tokens))
        one_window_mask = [-3] + window_masks + [-3] + [-4] * (sliding_window_size - 2 - len(original_tokens))
        assert len(one_window_token) == sliding_window_size
        assert len(one_window_mask) == sliding_window_size
        token_windows.append(one_window_token)
        mask_windows.append(one_window_mask)
    assert len(tokenized_document['sentence_map']) == sum([i >= 0 for j in mask_windows for i in j])

    text_len = np.array([len(s) for s in token_windows])

    return token_windows, mask_windows, text_len


def expand_with_speakers(tokenized_document: dict) -> Tuple[List[str], List[int]]:
    """
    add speaker name information
    :param tokenized_document: tokenized document information
    :return:
    """
    expanded_tokens = []
    expanded_masks = []
    for token_idx, token in enumerate(tokenized_document['sub_tokens']):
        if token_idx in tokenized_document['speakers']:
            speaker = [SPEAKER_START] + tokenized_document['speakers'][token_idx] + [SPEAKER_END]
            expanded_tokens.extend(speaker)
            expanded_masks.extend([-1] * len(speaker))
        expanded_tokens.append(token)
        expanded_masks.append(token_idx)
    return expanded_tokens, expanded_masks


def construct_sliding_windows(sequence_length: int, sliding_window_size: int):
    """
    construct sliding windows for BERT processing
    :param sequence_length: e.g. 9
    :param sliding_window_size: e.g. 4
    :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
    """
    sliding_windows = []
    stride = int(sliding_window_size / 2)
    start_index = 0
    end_index = 0
    while end_index < sequence_length:
        end_index = min(start_index + sliding_window_size, sequence_length)
        left_value = 1 if start_index == 0 else 0
        right_value = 1 if end_index == sequence_length else 0
        mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
               + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
        mask = mask[: end_index - start_index]
        sliding_windows.append((start_index, end_index, mask))
        start_index += stride
    assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
    return sliding_windows



if __name__ == "__main__":
    # ---------
    # python3 build_data_to_tfrecord.py 
    demo = False
    lowercase = False # expermental dataset should be False 
    config = util.initialize_from_env(use_tpu=False, config_file="experiments_tinybert.conf")
    for sliding_window_size in [256]: #  128, 384,]:  # 512]:
        for max_training_sentences in [8]:
            config["max_segment_len"] = sliding_window_size
            config["max_training_sentences"] = max_training_sentences
            print("=*="*20)
            print("current sliding window size is : {}".format(str(config["max_segment_len"])))
            print("current number of max training sentences is : {}".format(str(config["max_training_sentences"])))
            print("=*="*20)
            for data_sign in ["train", "dev", "test"]:
                print("%*%"*20)
                print(data_sign)
                print("%*%"*20)
                language = "english"
                vocab_file = "/xiaoya/pretrain_ckpt/spanbert_base_cased/vocab.txt"
                input_data_dir = "/xiaoya/data" 
                input_filename = "{}.english.v4_gold_conll".format(data_sign)
                input_file_path = os.path.join(input_data_dir, input_filename)
                
                if lowercase:
                    if demo:
                        output_data_dir = "/xiaoya/corefqa_data/lowercase_demo_overlap_{}_{}".format(str(config["max_segment_len"]), str(config["max_training_sentences"]))
                    else:
                        output_data_dir = "/xiaoya/corefqa_data/lowercase_overlap_{}_{}".format(str(config["max_segment_len"]), str(config["max_training_sentences"]))
                else:
                    if demo:
                        output_data_dir = "/xiaoya/corefqa_data/demo_overlap_{}_{}".format(str(config["max_segment_len"]), str(config["max_training_sentences"]))
                    else:
                        output_data_dir = "/xiaoya/corefqa_data/overlap_{}_{}".format(str(config["max_segment_len"]), str(config["max_training_sentences"]))

                print("current max training sentence is : {}".format(str(config["max_training_sentences"])))
                os.makedirs(output_data_dir, exist_ok=True)
                output_filename = "{}.{}".format(data_sign, str(sliding_window_size))
                print("$^$"*30)
                print(output_data_dir, output_filename)
                print("$^$"*30)
                # prepare_training_data(input_data_dir, output_data_dir, input_filename, output_filename, language, config, vocab_file, sliding_window_size)
                prepare_train_dataset(input_file_path, output_data_dir, output_filename, 
                    sliding_window_size, config, vocab_file=vocab_file, demo=demo, lowercase=lowercase)



