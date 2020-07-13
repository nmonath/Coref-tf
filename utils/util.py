from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import errno
import os
import shutil

import pyhocon
import tensorflow as tf
from models import corefqa
from models import mention_proposal 


repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])


def get_model(config, model_sign="corefqa"):
    if model_sign == "corefqa":
        return corefqa.CorefModel(config)
    else:
        return mention_proposal.MentionProposalModel(config)


def initialize_from_env(eval_test=False, config_params="train_spanbert_base", config_file="experiments_tinybert.conf", use_tpu=False, print_info=False):
    # if "GPU" in os.environ:
    #     set_gpus(int(os.environ["GPU"]))

    if not use_tpu:
        print("loading experiments.conf ... ")
        config = pyhocon.ConfigFactory.parse_file(os.path.join(repo_path, config_file)) 
    else: 
        print("loading experiments_tpu.conf ... ")
        config = pyhocon.ConfigFactory.parse_file(os.path.join(repo_path, config_file))

    config = config[config_params]

    if print_info:
        tf.logging.info("%*%"*20)
        tf.logging.info("%*%"*20)
        tf.logging.info("%%%%%%%% Configs are showed as follows : %%%%%%%%")
        for tmp_key, tmp_value in config.items():
            tf.logging.info(str(tmp_key) + " : " + str(tmp_value)) 
    
        tf.logging.info("%*%"*20)
        tf.logging.info("%*%"*20)

    config["log_dir"] = mkdirs(os.path.join(config["log_root"], config_params))

    if print_info:
        tf.logging.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def flatten(l):
    return [item for sublist in l for item in sublist]


def set_gpus(*gpus):
    # pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size, initializer=tf.truncated_normal_initializer(stddev=0.02)):
    return ffnn_bk(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def highway(inputs, num_layers, dropout):
    for i in range(num_layers):
        with tf.variable_scope("highway_{}".format(i)):
            j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
            f = tf.sigmoid(f)
            j = tf.nn.relu(j)
            if dropout is not None:
                j = tf.nn.dropout(j, dropout)
            inputs = f * j + (1 - f) * inputs
    return inputs


def shape(x, dim):
    return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn_bk(inputs, num_hidden_layers, hidden_size, output_size, dropout,
         output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
         hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
    if len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

    if len(inputs.get_shape()) == 3:
        batch_size = shape(inputs, 0)
        seqlen = shape(inputs, 1)
        emb_size = shape(inputs, 2)
        current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size],
                                         initializer=hidden_initializer)
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size], initializer=tf.zeros_initializer())
        current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size],
                                     initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size], initializer=tf.zeros_initializer())
    outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
    return outputs

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout,
         output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
         hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
    if len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
    current_inputs = inputs

    # for i in range(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights", [hidden_size, output_size],
                                         initializer=hidden_initializer)
    hidden_bias = tf.get_variable("hidden_bias", [output_size], initializer=tf.zeros_initializer())
    current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

    # if dropout is not None:
    #    current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

    # output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size],
    #                                  initializer=output_weights_initializer)
    # output_bias = tf.get_variable("output_bias", [output_size], initializer=tf.zeros_initializer())
    # outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)
    outputs = current_inputs

    # if len(inputs.get_shape()) == 3:
    #     outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
    return outputs


def linear(inputs, output_size):
    if len(inputs.get_shape()) == 3:
        batch_size = shape(inputs, 0)
        seqlen = shape(inputs, 1)
        emb_size = shape(inputs, 2)
        current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
    else:
        current_inputs = inputs
    hidden_weights = tf.get_variable("linear_w", [shape(current_inputs, 1), output_size])
    hidden_bias = tf.get_variable("bias", [output_size])
    current_outputs = tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias)
    return current_outputs



def batch_gather(emb, indices):
    batch_size = shape(emb, 0)
    seqlen = shape(emb, 1)
    if len(emb.get_shape()) > 2:
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
    offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
    gathered = tf.gather(flattened_emb, indices + offset)  # [batch_size, num_indices, emb]
    if len(emb.get_shape()) == 2:
        gathered = tf.squeeze(gathered, 2)  # [batch_size, num_indices]
    return gathered


class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1
