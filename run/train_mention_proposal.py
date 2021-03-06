#!/usr/bin/env python3
# -*- coding: utf-8 -*- 


"""
this file contains pre-training and testing the mention proposal model
"""

import os 
import math 
import logging
import numpy as np 
import tensorflow as tf
from utils import util
from utils.radam import RAdam
from data_utils.input_builder import file_based_input_fn_builder



tf.app.flags.DEFINE_string('f', '', 'kernel')
flags = tf.app.flags
flags.DEFINE_string("output_dir", "data", "The output directory of the model training.")
flags.DEFINE_string("eval_dir", "/home/lixiaoya/mention_proposal_output_dir", "The output directory of the saved mention proposal models.")
flags.DEFINE_bool("do_train", True, "Whether to train a model.")
flags.DEFINE_bool("do_eval", False, "Whether to test a model.")
flags.DEFINE_bool("do_predict", False, "Whether to test a model.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("concat_only", False, "Whether to use start/end embedding for calculating mention scores.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")
flags.DEFINE_integer("keep_checkpoint_max", 30, "How many checkpoint models keep at most.")
flags.DEFINE_string("config_filename", "experiments.conf", "the input config file name.")
flags.DEFINE_string("config_params", "train_spanbert_base", "specify the hyper-parameters in the config file.")
flags.DEFINE_string("logfile_path", "/home/lixiaoya/spanbert_large_mention_proposal.log", "the path to the exported log file.")
flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name "
                       "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not "
                       "specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not "
                       "specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer("num_tpu_cores", 1, "Only used if `use_tpu` is True. Total number of TPU cores to use.")
FLAGS = tf.flags.FLAGS


format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format, filename=FLAGS.logfile_path, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def model_fn_builder(config):

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        config = util.initialize_from_env(use_tpu=FLAGS.use_tpu, config_params=FLAGS.config_params, config_file=FLAGS.config_filename)

        input_ids = features["flattened_input_ids"]
        input_mask = features["flattened_input_mask"]
        text_len = features["text_len"]
        speaker_ids = features["speaker_ids"]
        genre = features["genre"] 
        gold_starts = features["span_starts"]
        gold_ends = features["span_ends"]
        cluster_ids = features["cluster_ids"]
        sentence_map = features["sentence_map"] 
        span_mention = features["span_mention"]
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = util.get_model(config, model_sign="mention_proposal")

        if FLAGS.use_tpu:
            def tpu_scaffold():
                return tf.train.Scaffold()
            scaffold_fn = tpu_scaffold
        else:
            scaffold_fn = None 

        if mode == tf.estimator.ModeKeys.TRAIN: 
            tf.logging.info("****************************** tf.estimator.ModeKeys.TRAIN ******************************")
            tf.logging.info("********* Features *********")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            total_loss, start_scores, end_scores, span_scores = model.get_mention_proposal_and_loss(input_ids, input_mask, \
                text_len, speaker_ids, genre, is_training, gold_starts,
                gold_ends, cluster_ids, sentence_map, span_mention=span_mention)

            if config["tpu"]:
                optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'], beta1=0.9, beta2=0.999, epsilon=1e-08)
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step()) 
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            else:
                optimizer = RAdam(learning_rate=config['learning_rate'], epsilon=1e-8, beta1=0.9, beta2=0.999)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step())
        
                train_logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=1)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn,
                    training_hooks=[train_logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL: 
            tf.logging.info("****************************** tf.estimator.ModeKeys.EVAL ******************************")
            total_loss, start_scores, end_scores, span_scores = model.get_mention_proposal_and_loss(input_ids, input_mask, \
                text_len, speaker_ids, genre, is_training, gold_starts,
                gold_ends, cluster_ids, sentence_map, span_mention)

            def metric_fn(start_scores, end_scores, span_scores, gold_span_label):
                if config["mention_proposal_only_concate"]:
                    pred_span_label = tf.cast(tf.reshape(tf.math.greater_equal(span_scores, config["threshold"]), [-1]), tf.bool)
                else:
                    start_scores = tf.reshape(start_scores, [-1, config["max_segment_len"]])
                    end_scores = tf.reshape(end_scores, [-1, config["max_segment_len"]])
                    start_scores = tf.tile(tf.expand_dims(start_scores, 2), [1, 1, config["max_segment_len"]])
                    end_scores = tf.tile(tf.expand_dims(end_scores, 2), [1, 1, config["max_segment_len"]])
                    sce_span_scores = (start_scores + end_scores + span_scores)/ 3
                    pred_span_label = tf.cast(tf.reshape(tf.math.greater_equal(sce_span_scores, config["threshold"]), [-1]), tf.bool)

                gold_span_label = tf.cast(tf.reshape(gold_span_label, [-1]), tf.bool)

                return {"precision": tf.compat.v1.metrics.precision(gold_span_label, pred_span_label), 
                        "recall": tf.compat.v1.metrics.recall(gold_span_label, pred_span_label)}

            eval_metrics = (metric_fn, [start_scores, end_scores, span_scores, span_mention])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            tf.logging.info("****************************** tf.estimator.ModeKeys.PREDICT ******************************")
            total_loss, start_scores, end_scores, span_scores = model.get_mention_proposal_and_loss(input_ids, input_mask, \
                text_len, speaker_ids, genre, is_training, gold_starts,
                gold_ends, cluster_ids, sentence_map, span_mention)

            predictions = {
                    "total_loss": total_loss,
                    "start_scores": start_scores,
                    "start_gold": gold_starts,
                    "end_gold": gold_ends,
                    "end_scores": end_scores, 
                    "span_scores": span_scores, 
                    "span_gold": span_mention
            }
            
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Please check the the mode ! ")
        
        return output_spec

    return model_fn


def mention_proposal_prediction(config, current_doc_result, concat_only=True):
    """
    current_doc_result: 
        "total_loss": total_loss,
        "start_scores": start_scores,
        "start_gold": gold_starts,
        "end_gold": gold_ends,
        "end_scores": end_scores, 
        "span_scores": span_scores, 
        "span_gold": span_mention

    """

    span_scores = current_doc_result["span_scores"]
    span_gold = current_doc_result["span_gold"] 

    if concat_only:
        scores = span_scores
    else:
        start_scores = current_doc_result["start_scores"], 
        end_scores = current_doc_result["end_scores"]   
        # start_scores = tf.tile(tf.expand_dims(start_scores, 2), [1, 1, config["max_segment_len"]])
        start_scores = np.tile(np.expand_dims(start_scores, axis=2), (1, 1, config["max_segment_len"]))
        end_scores = np.tile(np.expand_dims(end_scores, axis=2), (1, 1, config["max_segment_len"]))
        start_scores = np.reshape(start_scores, [-1, config["max_segment_len"], config["max_segment_len"]])
        end_scores = np.reshape(end_scores, [-1, config["max_segment_len"], config["max_segment_len"]])

        # end_scores -> max_training_sent, max_segment_len 
        scores = (start_scores + end_scores + span_scores)/3

    pred_span_label = scores >= 0.5
    pred_span_label = np.reshape(pred_span_label, [-1])
    gold_span_label = np.reshape(span_gold, [-1])

    return pred_span_label, gold_span_label


def main(_):
    config = util.initialize_from_env(use_tpu=FLAGS.use_tpu, config_params=FLAGS.config_params, config_file=FLAGS.config_filename, print_info=True)

    tf.logging.set_verbosity(tf.logging.INFO)
    num_train_steps = config["num_docs"] * config["num_epochs"]
    keep_chceckpoint_max = max(math.ceil(num_train_steps / config["save_checkpoints_steps"]), FLAGS.keep_checkpoint_max)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    tf.gfile.MakeDirs(FLAGS.output_dir)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        evaluation_master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max = keep_chceckpoint_max,
        save_checkpoints_steps=config["save_checkpoints_steps"],
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(config)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        eval_on_tpu=FLAGS.use_tpu,
        warm_start_from=tf.estimator.WarmStartSettings(config["init_checkpoint"],
            vars_to_warm_start="bert*"),
        model_fn=model_fn,
        config=run_config,
        train_batch_size=1,
        eval_batch_size=1,
        predict_batch_size=1)

    seq_length = config["max_segment_len"] * config["max_training_sentences"]


    if FLAGS.do_train:
        estimator.train(input_fn=file_based_input_fn_builder(config["train_path"], seq_length, config, 
            is_training=True, drop_remainder=True), max_steps=num_train_steps)
    

    if FLAGS.do_eval:
        best_dev_f1, best_dev_prec, best_dev_rec, test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best = 0, 0, 0, 0, 0, 0
        best_ckpt_path = ""
        checkpoints_iterator = [os.path.join(FLAGS.eval_dir, "model.ckpt-{}".format(str(int(ckpt_idx)))) for ckpt_idx in range(0, num_train_steps, config["save_checkpoints_steps"])]
        for checkpoint_path in checkpoints_iterator[1:]:
            eval_dev_result = estimator.evaluate(input_fn=file_based_input_fn_builder(config["dev_path"], seq_length, config,is_training=False, drop_remainder=False),
                steps=698, checkpoint_path=checkpoint_path)
            dev_f1 = 2*eval_dev_result["precision"] * eval_dev_result["recall"] / (eval_dev_result["precision"] + eval_dev_result["recall"]+1e-10)
            tf.logging.info("***** Current ckpt path is ***** : {}".format(checkpoint_path))
            tf.logging.info("***** EVAL ON DEV SET *****")
            tf.logging.info("***** [DEV EVAL] ***** : precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(eval_dev_result["precision"], eval_dev_result["recall"], dev_f1))
            if dev_f1 > best_dev_f1:
                best_dev_f1, best_dev_prec, best_dev_rec = dev_f1, eval_dev_result["precision"], eval_dev_result["recall"]
                best_ckpt_path = checkpoint_path
                eval_test_result = estimator.evaluate(input_fn=file_based_input_fn_builder(config["test_path"], seq_length, config,is_training=False, drop_remainder=False),steps=698, checkpoint_path=checkpoint_path)
                test_f1 = 2*eval_test_result["precision"] * eval_test_result["recall"] / (eval_test_result["precision"] + eval_test_result["recall"]+1e-10)
                test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best = test_f1, eval_test_result["precision"], eval_test_result["recall"]
                tf.logging.info("***** EVAL ON TEST SET *****")
                tf.logging.info("***** [TEST EVAL] ***** : precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(eval_test_result["precision"], eval_test_result["recall"], test_f1))
        tf.logging.info("*"*20)
        tf.logging.info("- @@@@@ the path to the BEST DEV result is : {}".format(best_ckpt_path))
        tf.logging.info("- @@@@@ BEST DEV F1 : {:.4f}, Precision : {:.4f}, Recall : {:.4f},".format(best_dev_f1, best_dev_prec, best_dev_rec))
        tf.logging.info("- @@@@@ TEST when DEV best F1 : {:.4f}, Precision : {:.4f}, Recall : {:.4f},".format(test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best))
        tf.logging.info("- @@@@@ mention_proposal_only_concate {}".format(config["mention_proposal_only_concate"]))


    if FLAGS.do_predict:
        tp, fp, fn = 0, 0, 0
        epsilon = 1e-10
        for doc_output in estimator.predict(file_based_input_fn_builder(config["test_path"], seq_length, config,
            is_training=False, drop_remainder=False), checkpoint_path=config["eval_checkpoint"],
            yield_single_examples=False): 
            # iterate over each doc for evaluation
            pred_span_label, gold_span_label = mention_proposal_prediction(config, doc_output,concat_only=FLAGS.concat_only)

            tem_tp = np.logical_and(pred_span_label, gold_span_label).sum()
            tem_fp = np.logical_and(pred_span_label, np.logical_not(gold_span_label)).sum()
            tem_fn = np.logical_and(np.logical_not(pred_span_label), gold_span_label).sum()

            tp += tem_tp
            fp += tem_fp
            fn += tem_fn

        p = tp / (tp+fp+epsilon)
        r = tp / (tp+fn+epsilon)
        f = 2*p*r/(p+r+epsilon)
        tf.logging.info("Average precision: {:.4f}, Average recall: {:.4f}, Average F1 {:.4f}".format(p, r, f))



if __name__ == '__main__':
    tf.app.run()






