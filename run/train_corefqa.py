#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
this file contains training and testing the CorefQA model. 
"""

import os 
import math 
import logging
import tensorflow as tf
from utils import util
from utils import metrics
from utils.radam import RAdam
from data_utils.input_builder import file_based_input_fn_builder


tf.app.flags.DEFINE_string('f', '', 'kernel')
flags = tf.app.flags
flags.DEFINE_string("output_dir", "data", "The output directory of the model training.")
flags.DEFINE_string("eval_dir", "/home/lixiaoya/corefqa_output_dir", "The output directory of the saved corefqa models.")
flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to test a model.")
flags.DEFINE_bool("do_predict", False, "Whether to test a model.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("iterations_per_loop", 100, "How many steps to make in each estimator call.")
flags.DEFINE_integer("keep_checkpoint_max", 30, "How many checkpoint models keep at most.")
flags.DEFINE_string("config_filename", "experiments.conf", "the input config file name.")
flags.DEFINE_string("config_params", "train_spanbert_large", "specify the hyper-parameters in the config file.")
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

        model = util.get_model(config, model_sign="corefqa")
    
        if FLAGS.use_tpu:
            tf.logging.info("****************************** Training on TPU ******************************")
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

            total_loss, topk_span_starts, topk_span_ends, top_antecedent_scores = model.get_predictions_and_loss(input_ids, input_mask, text_len, speaker_ids, 
                genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, span_mention)

            if config["tpu"]:
                optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'], beta1=0.9, beta2=0.999, epsilon=1e-08)
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step()) 
            else:
                optimizer = RAdam(learning_rate=config['learning_rate'], epsilon=1e-8, beta1=0.9, beta2=0.999)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step())

            training_logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=1)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn, 
                    training_hooks=[training_logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL: 
            tf.logging.info("****************************** tf.estimator.ModeKeys.EVAL ******************************")
            tf.logging.info("@@@@@ MERELY support tf.estimator.ModeKeys.PREDICT ! @@@@@")
            tf.logging.info("@@@@@ YOU can EVAL your checkpoints after the training process. @@@@@")  
            tf.logging.info("****************************** tf.estimator.ModeKeys.EVAL ******************************")
        
        elif mode == tf.estimator.ModeKeys.PREDICT :
            tf.logging.info("****************************** tf.estimator.ModeKeys.PREDICT ******************************")
            total_loss, topk_span_starts, topk_span_ends, top_antecedent_scores = model.get_predictions_and_loss(input_ids, input_mask, text_len, speaker_ids, 
                genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, span_mention) 
            top_antecedent = tf.math.argmax(top_antecedent_scores, axis=-1)
            predictions = {
                        "total_loss": total_loss, 
                        "topk_span_starts": topk_span_starts,
                        "topk_span_ends": topk_span_ends, 
                        "top_antecedent_scores": top_antecedent_scores,
                        "top_antecedent": top_antecedent,
                        "cluster_ids" : cluster_ids, 
                        "gold_starts": gold_starts, 
                        "gold_ends": gold_ends}   

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, 
                predictions=predictions, 
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Please check the the mode ! ")
        return output_spec
    return model_fn


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
        model_dir=FLAGS.output_dir,
        evaluation_master=FLAGS.master,
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
        checkpoints_iterator = [os.path.join(FLAGS.eval_dir, "model.ckpt-{}".format(str(int(ckpt_idx)))) for ckpt_idx in range(0, num_train_steps+1, config["save_checkpoints_steps"])]
        model = util.get_model(config, model_sign="corefqa")
        for checkpoint_path in checkpoints_iterator[1:]:
            dev_coref_evaluator = metrics.CorefEvaluator()
            for result in estimator.predict(file_based_input_fn_builder(config["dev_path"], seq_length, config, 
                is_training=False, drop_remainder=False), checkpoint_path=checkpoint_path, yield_single_examples=False):
                predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold = model.evaluate(result["topk_span_starts"], result["topk_span_ends"], result["top_antecedent"],
                    result["cluster_ids"], result["gold_starts"], result["gold_ends"])
                dev_coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)            
            dev_prec, dev_rec, dev_f1 = dev_coref_evaluator.get_prf()
            tf.logging.info("***** Current ckpt path is ***** : {}".format(checkpoint_path))
            tf.logging.info("***** EVAL ON DEV SET *****")
            tf.logging.info("***** [DEV EVAL] ***** : precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(dev_prec, dev_rec, dev_f1))
            if dev_f1 > best_dev_f1:
                best_ckpt_path = checkpoint_path
                best_dev_f1 = dev_f1
                best_dev_prec = dev_prec
                best_dev_rec = dev_rec 
                test_coref_evaluator = metrics.CorefEvaluator()
                for result in estimator.predict(file_based_input_fn_builder(config["test_path"], seq_length, config, 
                    is_training=False, drop_remainder=False), checkpoint_path=checkpoint_path, yield_single_examples=False):
                    predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold = model.evaluate(result["topk_span_starts"], result["topk_span_ends"], result["top_antecedent"], 
                        result["cluster_ids"], result["gold_starts"], result["gold_ends"])
                    test_coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

                test_pre, test_rec, test_f1 = test_coref_evaluator.get_prf()
                test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best = test_f1, test_pre, test_rec 
                tf.logging.info("***** EVAL ON TEST SET *****")
                tf.logging.info("***** [TEST EVAL] ***** : precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(test_pre, test_rec, test_f1))

        tf.logging.info("*"*20)
        tf.logging.info("- @@@@@ the path to the BEST DEV result is : {}".format(best_ckpt_path))
        tf.logging.info("- @@@@@ BEST DEV F1 : {:.4f}, Precision : {:.4f}, Recall : {:.4f},".format(best_dev_f1, best_dev_prec, best_dev_rec))
        tf.logging.info("- @@@@@ TEST when DEV best F1 : {:.4f}, Precision : {:.4f}, Recall : {:.4f},".format(test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best))


    if FLAGS.do_predict:
        coref_evaluator = metrics.CorefEvaluator()
        model = util.get_model(config, model_sign="corefqa")
        for result in estimator.predict(file_based_input_fn_builder(config["eval_path"], seq_length, config, 
            is_training=False, drop_remainder=False), yield_single_examples=False):
            
            predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold = model.evaluate(result["topk_span_starts"], result["topk_span_ends"], 
                result["top_antecedent"], result["cluster_ids"], result["gold_starts"], result["gold_ends"])
            coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        
        p, r, f = coref_evaluator.get_prf()
        tf.logging.info("Average precision: {:.4f}, Average recall: {:.4f}, Average F1 {:.4f}".format(p, r, f))



if __name__ == '__main__':
    tf.app.run()






