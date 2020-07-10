#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import metrics
import logging
import tensorflow as tf
import util
from radam import RAdam
from input_builder import file_based_input_fn_builder
from bert.modeling import get_assignment_map_from_checkpoint


format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


tf.app.flags.DEFINE_string('f', '', 'kernel')

flags = tf.app.flags
flags.DEFINE_string("output_dir", "data", "The output directory of the model training.")
flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("iterations_per_loop", 100, "How many steps to make in each estimator call.")
flags.DEFINE_integer("slide_window_size", 156, "size of sliding window.")
flags.DEFINE_integer("max_seq_length", 200, "Max sequence length for the input sequence.")
flags.DEFINE_string("config_filename", "experiments.conf", "the input config file name.")
flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name "
                       "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not "
                       "specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not "
                       "specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer("num_tpu_cores", 1, "Only used if `use_tpu` is True. Total number of TPU cores to use.")
FLAGS = tf.flags.FLAGS



def model_fn_builder(config):

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        config = util.initialize_from_env(use_tpu=FLAGS.use_tpu)

        tmp_features = {}
        max_f1 = 0 
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
        
        tmp_features["input_ids"] = input_ids 
        tmp_features["input_mask"] = input_mask 
        tmp_features["text_len"] = text_len 
        tmp_features["speaker_ids"] = speaker_ids
        tmp_features["genre"] = genre
        tmp_features["gold_starts"] = gold_starts
        tmp_features["gold_ends"] =  gold_ends
        tmp_features["speaker_ids"] = speaker_ids
        tmp_features["cluster_ids"] = cluster_ids
        tmp_features["sentence_map"] = sentence_map
        tmp_features["span_mention"] = span_mention 
        coref_evaluator = metrics.CorefEvaluator()


        tf.logging.info("********* Features *********")
        for name in sorted(tmp_features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, tmp_features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = util.get_model(config, model_sign="corefqa")

        tvars = tf.trainable_variables()
        # If you're using TF weights only, tf_checkpoint and init_checkpoint can be the same
        # Get the assignment map from the tensorflow checkpoint.
        # Depending on the extension, use TF/Pytorch to load weights.
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, config['tf_checkpoint'])
        init_from_checkpoint = tf.train.init_from_checkpoint # if config['init_checkpoint'].endswith('ckpt') # else load_from_pytorch_checkpoint
          
        if FLAGS.use_tpu:
            tf.logging.info("****************************** Training on TPU ******************************")
            def tpu_scaffold():
                init_from_checkpoint(config['init_checkpoint'], assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            init_from_checkpoint(config['init_checkpoint'], assignment_map)
            scaffold_fn = None 
        

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)


        total_loss, topk_span_starts, topk_span_ends, top_antecedent_scores = model.get_predictions_and_loss(input_ids, input_mask, text_len, speaker_ids, 
                genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, span_mention)

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("****************************** tf.estimator.ModeKeys.TRAIN ******************************")


            if tf.train.get_global_step() // 1000 == 0:
                tf.logging.info("****************************** tf.estimator.ModeKeys.TRAIN ******************************")
                predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold = model.evaluate(topk_span_starts, topk_span_ends, top_antecedent_scores, cluster_ids)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

                p, r, eval_f1 = coref_evaluator.get_prf()
                if eval_f1 >= max_f1:
                    max_f1 = eval_f1 
                tf.logging.info("evaL_f1={}, max_f1={}".format(str(eval_f1), str(max_f1)))

            if config["device"] == "tpu":
                optimizer = tf.train.AdamOptimizer(learning_rate=config['bert_learning_rate'], beta1=0.9, beta2=0.999, epsilon=1e-08)
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step()) 
            else:
                optimizer = RAdam(learning_rate=config['bert_learning_rate'], epsilon=1e-8, beta1=0.9, beta2=0.999)
                train_op = optimizer.minimize(total_loss, tf.train.get_global_step())
        
            training_logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=50)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn, 
                    training_hooks=[training_logging_hook])

        elif mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:

            predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold = model.evaluate(topk_span_starts, topk_span_ends, top_antecedent_scores, cluster_ids)
            
            predictions = {"predicted_clusters": predicted_clusters, 
            "gold_clusters": gold_clusters, 
            "mention_to_predicted": mention_to_predicted,
            "mention_to_gold": mention_to_gold}   
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError
        
        return output_spec

    return model_fn


def main(_):
    config = util.initialize_from_env(use_tpu=FLAGS.use_tpu, config_file=FLAGS.config_filename)

    tf.logging.set_verbosity(tf.logging.INFO)


    num_train_steps = config["num_docs"] * config["num_epochs"]

    # use_tpu = FLAGS.use_tpu
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
        save_checkpoints_steps=config["save_checkpoints_steps"],
        keep_checkpoint_max = 10,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(config)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=1,
        predict_batch_size=1)

    seq_length = config["max_segment_len"] * config["max_training_sentences"]

    if FLAGS.do_train:
        estimator.train(input_fn=file_based_input_fn_builder(config["train_path"], seq_length, config, 
            is_training=True, drop_remainder=True),
            max_steps=num_train_steps)

    if FLAGS.do_eval:
        coref_evaluator = metrics.CorefEvaluator()
        all_results = []
        for result in estimator.predict(file_based_input_fn_builder(config["eval_path"], seq_length, config,is_training=False, drop_remainder=False), yield_single_examples=True):
            all_results.append(result)
            coref_evaluator.update(result["predicted_clusters"], result["gold_clusters"], result["mention_to_predicted"], result["mention_to_gold"])
        p, r, f = coref_evaluator.get_prf()
        tf.logging.info("Average precision: {:.2f}, Average recall: {:.2f}, Average F1 {:.4f}".format(p, r, f))


if __name__ == '__main__':
    tf.app.run()






