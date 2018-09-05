from __future__ import absolute_import
from __future__ import print_function
import sys

import itertools
import re
import random
import math
import pickle
from os import listdir, mkdir
from shutil import rmtree

import numpy as np
import tensorflow as tf

SEG_LENGTH = 7
WORD_VEC_LENGTH = 300
FILTERS = 16
LABEL_COUNT = 9

CONV_WINDOW_MIN = 3
CONV_WINDOW_MAX = 5
BATCH_SIZE = 3
POOL_STRIDE_SIZE = 4
POOL_SQ_SIZE = 4


def seg_cnn(features, labels, mode='train'):
    # global LONGEST_SENTENCE
    # conv_units = [None]*5
    fully_con_input = dict()
    concat_list = list()
    for conv_window_count in range(CONV_WINDOW_MIN, CONV_WINDOW_MAX+1):
        fully_con_input[conv_window_count] = [None]*5
        for i in range(0, 5):
            data = tf.reshape(
                features[i], [-1, SEG_LENGTH, WORD_VEC_LENGTH, 1])
            # Kernel size should range from 3 to 5 according to paper
            conv = tf.layers.conv2d(inputs=data, filters=FILTERS,
                                    kernel_size=conv_window_count, padding='same', activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(
                inputs=conv, pool_size=POOL_SQ_SIZE, strides=POOL_STRIDE_SIZE)
            fully_con_input[conv_window_count][i] = pool
            concat_list.append(pool)

    fcl_data = tf.concat(concat_list, 1)
    total_vec = 0
    # for i in range(0, len(concat_list)):
    #     total_vec += int((SEG_LENGTH*float((POOL_SQ_SIZE)/POOL_STRIDE_SIZE**2)) *
    #                      (300*((POOL_SQ_SIZE**2)/POOL_STRIDE_SIZE))*LABEL_COUNT)

    flat_fcl_data = tf.reshape(fcl_data, [-1, 300*60])

    fully_con_layer = tf.layers.dense(
        inputs=flat_fcl_data, units=128, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        fully_con_layer, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    logit = tf.layers.dense(inputs=dropout, units=LABEL_COUNT)
    predictions = {
        'classes': tf.argmax(input=logit, axis=1),
        'probabilities': tf.nn.softmax(logit, name='softmax_tensor')
    }
    onehot_labels = tf.one_hot(indices=tf.cast(
        labels, tf.int32), depth=LABEL_COUNT)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logit)

    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    #     logits=logit, labels=labels)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=labels, logits=logit)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    rmtree('tmp', ignore_errors=True)
    mkdir('tmp')
    tf.logging.set_verbosity(tf.logging.INFO)

    with open('clean_data/segcnn_inp_label_list.pickle', 'rb') as label_handle:
        labels = pickle.load(label_handle)
    with open('clean_data/segcnn_inp_sent_by_seg.pickle', 'rb') as sentence_handle:
        sentences = pickle.load(sentence_handle)

    if len(labels) != len(sentences[0]):
        print(
            f"Uneven label-feature matrix, {len(labels)} : {len(sentences)}, exiting now!")
        exit(1)

    LABEL_COUNT = len(list(set(labels)))

    # for i, sentence in enumerate(sentences):
    #     for j, segment in enumerate(sentences[i]):
    #         for k, word2vec_list in enumerate(sentences[i][j]):
    #             sentences[i][j][k] = np.asarray(sentences[i][j][k])
    #         sentences[i][j] = np.asarray(sentences[i][j])
    #     sentences[i] = np.asarray(sentences[i])

    try:
        tt_split = float(input(
            'Enter train-test split, for example .5 means the training set is' +
            ' 50% of the data, and the rest is test.'))
    except Exception:
        tt_split = .5
        print(f'Invalid entry, using {tt_split}.')

    indexes = range(0, len(labels))
    train_indexes = random.sample(
        indexes, math.floor(float(len(labels)) * tt_split))
    test_indexes = [x for x in indexes if x not in train_indexes]

    # For using list of lists rep
    # train_features = np.asarray([sentences[x] for x in train_indexes])
    # test_features = np.asarray([sentences[x] for x in test_indexes])

    # For using dict of list
    train_features = {0: [], 1: [], 2: [], 3: [], 4: []}
    test_features = {0: [], 1: [], 2: [], 3: [], 4: []}
    for i in train_indexes:
        for j in range(0, 5):
            if not isinstance(sentences[j][i], int):
                train_features[j].append(np.asarray(sentences[j][i]))
            else:
                print(sentences[i][j])
                input()
    for i in test_indexes:
        for j in range(0, 5):
            test_features[j].append(np.asarray(sentences[j][i]))
    for i in range(0, 5):
        train_features[i] = np.asarray(train_features[i])
        test_features[i] = np.asarray(test_features[i])

    train_labels = np.asarray([labels[x] for x in train_indexes])
    test_labels = np.asarray([labels[x] for x in test_indexes])

    med_classifier = tf.estimator.Estimator(model_fn=seg_cnn)
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_features,
        y=train_labels,
        batch_size=train_labels.size,
        num_epochs=15,
        shuffle=True)
    med_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])
    # Evaluate the model and print results

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_features,
        y=test_labels,
        num_epochs=5,
        shuffle=False)

    eval_results = med_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
