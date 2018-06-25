""" Code modified from https://github.com/JiaxuanYou/crop_yield_prediction"""

import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import time
import scipy.misc
from datetime import datetime
import math

class CNN_Config():
    # there are 32 buckets in the histogram, 32 timesteps per year, and 9 "bands"
    B, W, H, C = 32, 32,32, 9
    
    loss_lambda = 0.5
    train_step = 10000000
    lr = 1e-3
    weight_decay = 0.005
    drop_out = 0.5

    def __init__(self, season_frac=None):
        if season_frac is not None:
            self.H = int(season_frac*self.H)
    
def conv2d(input_data, out_channels, filter_size,stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        tf.summary.histogram(name + ".W", W)
        tf.summary.histogram(name + ".b", b)
        return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b

def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")

def conv_relu_batch(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        tf.summary.histogram(name + ".a", a)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.relu(b)
        tf.summary.histogram(name + ".r", r)
        return r

def dense(input_data, H, N=None, name="dense"):
    if not N:
        N = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, H])
        tf.summary.histogram(name + ".W", W)
        tf.summary.histogram(name + ".b", b)
        return tf.matmul(input_data, W, name="matmul") + b

def batch_normalization(input_data, axes=[0], name="batch"):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
        tf.summary.histogram(name + ".mean", mean)
        tf.summary.histogram(name + ".variance", variance)
        return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, axes=[0,1,2], train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)
        tf.summary.histogram(self.name + ".normed", normed)

        return normed

class CNN_NeuralModel():
    def __init__(self, config, name):

        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name="x")
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        self.conv1_1 = conv_relu_batch(self.x, 128, 3,1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(self.conv1_1, self.keep_prob)
        conv1_2 = conv_relu_batch(conv1_1_d, 128, 3,2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, self.keep_prob)

        conv2_1 = conv_relu_batch(conv1_2_d, 256, 3,1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, self.keep_prob)
        conv2_2 = conv_relu_batch(conv2_1_d, 256, 3,2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, self.keep_prob)

        conv3_1 = conv_relu_batch(conv2_2_d, 512, 3,1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, self.keep_prob)
        conv3_2= conv_relu_batch(conv3_1_d, 512, 3,1, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, self.keep_prob)
        conv3_3 = conv_relu_batch(conv3_2_d, 512, 3,2, name="conv3_3")
        conv3_3_d = tf.nn.dropout(conv3_3, self.keep_prob)

        dim = np.prod(conv3_3_d.get_shape().as_list()[1:])
        flattened = tf.reshape(conv3_3_d, [-1, dim])

        self.feature = dense(flattened, 2048, name="feature")

        self.pred = tf.squeeze(dense(self.feature, 1, name="dense"))
        self.loss_err = tf.nn.l2_loss(self.pred - self.y)

        with tf.variable_scope('dense') as scope:
            scope.reuse_variables()
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')
        with tf.variable_scope('conv1_1/conv2d') as scope:
            scope.reuse_variables()
            self.conv_W = tf.get_variable('W')
            self.conv_B = tf.get_variable('b')

        loss_err_summary_op = tf.summary.scalar("CNN/loss_err", self.loss_err)
        self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss_reg_summary_op = tf.summary.scalar("CNN/loss_reg", self.loss_reg)
        self.loss = config.loss_lambda * self.loss_err + (1 - config.loss_lambda) * self.loss_reg
        loss_total_summary_op = tf.summary.scalar("CNN/loss", self.loss)
        self.loss_summary_op = tf.summary.merge([loss_err_summary_op, loss_reg_summary_op, loss_total_summary_op])

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.summary_op = None

        self.writer = None
        self.saver = None

