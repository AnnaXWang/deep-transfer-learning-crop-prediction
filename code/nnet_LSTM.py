""" Code modified from https://github.com/JiaxuanYou/crop_yield_prediction"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from datetime import datetime
import math

class LSTM_Config():
    B, W, C = 32,32,9
    H = 32 #all season lengths will be 32
    loss_lambda = 0.75
    lstm_layers = 1
    lstm_H = 200
    dense = 356
    season_len = 32

    train_step = 10000000
    lr = 0.005
    #keep probability
    drop_out = 0.75

    def __init__(self, season_frac=None):
        if season_frac is not None:
            self.H = int(season_frac*self.H)
    
def dense(input_data, H, N=None, name = "dense"):
    if not N:
        N = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, H])
        tf.summary.histogram(name + ".W", W)
        tf.summary.histogram(name + ".b", b)
        return tf.matmul(input_data, W, name="matmul") + b

def lstm_net(input_data,output_data,config,keep_prob = 1,name='lstm_net'):
    with tf.variable_scope(name):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(config.lstm_H,state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.lstm_layers,state_is_tuple=True)
        state = cell.zero_state(config.B, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, input_data, 
                       initial_state=state, time_major=True)
        tf.summary.histogram(name + '.outputs', outputs)
        output_final = tf.squeeze(tf.slice(outputs, [config.H-1,0,0] , [1,-1,-1]))
        tf.summary.histogram(name + '.output_final', output_final)
        fc1 = dense(output_final, config.dense, name="dense")

        logit = tf.squeeze(dense(fc1,1,name='logit'))
        tf.summary.histogram(name + '.logit', logit)
        loss_err = tf.nn.l2_loss(logit - output_data)
        loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        total_loss = config.loss_lambda * loss_err + (1 - config.loss_lambda) * loss_reg

        tf.summary.scalar(name + '.loss_err', loss_err)
        tf.summary.scalar(name + '.loss_reg', loss_reg)
        tf.summary.scalar(name + '.loss_total', total_loss)

        return logit,total_loss,fc1

class LSTM_NeuralModel():
    def __init__(self, config, name):
        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name="x")
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.summary_op = None
        self.loss_summary_op = None
        self.writer = None
        self.saver = None

        input_data = tf.transpose(self.x, [2,0,1,3])
        dim = input_data.get_shape().as_list()
        input_data = tf.reshape(input_data,[dim[0],-1,dim[2]*dim[3]])
        print 'lstm input shape',input_data.get_shape()

        with tf.variable_scope('LSTM') as scope:
            self.pred,self.loss,self.feature = lstm_net(input_data, self.y, config, keep_prob=self.keep_prob)
            self.loss_summary_op = tf.summary.scalar("LSTM/lstm_net/loss", self.loss)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        with tf.variable_scope('LSTM/lstm_net/logit') as scope:
            scope.reuse_variables()
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')

    def add_finetuning_layer(self, op_to_restore, config_dense, config_loss_lambda, config_lr):
        with tf.variable_scope('fine-tuning'):
           # op_to_restore = tf.stop_gradient(op_to_restore)
            fc1 = dense(op_to_restore, config_dense, name="dense")
            logit = tf.squeeze(dense(fc1,1,name='logit'))
            tf.summary.histogram('.TLlogit', logit)
            loss_err = tf.nn.l2_loss(logit - self.y)
            loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            total_loss = config_loss_lambda * loss_err + (1 - config_loss_lambda) * loss_reg
            self.pred,self.loss,self.feature = logit,total_loss,fc1
            self.train_op = tf.train.AdamOptimizer(config_lr).minimize(self.loss)
