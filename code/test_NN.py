from nnet_LSTM import *
from nnet_CNN import *
from train_NN import run_NN
import sys
import os
import logging
from util import Progbar
from sklearn.metrics import r2_score
import getpass
from oauth2client.service_account import ServiceAccountCredentials
from GP import GaussianProcess
from scipy.stats.stats import pearsonr   
from itertools import compress
import gspread
from spreadsheet_auth import get_credentials

t = time.localtime()
timeString  = time.strftime("%Y-%m-%d_%H-%M-%S", t)

def initialize_uninitialized_vars(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

if __name__ == "__main__":
    season_len = None
    if len(sys.argv) >= 5:
        season_len = float(sys.argv[4])

    directory = sys.argv[1]   
    CNN_or_LSTM = sys.argv[2]

    weights_directory_name = os.path.join(sys.argv[3],'weights', 'model.weights')
    nn_weights_dir = os.path.expanduser(os.path.join('~/bucket2/nnet_data', weights_directory_name))
    print "Loading weights from this directory: " + nn_weights_dir        

    # load data to memory
    train_data_file = os.path.join(directory, 'train_hists.npz') 
    train_labels_file = os.path.join(directory, 'train_yields.npz')
    test_data_file =  os.path.join(directory, 'test_hists.npz')
    test_labels_file = os.path.join(directory, 'test_yields.npz')

    if CNN_or_LSTM == 'CNN':
        # Create a coordinator
        config = CNN_Config(season_len)
        model= CNN_NeuralModel(config,'net')
        print "CNN......"
    elif CNN_or_LSTM == 'LSTM':
        # Create a coordinator
        config = LSTM_Config(season_len)
        model= LSTM_NeuralModel(config,'net')
        print "LSTM....."
    else:
        print "ERROR: did not specify NN"

    model.summary_op = tf.summary.merge_all()
    model.saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)

    # Launch the graph.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())
    model.saver.restore(sess, nn_weights_dir)
    graph = tf.get_default_graph()
    if CNN_or_LSTM == "LSTM":
        
        op_to_restore = graph.get_tensor_by_name("LSTM/lstm_net/dense/matmul:0")
        model.add_finetuning_layer(op_to_restore, 256, config.loss_lambda, config.lr)

    initialize_uninitialized_vars(sess)

    rows_to_append = []
    rows_to_append.append(['TRANSFER LEARNING - 256 dense layer'])
    rows_to_append.append(['Weights loaded from: ', sys.argv[3]])
    rows_to_append.append(['Test dataset', sys.argv[1]])
    testing_experiment_doc = '' #INSERT NAME OF OUTPUT GOOGLE SHEETS HERE
    run_NN(model, sess, directory, CNN_or_LSTM, config, testing_experiment_doc, rows_to_add_to_google_doc = rows_to_append)
