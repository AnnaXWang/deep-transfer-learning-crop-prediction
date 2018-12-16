import argparse
from nnet_LSTM import *
from nnet_CNN import *
import sys
import os
import logging
from util import Progbar
from sklearn.metrics import r2_score
import getpass
from oauth2client.service_account import ServiceAccountCredentials
#from GP import GaussianProcess
from scipy.stats.stats import pearsonr  
from constants import GBUCKET, DATASETS

#import gspread
#from spreadsheet_auth import get_credentials

t = time.localtime()
timeString  = time.strftime("%Y-%m-%d_%H-%M-%S", t)

AUTH_FILE_NAME = "static_data_files/crop-yield-2049f6b103d2.json"
input_data_dir = None
output_data_dir = None
worksheet = None

def file_generator(inputs_data_path, labels_data_path, batch_size, add_data_path = None, add_labels_path = None, permuted_band = -1):
    current_batch_labels = []
    current_batch_inputs = []
    inputs_data = np.load(inputs_data_path)['data']
    labels_data = np.load(labels_data_path)['data']

    if permuted_band != -1:
        n_training_examples = inputs_data.shape[0]
        examples_permutation = np.random.permutation(n_training_examples)
        inputs_data[:, :, :, permuted_band] = inputs_data[examples_permutation, :, :, permuted_band]

    assert(len(inputs_data) == len(labels_data))
    if add_data_path is not None:
        inputs_data_2 = np.load(add_data_path)['data']
        inputs_data = np.append(inputs_data, inputs_data_2, axis = 0)
    if add_labels_path is not None:
        labels_data_2 = np.load(add_labels_path)['data']
        labels_data = np.append(labels_data, labels_data_2, axis = 0)
    
    while(len(inputs_data) < batch_size):
        #need to pad
        inputs_data = np.append(inputs_data, inputs_data, axis = 0)
        labels_data = np.append(labels_data, labels_data, axis = 0)
        print "appended to pad"
    for idx in xrange(len(inputs_data)):
       current_batch_inputs.append(inputs_data[idx])
       current_batch_labels.append(labels_data[idx]) 
       if len(current_batch_labels) == batch_size:
            yield current_batch_inputs, current_batch_labels
            current_batch_inputs = []
            current_batch_labels = []

def return_train_batch(inputs_data_path, labels_data_path, batch_size, permuted_band = -1):
    inputs_data = np.load(inputs_data_path)['data']
    labels_data = np.load(labels_data_path)['data']
    if permuted_band != -1:
        n_training_examples = inputs_data.shape[0]
        examples_permutation = np.random.permutation(n_training_examples)
        inputs_data[:, :, :, permuted_band] = inputs_data[examples_permutation, :, :, permuted_band]
    while True:
        indices = np.random.randint(0, len(inputs_data), size = batch_size)
        histograms = np.array([inputs_data[x] for x in indices])
        yields = np.array([labels_data[x] for x in indices])
        yield histograms, yields

def create_save_files(train_dir, train_name):
    scores_file = os.path.join(train_dir, 'scores_' + train_name + '.txt')
    sess_file = os.path.join(train_dir, 'weights', 'model.weights')
    train_predictions_file = os.path.join(train_dir, 'train_predictions.npz')
    dev_predictions_file = os.path.join(train_dir, 'dev_predictions.npz')
    test_predictions_file = os.path.join(train_dir, 'test_predictions.npz')
    model_w_file = os.path.join(train_dir, 'model_w.npz')
    model_b_file = os.path.join(train_dir, 'model_b.npz')
    train_feature_file = os.path.join(train_dir, 'train_features.npz')
    dev_feature_file = os.path.join(train_dir, 'dev_features.npz')
    test_feature_file = os.path.join(train_dir, 'test_features.npz')
    return scores_file, sess_file, train_predictions_file, test_predictions_file, dev_predictions_file, model_w_file, model_b_file, train_feature_file, test_feature_file, dev_feature_file

def authorize_gspread(output_google_doc, worksheet_name, create_worksheet = False):
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(AUTH_FILE_NAME, scope)
    gc = gspread.authorize(credentials)
    sh = gc.open_by_url(output_google_doc)
    if create_worksheet:
        wks = sh.add_worksheet(title=worksheet_name, rows="1", cols="20")
    else:
        wks = sh.worksheet(worksheet_name)
    headers = gspread.httpsession.HTTPSession(headers={'Connection': 'Keep-Alive'})
    gc = gspread.Client(auth=credentials, http_session=headers)
    gc.login()
    return wks

def worksheet_append_wrapper(worksheet,row_to_append):
    try:
        worksheet.append_row(row_to_append)
    except:
        time.sleep(3)
        worksheet.append_row(row_to_append)

def end_and_output_results(worksheet, train_RMSE_min, train_R2_max, dev_RMSE_min, dev_R2_max, test_RMSE_min, test_R2_max):
    test_RMSE_to_append = ['best test_RMSE_min', test_RMSE_min]
    test_R2_to_append = ['best test_R2_max',test_R2_max]
    #worksheet_append_wrapper(worksheet, test_RMSE_to_append)
    #worksheet_append_wrapper(worksheet, test_R2_to_append)

    return test_RMSE_min, test_R2_max


def run_NN(model, sess, directory, CNN_or_LSTM, config, output_google_doc, restrict_iterations = None, rows_to_add_to_google_doc = None, permuted_band = -1):
    input_data_dir = os.path.join(GBUCKET, DATASETS, directory)
    train_name = "train_{}_{}_{}_{}_{}_{}".format(CNN_or_LSTM,config.lr, config.drop_out, config.train_step, timeString, sys.argv[1].replace('/','_'))
    train_dir = os.path.expanduser(os.path.join('~/bucket2/nnet_data', getpass.getuser(), train_name))
    output_data_dir = train_dir
    os.mkdir(train_dir)
    train_logfile = os.path.join(train_dir, train_name + '.log')
    logging.basicConfig(filename=train_logfile,level=logging.DEBUG)
    model.writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())

    #google drive set-up
    dataset_name = directory[directory.find('/') + 1:]
    """
    worksheet_name = CNN_or_LSTM + "_" + dataset_name + "_" + timeString
    worksheet = authorize_gspread(output_google_doc, worksheet_name, True)
    if rows_to_add_to_google_doc is not None:
        for row in rows_to_add_to_google_doc:
            worksheet.append_row(row)
    if CNN_or_LSTM == "LSTM":
        worksheet.append_row(['learning rate', 'drop out', 'train step', 'loss_lambda', 'lstm_H', 'dense', 'dataset', 'NN', "NN_output_dir", "data_input_dir", "permuted_band"])
        worksheet.append_row([str(config.lr),str(config.drop_out),str(config.train_step), str(config.loss_lambda), str(config.lstm_H), str(config.dense),sys.argv[1].replace('/','_'), CNN_or_LSTM, train_dir.split('nnet_data/')[-1], sys.argv[1], str(permuted_band)])
    else:
        worksheet.append_row(['learning rate', 'drop out', 'train step', 'loss_lambda', 'dataset', 'NN', "NN_output_dir", "data_input_dir"])
        worksheet.append_row([str(config.lr),str(config.drop_out),str(config.train_step), str(config.loss_lambda), sys.argv[1].replace('/','_'), CNN_or_LSTM, train_dir.split('nnet_data/')[-1], sys.argv[1]])

    worksheet.append_row(['Dataset','ME', 'RMSE', 'R2', 'min RMSE', 'R2 for best RMSE', 'correlation_coeff', 'correlation_coeff var' ,'training loss'])
    """

    scores_file, sess_file, train_predictions_file, test_predictions_file, dev_predictions_file, model_w_file, model_b_file, train_feature_file, test_feature_file, dev_feature_file = create_save_files(train_dir, train_name)

    # load data to memory
    train_data_file = os.path.join(input_data_dir, 'train_hists.npz') 
    train_labels_file = os.path.join(input_data_dir, 'train_yields.npz')
    dev_data_file = os.path.join(input_data_dir, 'dev_hists.npz') 
    dev_labels_file = os.path.join(input_data_dir, 'dev_yields.npz')
    test_data_file =  os.path.join(input_data_dir, 'test_hists.npz')
    test_labels_file = os.path.join(input_data_dir, 'test_yields.npz')
    
    summary_train_loss = []
    summary_eval_loss = []
    summary_RMSE = []
    summary_ME = []
    summary_R2 = []
    
    train_RMSE_min = 1e10
    test_RMSE_min = 1e10
    dev_RMSE_min = 1e10
    
    train_R2_max = 0
    test_R2_max = 0
    dev_R2_max = 0
    
    prev_train_loss = 1e10

    try:
        count = 0
        target = 25
        prog = Progbar(target=target)
        
        #TRAINING PORTION
        for i in range(config.train_step):
            if i==3000:
                config.lr/=10
            if i==8000:
                config.lr/=10
            batch = next(return_train_batch(train_data_file, train_labels_file, config.B, permuted_band = permuted_band))
            _, train_loss, summary, loss_summary = sess.run([model.train_op, model.loss, model.summary_op, model.loss_summary_op], feed_dict={
                model.x: batch[0],
                model.y: batch[1],
                model.lr: config.lr,
                model.keep_prob: config.drop_out
                })
            prog.update(count + 1, [("train loss", train_loss)])
            count += 1
            if (i % 200):
                model.writer.add_summary(summary, i)
            else:
                model.writer.add_summary(loss_summary, i)

            if i % target == 0:
                count = 0
                prog = Progbar(target=target)
                print "finished " + str(i)
                train_pred = []
                train_real = []
                train_features = []
                for batch in file_generator(train_data_file, train_labels_file, config.B, permuted_band = permuted_band):
                    pred_temp, feature, weight, bias = sess.run([model.pred, model.feature, model.dense_W, model.dense_B], feed_dict={
                        model.x: batch[0],
                        model.y: batch[1],
                        model.keep_prob: 1
                        })
                    train_pred.append(pred_temp)
                    train_features.append(feature)
                train_pred=np.concatenate(train_pred)
                train_features = np.concatenate(train_features)
                train_real = np.load(train_labels_file)['data'][:len(train_pred)]
                RMSE=np.sqrt(np.mean((train_pred-train_real)**2))
                ME=np.mean(train_pred-train_real)
                sklearn_r2 = r2_score(train_real, train_pred)

                correlation_coeff = pearsonr(train_pred, train_real)
                if RMSE < train_RMSE_min:
                    train_RMSE_min = RMSE
                    train_R2_max = sklearn_r2
                """
                try:
                    worksheet = authorize_gspread(output_google_doc, worksheet_name)
                except:
                    time.sleep(3)
                    worksheet = authorize_gspread(output_google_doc, worksheet_name)
                """
                 
                print 'Train set','test ME',ME, 'train RMSE',RMSE,'train R2',sklearn_r2,'train RMSE_min', train_RMSE_min,'train R2 for min RMSE',train_R2_max
                logging.info('Train set train ME %f train RMSE %f train R2 %f train RMSE min %f train_R2_for_min_RMSE %f',ME, RMSE, sklearn_r2, train_RMSE_min,train_R2_max)
                
                line_to_append = ['Train',str(ME), str(RMSE), str(sklearn_r2), str(train_RMSE_min), str(train_R2_max), str(correlation_coeff[0]), str(correlation_coeff[1]), str(train_loss)]
                #worksheet_append_wrapper(worksheet, line_to_append)
                
                prev_train_loss = train_loss
                #print scores on dev set
                pred = []
                real = []
                dev_features = []
                for batch in file_generator(dev_data_file, dev_labels_file, config.B, permuted_band = permuted_band):
                    pred_temp, feature = sess.run([model.pred, model.feature], feed_dict={
                        model.x: batch[0],
                        model.y: batch[1],
                        model.keep_prob: 1
                        })
                    pred.append(pred_temp)  
                    dev_features.append(feature)
                pred=np.concatenate(pred)
                dev_features = np.concatenate(dev_features)
                real = np.load(dev_labels_file)['data']
                pred = pred[:len(real)]
                real = real[:len(pred)]
                RMSE=np.sqrt(np.mean((pred-real)**2))
                ME=np.mean(pred-real)
                sklearn_r2 = r2_score(real, pred)
                correlation_coeff = pearsonr(pred, real)
                
                found_min = False
                if RMSE < dev_RMSE_min:
                    print "Found a new dev RMSE minimum"
                    found_min = True
                    dev_RMSE_min = RMSE
                    model.saver.save(sess, sess_file)

                    np.savez(train_predictions_file, data=train_pred)
                    np.savez(dev_predictions_file, data=pred)
                    np.savez(model_w_file, data=weight)
                    np.savez(model_b_file, data=bias)
                    np.savez(train_feature_file, data=train_features)
                    np.savez(dev_feature_file, data=dev_features)
                    
                    dev_R2_max = sklearn_r2

                print 'Dev set', 'dev ME', ME, 'dev RMSE',RMSE, 'dev R2',sklearn_r2,'dev RMSE_min',dev_RMSE_min,'dev R2 for min RMSE',dev_R2_max
                logging.info('Dev set dev ME %f dev RMSE %f dev R2 %f dev RMSE min %f dev_R2_for_min_RMSE %f',ME,RMSE,sklearn_r2,dev_RMSE_min,dev_R2_max)
                
                line_to_append = ['dev',str(ME), str(RMSE), str(sklearn_r2), str(dev_RMSE_min), str(dev_R2_max), str(correlation_coeff[0]), str(correlation_coeff[1])]
                if found_min:
                    line_to_append.append('')
                    line_to_append.append('new dev RMSE min')

                #worksheet_append_wrapper(worksheet, line_to_append)

                #print scores on test set
                test_pred = []
                test_real = []
                test_features = []
                for batch in file_generator(test_data_file, test_labels_file, config.B, permuted_band = permuted_band):
                    pred_temp, feature = sess.run([model.pred, model.feature], feed_dict={
                        model.x: batch[0],
                        model.y: batch[1],
                        model.keep_prob: 1
                        })
                    test_pred.append(pred_temp)  
                    test_features.append(feature)
                test_pred=np.concatenate(test_pred)
                test_features = np.concatenate(test_features)
                test_real = np.load(test_labels_file)['data']
                test_pred = test_pred[:len(test_real)]
                test_real = test_real[:len(test_pred)]
                RMSE=np.sqrt(np.mean((test_pred-test_real)**2))
                ME=np.mean(test_pred-test_real)
                sklearn_r2 = r2_score(test_real, test_pred)
                correlation_coeff = pearsonr(test_pred, test_real)
                
                if found_min:
                    np.savez(test_predictions_file, data=test_pred)
                    test_RMSE_min = RMSE
                    test_R2_max = sklearn_r2

                print 'Test set', 'test ME', ME, 'test RMSE',RMSE, 'test R2',sklearn_r2,'test RMSE_min',test_RMSE_min,'test R2 for min RMSE',test_R2_max
                print
                logging.info('Test set test ME %f test RMSE %f test R2 %f test RMSE min %f test_R2_for_min_RMSE %f',ME,RMSE,sklearn_r2,test_RMSE_min,test_R2_max)
                
                line_to_append = ['test',str(ME), str(RMSE), str(sklearn_r2), str(test_RMSE_min), str(test_R2_max), str(correlation_coeff[0]), str(correlation_coeff[1])]
                #worksheet_append_wrapper(worksheet, line_to_append)

                summary_train_loss.append(str(train_loss))
                summary_RMSE.append(str(RMSE))
                summary_ME.append(str(ME))
                summary_R2.append(str(sklearn_r2))
           
            if restrict_iterations is not None and i == restrict_iterations:
                return end_and_output_results(worksheet, train_RMSE_min, train_R2_max, dev_RMSE_min, dev_R2_max, test_RMSE_min, test_R2_max)

                
    except KeyboardInterrupt:
        print 'stopped'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains neural network architectures.')
    parser.add_argument('dataset_source_dir', help="Directory within {} containing train and test files.".format(os.path.join(GBUCKET, DATASETS)))
    parser.add_argument('nnet_architecture', help='Options: CNN, LSTM')
    parser.add_argument('-s', '--season_frac', type=float, help='Fraction of season data included in each example harvest.')
    parser.set_defaults(season_frac=1)
    parser.add_argument('-p', '--permuted_band', type=int, help='Band to permute.')
    parser.set_defaults(permuted_band=-1)
    parser.add_argument('-it', '--num_iters', type=float, help='Number of training iterations.')

    args = parser.parse_args()
    
    directory = args.dataset_source_dir
    if args.nnet_architecture == 'CNN':
        config = CNN_Config(args.season_frac)
        model= CNN_NeuralModel(config,'net')
        print "Running CNN..."
    elif args.nnet_architecture == 'LSTM':
        config = LSTM_Config(args.season_frac)
        model= LSTM_NeuralModel(config,'net')
        print "Running LSTM..."
    else:
        print 'Error: did not specify a valid neural network architecture. Ending..'
        sys.exit()

    model.summary_op = tf.summary.merge_all()
    model.saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
    
    # Launch the graph.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())

    #INSERT NAME OF RESULTS GOOGLE SHEET HERE
    experiment_doc_name = ""
    run_NN(model, sess, directory, args.nnet_architecture, config, experiment_doc_name, restrict_iterations = args.num_iters, permuted_band = args.permuted_band)
