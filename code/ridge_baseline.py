import matplotlib
matplotlib.use('Agg')
import sys
from constants import HIST_BINS_LIST, NIR_BAND, RED_BAND, GBUCKET, NUM_REF_BANDS,BASELINE_DIR, DATASETS
import numpy as np
import pdb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from collections import Counter
import csv
import random
import math
import os
import getpass
from os import listdir
from os.path import isfile, join, getsize, expanduser, normpath, basename
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import datetime
now = datetime.datetime.now()

sns.set()

def return_files(path):
    return [path+f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]

def get_band_avg(img_hist, bins):
    """
    Returns the weighted reflectance of a band for a histogram of a particular image.
    @img_hist Distribution over bins. Should have dimensions (num_bins)
    @bins Should be the array of delimiters used as bins for the original histogram
    """
    if len(img_hist) != len(bins) - 1:
        raise Exception('Mismatch in histogram and bin list dimensions.')
    total = 0
    for i in range(len(img_hist)):
        add = img_hist[i]*(bins[i + 1] + bins[i])/2
        total += add
    if total == 0: return -1
    return total

def calc_hist_ndvi(img_time_slice):
    """
    Returns the hist_ndvi hist_index for the time slice with dimensions (num_bins, num_bands)
    """
    nir = get_band_avg(img_time_slice[:, NIR_BAND], HIST_BINS_LIST[NIR_BAND])
    red = get_band_avg(img_time_slice[:, RED_BAND], HIST_BINS_LIST[RED_BAND])
    if nir + red == 0: return -1
    return (nir - red)/(nir + red)

def calc_bucket_maxargs(img_time_slice):
    """
    Returns a list of argmax values for all buckets in the time slice.
    """
    argmax_vals = []
    for b in range(img_time_slice.shape[1]):
        argmax_vals.append(np.argmax(img_time_slice[:, b]))
    return argmax_vals

def make_time_series(img_tensor, calc_hist_index_funct):
    """
    Returns an array of indices with dimensions (num_time_steps).
    Assumes that bands NIR_BAND and SW_IR_BAND are present
    @img_tensor Should have dimensions (num_bins, num_time_steps, num_bands)
    @calc_hist_index_funct is the function used to calculate the hist_index
    """
    num_time_steps = img_tensor.shape[1]
    series = []
    for t in range(num_time_steps):
        result = calc_hist_index_funct(img_tensor[:, t, :])
        #if result == -1:
        
        series.append(result)
    return np.array(series)

def get_data(directory):
    hist_train_path = join(directory, 'train_hists.npz')
    ndvi_train_path = join(directory, 'train_ndvi.npz')
    y_train_path = join(directory, 'train_yields.npz')
    
    hist_dev_path = join(directory, 'dev_hists.npz')
    ndvi_dev_path = join(directory, 'dev_ndvi.npz')
    y_dev_path = join(directory, 'dev_yields.npz')
    
    hist_test_path = join(directory, 'test_hists.npz')
    ndvi_test_path = join(directory, 'test_ndvi.npz')
    y_test_path = join(directory, 'test_yields.npz')
    
    y_train = np.load(y_train_path)['data']
    valid_samples = np.where(y_train > 0)
    y_train = y_train[valid_samples]
    hist_train = np.load(hist_train_path)['data'][valid_samples]
    ndvi_train = np.load(ndvi_train_path)['data'][valid_samples]
    assert len(ndvi_train[0].shape) == 1
    
    y_dev = np.load(y_dev_path)['data']
    valid_samples = np.where(y_dev > 0)
    y_dev = y_dev[valid_samples]
    hist_dev = np.load(hist_dev_path)['data'][valid_samples]
    ndvi_dev = np.load(ndvi_dev_path)['data'][valid_samples]
    assert len(ndvi_dev[0].shape) == 1

    y_test = np.load(y_test_path)['data']
    valid_samples = np.where(y_test > 0)
    y_test = y_test[valid_samples]
    hist_test = np.load(hist_test_path)['data'][valid_samples]
    ndvi_test = np.load(ndvi_test_path)['data'][valid_samples]
    assert len(ndvi_test[0].shape) == 1

    return hist_train, ndvi_train, y_train, hist_dev, ndvi_dev, y_dev, hist_test, ndvi_test, y_test

def get_rmse(y_1, y_2):
    return math.sqrt(mean_squared_error(y_1, y_2))

def run_devs(x_train, y_train, x_dev, y_dev, x_test, y_test, run_type, savename, title):
    all_alphas = [2 ** x for x in range(0,20)]
    
    print 'Train set length:', len(x_train)
    print 'Train example length:', x_train[0].shape
    print 'Test set length:', len(x_dev)
    print 'Test example length:', x_dev[0].shape
    print
    print 'Linear regression'
    m = LinearRegression(normalize = False)
    m.fit(x_train, y_train)
    print "R^2 on train = ", m.score(x_train, y_train)
    print "R^2 on dev = ", m.score(x_dev, y_dev)
    print 'RMSE on train =', get_rmse(m.predict(x_train), y_train)
    print 'RMSE on dev =', get_rmse(m.predict(x_dev), y_dev)
    print

    all_train_R2 = []
    all_dev_R2 = []
    all_train_RMSE = []
    all_dev_RMSE = []
    
    print 'Ridge regression'
    for a in all_alphas:
        print 'alpha = ' + str(a)
        clf = Ridge(alpha=a, normalize = False)
        clf.fit(x_train, y_train)
        train_R2 = clf.score(x_train, y_train)
        dev_R2 = clf.score(x_dev, y_dev)
        print "R^2 on train = ", train_R2
        print "R^2 on dev = ", dev_R2
        all_train_R2.append(train_R2)
        all_dev_R2.append(dev_R2)
        train_RMSE = get_rmse(clf.predict(x_train), y_train)
        dev_RMSE = get_rmse(clf.predict(x_dev), y_dev)
        print 'RMSE on train =', train_RMSE 
        print 'RMSE on dev =', dev_RMSE
        all_train_RMSE.append(train_RMSE)
        all_dev_RMSE.append(dev_RMSE)
    
    print
    print '>>>BEST R^2 AND RMSE<<<'
    max_index = np.argmax(all_dev_R2)
    clf = Ridge(alpha=all_alphas[max_index], normalize = False).fit(x_train, y_train) 
    print 'alpha of best dev R^2', all_alphas[max_index]
    print 'Best dev RMSE and R^2', all_dev_RMSE[max_index], all_dev_R2[max_index]
    print 'Train RMSE and R^2', all_train_RMSE[max_index], all_train_R2[max_index]
    print 'Test RMSE and R^2', get_rmse(clf.predict(x_test), y_test), clf.score(x_test, y_test)
    log_alphas = [np.log(x) for x in all_alphas] 
    
    if run_type == "argmax":
        plt.plot(log_alphas, all_train_R2, '-b', label = 'Band mode train')
        plt.plot(log_alphas, all_dev_R2, '-r', label = 'Band mode dev')
        plt.plot([log_alphas[max_index]], [clf.score(x_test, y_test)], 'g.', markersize=20,  label = 'Band mode test')
    else:
        plt.plot(log_alphas, all_train_R2, '--b', label = 'NDVI train')
        plt.plot(log_alphas, all_dev_R2, '--r', label = 'NDVI dev')
        plt.scatter([log_alphas[max_index]], [clf.score(x_test, y_test)], facecolors='none',linewidth=4,edgecolors='g', label = 'NDVI test')
    plt.legend(loc='best')
    plt.ylabel('R$^2$')
    plt.xlabel('$\\log\\ \\alpha$')
    plt.title(title)
    plt.savefig(savename + '.png')

def bin_argmax_feature_transform(x_hists):
    bin_argmax_feature_lists = []
    for x_hist in x_hists:
        argmax_vals = make_time_series(x_hist, calc_bucket_maxargs)
        argmax_vals = argmax_vals.reshape((argmax_vals.shape[0]*argmax_vals.shape[1],))
        bin_argmax_feature_lists.append(argmax_vals)
    return np.array(bin_argmax_feature_lists)

def hist_index_feature_transform(x_hists):
    hist_index_feature_lists = []
    good_results = []
    for i, x_hist in enumerate(x_hists):
        result = make_time_series(x_hist, calc_hist_ndvi)
        if result[0] != -1:
            good_results.append(i) 
        hist_index_feature_lists.append(result)
    return np.array(hist_index_feature_lists), good_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and evaluates baselines.')
    parser.add_argument('dataset_source_dir', help="Directory within {} containing train and test files.".format(join(GBUCKET, DATASETS)))

    args = parser.parse_args()
    
    directory = expanduser(args.dataset_source_dir)
    abs_prefix = join(GBUCKET, DATASETS)
    if directory.find(abs_prefix) == -1:
        directory = join(abs_prefix, directory)
 
    hists_train, ndvi_train,  y_train, hists_dev, ndvi_dev, y_dev, hists_test, ndvi_test, y_test = get_data(directory)

    title = basename(normpath(directory)) + '_' +  now.strftime('%Y-%m-%d_%H:%M')
    savename = join(GBUCKET, BASELINE_DIR, getpass.getuser(), title) 
    print 'Training and testing baseline. Outputting results to {}[.txt|.png]'.format(savename)
    old_stdout = sys.stdout
    sys.stdout = open(savename + '.txt', 'w')
    
    print
    print 'With bin argmax features'
    x_train = bin_argmax_feature_transform(hists_train)
    x_dev = bin_argmax_feature_transform(hists_dev)
    x_test = bin_argmax_feature_transform(hists_test)
    run_devs(x_train, y_train, x_dev, y_dev, x_test, y_test, 'argmax', savename, title)

    #x_train, good_results = hist_index_feature_transform(hists_train)
    #x_train = x_train[good_results]
    #y_train = y_train[good_results]
    #x_test, good_results = hist_index_feature_transform(hists_test)
    #x_test = x_test[good_results]
    #y_test = y_test[good_results]
    
    print
    print 'With NDVI index features'
    if ndvi_train[0] is not None: # not dummy data
        run_devs(ndvi_train, y_train, ndvi_dev, y_dev, ndvi_test, y_test, 'NDVI', savename, title) #don't clear figure before this! this resaves with hist_index features
    sys.stdout.close()
    sys.stdout = old_stdout
    print 'Baseline complete.'





