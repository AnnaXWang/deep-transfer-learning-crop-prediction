from constants import NUM_IMGS_PER_YEAR, NUM_TEMP_BANDS, NUM_REF_BANDS, CROP_SENTINEL, GBUCKET
import numpy as np
import gdal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir, mkdir
from os.path import isfile, join, getsize, expanduser, basename, normpath, isdir
import argparse
SUMMARY_STAT_NAMES = ['max', 'mean', 'min']


def visualize_histogram(hist, title, save_folder, show=False):
    """
    Outputs an image of a histogram's multiple bands for a sanity check
    """
    num_bands = hist.shape[2]
    f, axarr = plt.subplots(num_bands, sharex=True)
    for band in range(num_bands):
        axarr[band].imshow(hist[:,:,band])
    plt.suptitle(title)
    plt.savefig(join(save_folder, title +'.png'))
    if show:
        plt.show()

def return_files(path):
    return [f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]


def analyze_histograms(directory, save_directory):
    hist_files = return_files(directory)
    count = 0
    for idx, f in enumerate(hist_files):
        histograms = np.load(join(directory, f))
        shape = histograms.shape
        if idx == 0:
            num_bins = shape[0]
            num_bands = shape[2]
            histogram_sums = np.zeros(( num_bins, num_bands))
            mode_matrix = np.empty((num_bands,  len(SUMMARY_STAT_NAMES), len(hist_files)))
        histogram_sums += np.sum(histograms, axis=1) #sum along time axis

        place = basename(f)[:-4]
        modes = np.argmax(histograms, axis=0)
        maxes = np.max(modes, axis=0)
        means = np.average(modes, axis=0)
        mins = np.min(modes, axis=0)
        mode_matrix[:, 0, idx] = maxes
        mode_matrix[:, 1, idx] = means
        mode_matrix[:, 2, idx] = mins
        visualize_histogram(histograms, place, save_directory)
        plt.clf()
        
        count += 1
        print place, shape, str(count) + '/' + str(len(hist_files))
        print ' '.join('{:02}'.format(int(m)) for m in maxes), 'max mode bins'
        print ' '.join('{:02}'.format(int(m)) for m in means), 'mean mode bins'
        print  ' '.join('{:02}'.format(int(m)) for m in mins), 'min mode bins'
        print 
    
    #plot summed histograms
    plt.figure()
    for band in range(num_bands):
        plt.subplot(num_bands, 1, band+1)
        plt.bar(list(range(len(histogram_sums[:, band]))), histogram_sums[:, band])
        plt.ylabel(str(band))
        plt.yticks([])
        plt.xticks(list(range(0, num_bins, 2)))
    plt.suptitle('Histogram density sums')
    plt.savefig(join(save_directory, '111_density_sums.png'))

    for band in range(num_bands):
        #plot mode matrix summary stats
        plt.figure()
        for idx, stat_name in enumerate(SUMMARY_STAT_NAMES):
            plt.subplot(len(SUMMARY_STAT_NAMES), 1, idx + 1)
            plt.hist(mode_matrix[band, idx], bins=num_bins)
            plt.ylabel(stat_name)
            plt.xticks(range(0, num_bins, 2))
        save_name = '111_band_' + str(band) + '_mode_visualization.png'
        plt.suptitle('Band ' + str(band) + ' mode histograms')
        plt.savefig(join(save_directory, save_name))
        plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyzes makeup of histograms in target directory. Outputs figures to bucket.')
    parser.add_argument('histogram_directory', type=str,help='Directory of histograms. Bucket prefix automatically added if not included.')
    hist_dir = parser.parse_args().histogram_directory
    if hist_dir.find(basename(normpath(GBUCKET))) == -1: hist_dir = join(GBUCKET, hist_dir) 
    hist_dir_name = basename(normpath(hist_dir))
    save_directory = join(GBUCKET, 'hist_visualizations',hist_dir_name)
    if not isdir(save_directory): mkdir(save_directory)
    analyze_histograms(hist_dir, save_directory)
