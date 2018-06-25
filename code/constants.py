import numpy as np
import os
HIST_BINS_LIST = [np.linspace(1, 2200, 33),
                  np.linspace(900, 4999, 33), 
                  np.linspace(1, 1250, 33),
                  np.linspace(150, 1875, 33),
                  np.linspace(750, 4999, 33),
                  np.linspace(300, 4999, 33),
                  np.linspace(1, 4999, 33),
                  np.linspace(13000,16500,33),
                  np.linspace(13000,15500,33)]

RED_BAND = 0 # 0-indexed, so this is band 1
NIR_BAND = 1 # Near infrared. 
SW_IR_BAND = 6 # Shortwave infrared
NUM_IMGS_PER_YEAR = 45
NUM_REF_BANDS = 7
NUM_TEMP_BANDS = 2
CROP_SENTINEL = 12
LOCAL_DATA_DIR = './static_data_files/'
GBUCKET = os.path.expanduser('~/bucket2/')
VISUALIZATIONS = 'visualizations'
DATASETS = 'datasets'
BASELINE_DIR = 'baseline_results'

