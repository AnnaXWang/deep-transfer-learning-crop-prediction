#let's get started
import numpy as np
import gdal
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, getsize

TIF_DIRECTORY = 'ARGtif_2012-12-31_2014-12-31/'
BAND_CENTRAL_WL = ["645", "858.5", "469", "555", "1240", "1640", "2130"]
BANDWIDTH = ["620 - 670", "841 - 876", "459 - 479", "545 - 565", "1230 - 1250", "1628 - 1652", "2105 - 2155"]

# 45-46 images per year, 7 bands per image

def return_files(path):
    return [path+f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.') and f.endswith('.tif'))]

def read(tif_path, H, W):
    '''
    Reads the middle HxW image from the tif given by tif_path
    '''
    gdal_dataset = gdal.Open(tif_path)
    # x_size and y_size and the width and height of the entire tif in pixels
    x_size, y_size = gdal_dataset.RasterXSize, gdal_dataset.RasterYSize
    print("TIF Size (W, H): ", x_size, y_size)
    # Mid point minus half the width and height we want to read will give us our top left corner
    if W > x_size:
        raise Exception("Requested width exceeds tif width.")
    if H > y_size:
        raise Exception("Requested height exceeds tif height.")
    gdal_result = gdal_dataset.ReadAsArray((x_size - W)//2, (y_size - H)//2, W, H)
    # If a tif file has only 1 band, then the band dimension will be removed.
    if len(gdal_result.shape) == 2:
        gdal_result = np.reshape(gdal_result, [1] + list(gdal_result.shape))
    # gdal_result is a rank 3 tensor as follows (bands, height, width)
    return np.transpose(gdal_result, (1, 2, 0))

if __name__ == "__main__":
    all_tif_files = return_files(TIF_DIRECTORY)
    for tif_file_path in all_tif_files:
        img = read(tif_file_path, 100, 100)
        print img.shape
        for i in xrange(0,7):
            file_name = tif_file_path[tif_file_path.find("/") + 1:]
            file_name = file_name[:file_name.find("_")]
            plt.title(file_name + ' | Band ' + str(i + 1) + "\n  Central Wavelength: " + BAND_CENTRAL_WL[i] + " | Bandwidth: " + BANDWIDTH[i])
            plt.imshow(img[:,:,i])
            plt.show()


