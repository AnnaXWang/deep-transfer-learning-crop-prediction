from constants import HIST_BINS_LIST, NUM_IMGS_PER_YEAR, NUM_TEMP_BANDS, NUM_REF_BANDS, CROP_SENTINEL, GBUCKET, RED_BAND, NIR_BAND
import numpy as np
import gdal
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from os import listdir, mkdir
from os.path import isfile, join, getsize, expanduser, basename, isdir
import argparse

NUM_YEARS_EXTEND_FORWARD = 3 # to extend from 2013 to 2016

def return_tif_filenames(path):
    return [f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.') and f.endswith('.tif'))]


def read_tif(tif_path):
    """
    Reads tif image into a tensor
    """
    try:
        gdal_dataset = gdal.Open(tif_path)
    except:
        print 'Error opening', tif_path, 'with gdal'
        return None
    if gdal_dataset is None:
    	print 'gdal returned None from', tif_path
    	return None
    gdal_result = gdal_dataset.ReadAsArray().astype(np.uint16)
    # If a tif file has only 1 band, then the band dimension will be removed.
    if len(gdal_result.shape) == 2:
        gdal_result = np.reshape(gdal_result, [1] + list(gdal_result.shape))
    # gdal_result is a rank 3 tensor as follows (bands, height, width). Transpose.
    return np.transpose(gdal_result, [1, 2, 0])


def calc_histograms(image, bin_seq_list):
    """
    Makes a 3D tensor of pixel histograms [normalized bin values, time, band]
    input is a 3D image in tensor form [H, W, time/num_bands + band]
    """
    num_bands = len(bin_seq_list)
    num_bins = len(bin_seq_list[0])-1
    if image.shape[2] % num_bands != 0:
        raise Exception('Number of bands does not match image depth.')
    num_times = image.shape[2]/num_bands
    hist = np.zeros([num_bins, num_times, num_bands])
    for i in range(image.shape[2]):
        band = i % num_bands
        density, _ = np.histogram(image[:, :, i], bin_seq_list[band], density=False)
        total = density.sum() # normalize over only values in bins
        hist[:, i / num_bands, band] = density/float(total) if total > 0 else 0
    return hist


def calc_ndvi(sat_tensor):
    num_times = sat_tensor.shape[2]/NUM_REF_BANDS
    sat_tensor = np.transpose(sat_tensor.reshape(-1, sat_tensor.shape[2]), [1, 0]) #bands*time, flattened image
    ndvi_arr = np.empty([num_times])
    for t in range(num_times):
        offset = t*NUM_REF_BANDS
        sat_slice = sat_tensor[[offset + RED_BAND, offset + NIR_BAND], :]
        zeros_mask = np.where((sat_slice[0] > 0) + (sat_slice[1] > 0))
        sat_slice = np.squeeze(sat_slice[:, zeros_mask]).astype(np.float32) # remove indices where NIR = RED = 0
        red = sat_slice[0]
        nir = sat_slice[1]
        ndvi_arr[t] = np.average((nir - red)/(nir + red))
    return ndvi_arr


def mask_image(img, mask, num_bands, num_years_extend_backward, num_years_extend_forward):
    """
    Masks away non-crop pixels in all 2D slices of 3D image tensor of shape X x Y x (bands/time)
    """
    num_imgs = img.shape[2]/num_bands
    assert num_imgs == int(num_imgs)
    remainder_imgs = num_imgs % NUM_IMGS_PER_YEAR
    for t in range(num_imgs):
        mask_year = int((t-remainder_imgs)/NUM_IMGS_PER_YEAR)
        if mask_year < num_years_extend_backward:
            mask_slice = mask[:,:,0]
        elif mask_year >= mask.shape[2] + num_years_extend_backward:
            assert mask_year < mask.shape[2] + num_years_extend_backward + num_years_extend_forward
            mask_slice = mask[:,:,-1]
        else:
            mask_slice = mask[:, :, mask_year - num_years_extend_backward] 
        for b in range(num_bands):
            img[:, :, t*num_bands + b] = np.multiply(img[:, :, t*num_bands + b], mask_slice)
    return img



def get_places(filenames):
    """
    Gets places of tif imagery from filenames in list. Assumes names are of form
    <country>_<img type>_<place name>.tif|_<date info>.tif
    """
    places = []
    for f in filenames:
        place = f.split('_')[2]
        if place.find('tif') != -1: place = place[:-1]
        places.append(place)
    return places

def collect_tif_path_dict(sat_dir, temp_dir, mask_dir, verbose=True):
    """
    Returns a dictionary of form {name of place : (sat path, temp path, mask path)}
    """
    all_sat_files = return_tif_filenames(sat_dir)
    sat_places = get_places(all_sat_files)
    all_temp_files = return_tif_filenames(temp_dir)
    temp_places = get_places(all_temp_files)
    all_mask_files = return_tif_filenames(mask_dir)
    mask_places = get_places(all_mask_files)
    tif_dict = {}
    for s_i, place in enumerate(sat_places):
        if place not in temp_places or place not in mask_places:
            if verbose: print place, 'missing temp and/or mask file'
            continue
        t_i = temp_places.index(place)
        m_i = mask_places.index(place)
        tif_dict[place] = (all_sat_files[s_i], all_temp_files[t_i], all_mask_files[m_i])
    return tif_dict


if __name__ == '__main__':
    gdal.SetCacheMax(2**35)

    parser = argparse.ArgumentParser(description='Sorts histograms, yield data, and location information into usable datasets.')
    parser.add_argument("-d", "--target_dir", help="Directory within {} to store output files".format(GBUCKET), required=True)
    parser.add_argument("-s", "--sat", help="Directory within {} containing satellite data".format(GBUCKET), required=True)
    parser.add_argument("-c", "--cover", help="Directory within {} containing cover data".format(GBUCKET), required=True)
    parser.add_argument("-t", "--temp", help="Directory within {} containing temperature data".format(GBUCKET), required=True)
    parser.add_argument("-i", "--indices_only",help="Only calculate indices.",dest="indices_only",action="store_true")
    parser.set_defaults(indices_only=False)

    args = parser.parse_args()

    print 'Creating histograms'
    print 'Note that the final histograms will begin at the temperature start date'
    print 'We assume that the temperature and reflectance imagery end on the same date, 2016/12/31'
    print 'Keep this in mind when creating offsets later on in the pipeline'
    print 'Mask will be extended into past and future in order to match reflectance and temperature imagery'
   
    indices_only = args.indices_only
    sat_directory = join(GBUCKET, args.sat + '/')
    temp_directory = join(GBUCKET, args.temp + '/')
    mask_directory = join(GBUCKET, args.cover + '/')
    target_folder_name = join(GBUCKET, args.target_dir)
    if not isdir(target_folder_name): mkdir(target_folder_name)

    
    tif_dict = collect_tif_path_dict(sat_directory, temp_directory, mask_directory)

    count = 0
    num_tifs = len(tif_dict)
    for place, tif_path_tuple in tif_dict.iteritems():
        hist_save_path = join(target_folder_name, place + '_histogram')
        ndvi_save_path = join(target_folder_name, place + '_ndvi')

        if isfile(hist_save_path + '.npy') and isfile(ndvi_save_path + '.npy'): 
            print place, 'already processed. Continuing...'
            count += 1
            continue 
        
        sat_path, temp_path, mask_path = tif_path_tuple
        sat_tensor = read_tif(sat_directory + sat_path) 
        assert sat_tensor is not None
        temp_tensor = read_tif(temp_directory + temp_path)
        assert temp_tensor is not None
        mask_tensor = read_tif(mask_directory + mask_path) 
        assert mask_tensor is not None

        if sat_tensor.shape[:2] != temp_tensor.shape[:2] or\
           sat_tensor.shape[:2] != mask_tensor.shape[:2]:
               print place, 'slice shapes do not match! sat, temp, mask shapes:', sat_tensor.shape[:2], temp_tensor.shape[:2], mask_tensor.shape[:2]
               count += 1
               continue

        mask_tensor[mask_tensor != CROP_SENTINEL] = 0
        mask_tensor[mask_tensor == CROP_SENTINEL] = 1

        num_sat_imgs_orig = sat_tensor.shape[2]/NUM_REF_BANDS
        num_temp_imgs = temp_tensor.shape[2]/NUM_TEMP_BANDS
 
        # assume that temperature range shorter than sat
        sat_tensor = sat_tensor[:, :, NUM_REF_BANDS*(num_sat_imgs_orig - num_temp_imgs):]
        
        mask_years_missing = int(num_temp_imgs/NUM_IMGS_PER_YEAR) - NUM_YEARS_EXTEND_FORWARD - mask_tensor.shape[2]

        sat_tensor = mask_image(sat_tensor, mask_tensor, NUM_REF_BANDS, mask_years_missing, NUM_YEARS_EXTEND_FORWARD)
        ndvi = calc_ndvi(sat_tensor)
        
        np.save(ndvi_save_path, ndvi)

        if not indices_only:
            temp_tensor = mask_image(temp_tensor, mask_tensor, NUM_TEMP_BANDS, mask_years_missing, NUM_YEARS_EXTEND_FORWARD)
            sat_histograms = calc_histograms(sat_tensor, HIST_BINS_LIST[:NUM_REF_BANDS]) 
            temp_histograms = calc_histograms(temp_tensor,HIST_BINS_LIST[NUM_REF_BANDS:NUM_REF_BANDS+NUM_TEMP_BANDS])
            histograms = np.concatenate((sat_histograms, temp_histograms), axis=2)
            np.save(hist_save_path, histograms)

        
        count += 1
        print "{} {}/{}".format(place, count, num_tifs)
    
    print 'Generated', count, 'histogram' if count == 1 else 'histograms'

