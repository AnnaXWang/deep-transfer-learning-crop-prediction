import ee
import time
import sys
from unidecode import unidecode
import argparse
import os.path

BUCKET = 'es262-yields-team'
BUCKET_VM_REL = os.path.expanduser('~/bucket2/')

IMG_COLLECTIONS = ['MODIS/MOD09A1', 'MODIS/006/MYD11A2', 'MODIS/051/MCD12Q1']
IMG_START_DATES = ['2000-02-24', '2002-07-31', '2001-01-01']
IMG_END_DATES = ['2016-12-31', '2016-12-31', '2013-12-31']
IMG_COLLECTION_BANDS = [[0, 1, 2, 3, 4, 5, 6], [0, 4], [0]]
IMG_COLLECTION_CODES = ['sat', 'temp', 'cover']

#IMPORTANT: USA MODIS code optimized for pulling from subset of US states
#represented by the FIPS codes and postcodes below
USA_FIPS_CODES = {
    "29": "MO", "20": "KS", "31": "NE", "19": "IA", "38": "ND", "46": "SD",
    "27": "MN", "05": "AR", "17": "IL", "18": "IN", "39": "OH"
}


REGIONS = ['argentina', 'brazil', 'india', 'usa']
BOUNDARY_FILTERS = [[-74, -52, -54, -21], [-34, -34, -74, 6], [68, 6, 97.5, 37.2], [-80, 32, -104.5, 49]]
FTR_COLLECTIONS = ['users/nikhilarundesai/cultivos_maiz_sembrada_1314', 'users/nikhilarundesai/BRMEE250GC_SIR',
                   'users/nikhilarundesai/India_Districts', 'users/nikhilarundesai/US_Counties']

CLEAN_NAME = lambda r, l: unidecode(r.get('properties').get(l)).lower().translate(None, "'()/&-")
GET_FIPS = lambda r, l: USA_FIPS_CODES[r.get('properties').get(l)].lower()
FTR_KEY_FNS = [
    lambda region: CLEAN_NAME(region, 'partido') + "-" + CLEAN_NAME(region, 'provincia'),
    lambda region: CLEAN_NAME(region, 'NM_MESO') + "-brasil",
    lambda region: CLEAN_NAME(region, 'DISTRICT') + "-" + CLEAN_NAME(region, 'ST_NM'),
    lambda region: CLEAN_NAME(region, 'NAME') + "-" + GET_FIPS(region, 'STATEFP')
]
FTR_FILTER_FNS = [
    lambda region: True,
    lambda region: True,
    lambda region: True,
    lambda region: region.get('properties').get('STATEFP') in USA_FIPS_CODES
]

USAGE_MESSAGE = 'Usage: python pull_modis.py <' + ', '.join(IMG_COLLECTION_CODES) + '> <' + \
  ', '.join(REGIONS) + '>' # + '<folder in bucket/ (optional)>'
NUM_ARGS = 3
NUM_OPTIONAL_ARGS = 1

# Transforms an Image Collection with 1 band per Image into a single Image with items as bands
# Author: Jamie Vleeshouwer
def appendBand(current, previous):
    # Rename the band
    previous=ee.Image(previous)
    current = current.select(IMG_COLLECTION_BANDS[img_collection_index])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum

def export_to_cloud(img, fname, folder, expregion, eeuser=None, scale=500):
  # print "export to cloud"
  expcoord = expregion.geometry().coordinates().getInfo()[0]
  expconfig = dict(description=fname, bucket=folder, fileNamePrefix=fname, dimensions=None, region=expcoord,
                   scale=scale, crs='EPSG:4326', crsTransform=None, maxPixels=1e13)
  task = ee.batch.Export.image.toCloudStorage(image=img.clip(expregion), **expconfig)
  task.start()
  while task.status()['state'] == 'RUNNING':
    print 'Running...'
    time.sleep(10)
  print 'Done.', task.status()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Pull MODIS data for specified countries and imagery types.")
  parser.add_argument("collection_name", choices=IMG_COLLECTION_CODES, help="Type of imagery to pull.")
  parser.add_argument("region_name", choices=REGIONS, help="Identifier of region in GEE local database.")
  parser.add_argument("-t", "--target_folder",type=str, help="Bucket folder where files will ultimately be moved. Checks if file already downloaded. Enter empty string to just check bucket.")
  parser.add_argument("-s", "--scale",type=int,default=500,help="Scale in meters at which to pull; defaults to 500")
  args = parser.parse_args()

  img_collection_index = IMG_COLLECTION_CODES.index(args.collection_name)
  image_collection = IMG_COLLECTIONS[img_collection_index]
  start_date = IMG_START_DATES[img_collection_index]
  end_date = IMG_END_DATES[img_collection_index]
  rgn_idx = REGIONS.index(sys.argv[2])
  ftr_collection = FTR_COLLECTIONS[rgn_idx]
  boundary_filter = BOUNDARY_FILTERS[rgn_idx]
  ftr_key_fn = FTR_KEY_FNS[rgn_idx]
  ftr_filter_fn = FTR_FILTER_FNS[rgn_idx]

  ee.Initialize()
  county_region = ee.FeatureCollection(ftr_collection)

  imgcoll = ee.ImageCollection(image_collection) \
      .filterBounds(ee.Geometry.Rectangle(boundary_filter))\
      .filterDate(start_date,end_date)
  img=imgcoll.iterate(appendBand)
  img=ee.Image(img)

  if img_collection_index != 1: #temperature index <<< is this min max filtering needed?
      img_0=ee.Image(ee.Number(0))
      img_5000=ee.Image(ee.Number(5000))

      img=img.min(img_5000)
      img=img.max(img_0)

  feature_list = county_region.toList(1e5)
  feature_list_computed = feature_list.getInfo()

  keys_with_issues = []
  count_already_downloaded = 0
  count_filtered = 0
  for idx, region in enumerate(feature_list_computed):
    if not ftr_filter_fn(region):
        count_filtered += 1
        continue
    
    subunit_key = ftr_key_fn(region)
    file_name = sys.argv[2] + '_' + sys.argv[1] + '_' + subunit_key + "_" + start_date + "_" + end_date
    if args.target_folder is not None and \
    os.path.isfile(os.path.join(BUCKET_VM_REL, args.target_folder, file_name + '.tif')): 
        print subunit_key, 'already downloaded. Continuing...' 
        count_already_downloaded += 1
        continue
    
    try:
        export_to_cloud(img, file_name, BUCKET, ee.Feature(region), scale=args.scale)
    except KeyboardInterrupt:
        print 'received SIGINT, terminating execution'
        break
    except Exception as e:
      print 'issue with {} ({})'.format(subunit_key, e.message)
      keys_with_issues.append((subunit_key, e.message))

  print 'Successfully ordered', len(feature_list_computed)-len(keys_with_issues)-count_already_downloaded-count_filtered, 'new tifs from GEE'
  print 'Already had', count_already_downloaded
  print 'Failed to order', len(keys_with_issues)
  print 'Filtered', count_filtered
  print 'There were issues with:\n\t' + ',\n\t'.join([(k[0] + ' (' + k[1] + ')') for k in keys_with_issues])
