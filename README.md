This project is based on https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction.

The paper associated with the work can be found [here] (https://www.dropbox.com/s/ei49eck573yxi6f/deep-transfer-learning.pdf?dl=0)

Follow the steps below to install and run the system:

1. Check out code
  * Note the code in the [original github site] (https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction) has a few syntax errors, which can be easily fixed. After we fork out a branch and fixed all syntax errors we'll update the code repository link.
2. Set up Google Bucket
  * Create a new Google Bucket (see this [link] (https://cloud.google.com/storage/docs/creating-buckets))
  * Or use an existing Google Bucket, one has been created for this project [here] (https://console.cloud.google.com/storage/browser/wm-crop-yield-sri-2018?project=wm-crop-yield-sri-2018&folder=true&organizationId=true)
  * Assume the bucket name is \<bucket_name>
3. Mount Google Bucket as local storage
  * Use cloud storage FUSE to mount Google Bucket as a local dictory (see [here] (https://cloud.google.com/storage/docs/gcs-fuse))
  * First install cloud storage FUSE (see [here] (https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md)), for MacOS, go to [this site] (https://osxfuse.github.io/)
  * Then create an account credential by creating a private key JSON file (see [here] (https://cloud.google.com/storage/docs/authentication#service_accounts)), use role "Owner".
  * Downlaod the created private key JSON file and store locally
  * Set environment variable: 
  ```
  export GOOGLE_APPLICATION_CREDENTIALS=<Path_To_JSON_file>
  ```
  * Create a local directory \<local_directory>
  * Mount Google Bucket to local directory:
  ```
  $ gcsfuse <bucket_name> <local_directory>
  ```
  * Verify the mounting by creating a temp directory inside the \<local_directory>, check Google Bucket web console to verify the temp direct is there
4. Download satellite images 
  * Download staellite images from [here] (https://lpdaac.usgs.gov/node/804), see [project paper] (https://www.dropbox.com/s/ei49eck573yxi6f/deep-transfer-learning.pdf?dl=0) for more details.
  * Store the downloaded images to Google Bucket \<bucket_name>
5. Install GDAL package
  * This two links [link1] (https://gis.stackexchange.com/questions/9553/installing-gdal-and-ogr-for-python) and [link2] (https://hackernoon.com/install-python-gdal-using-conda-on-mac-8f320ca36d90) have some details of how to install GDAL
  * For MacOS, first install anaconda from [here] (https://conda.io/docs/user-guide/install/macos.html)
  * Then set conda in system PATH: ```export PATH=~/anaconda2/bin:$PATH```
  * Create a new python environment with conda:
  ```
  conda create -n py27 python=2.7 anaconda
  ```
  * Activiate the environment:
  ```
  source activate py27
  ```
  * Install GDAL:
  ```
  conda install gdal
  ```
6. Run the code
  * First generate histogram:
  ```
  python histograms.py
  ```
  Use option '--help' to see argument requirements
  * Then generate data set:
  ```
  python make_datasets.py
  ```
  Use option '--help' to see detailed instructions and argument requirements, here is an example:
  ```
  python make_datasets.py data data soy_bean argentina 11 15
  ```
  It generates the training dataset for soy bean producton in Argentina using harvest data from 2011 to 2015
  * The function to normalize yield data is this:
  ```
  python make_yields.py
  ```
  The harvest data is in code/static_data_file directory