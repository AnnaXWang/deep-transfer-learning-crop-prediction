This project utilizes Satellite images and Machine Learning algorithms to build models and predict corp yileds. It is originally based on https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction, the project is current based on https://github.com/min-yin-sri/deep-transfer-learning-crop-prediction. 

The paper associated with the work can be found [here] (https://www.dropbox.com/s/ei49eck573yxi6f/deep-transfer-learning.pdf?dl=0)

Follow the steps below to install and run the system:

1. **Check out code**
  * Note the code in the [original github site] (https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction) has a few syntax errors, we have fixed them in our forked repository https://github.com/min-yin-sri/deep-transfer-learning-crop-prediction.
2. **Set up Google Bucket**
  * Create a new Google Bucket (see this [link] (https://cloud.google.com/storage/docs/creating-buckets))
  * Or use an existing Google Bucket, one has been created for this project [here] (https://console.cloud.google.com/storage/browser/wm-crop-yield-sri-2018?project=wm-crop-yield-sri-2018&folder=true&organizationId=true)
  * Assume the bucket name is \<bucket_name>
3. **Mount Google Bucket as local storage**
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
4. **Install GDAL package**
  * This two links [link1] (https://gis.stackexchange.com/questions/9553/installing-gdal-and-ogr-for-python) and [link2] (https://hackernoon.com/install-python-gdal-using-conda-on-mac-8f320ca36d90) have some details of how to install GDAL
  * For MacOS, first install anaconda from [here] (https://conda.io/docs/user-guide/install/macos.html)
  * For Linux, first install anaconda from here: https://conda.io/docs/user-guide/install/linux.html
  * Then set conda in system PATH: ```export PATH=~/anaconda2/bin:$PATH```
  * To verify the installation of anacode: ```conda list```
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
5. **Download satellite images** 
  * The staellite images are from [here] (https://lpdaac.usgs.gov/node/804), see [project paper] (https://www.dropbox.com/s/ei49eck573yxi6f/deep-transfer-learning.pdf?dl=0) for more details.
  * Run
  ```
  python pull_modi.py
  ```
  Store the downloaded images to Google Bucket \<bucket_name>
  
6. **Generate histogram**
  ```
  python histograms.py --help
  ```
  Shows the argument requirements, the following sample command will generate Brazil histograms with given Brazil satellite, temperatory, and cover images:
  ```
  python histograms.py -d brazil_histograms_09142018 -s brazil_sat_s000224_e161231_scl750 -c brazil_cover_s010101_e131231_scl750 -t brazil_temp_s020731_e161231_scl750
  ```
  Note all directories are relative to the local mounted bucket.
  
7. **enerate data set**
  ```
  python make_datasets.py
  ```
  Use option '--help' to see detailed instructions and argument requirements, here is an example:
  ```
  python make_datasets.py data data soy_bean argentina 11 15
  ```
    It generates the training dataset for soy bean producton in Argentina using harvest data from 2011 to 2015.
  
8. **Run training**
  ```
  python train_NN.py train_data_location LSTM 
  ```
  LSTM can also be CNN, they are training architetures.
  nnet_data contains the trained models and logs.
  Training reuslts are also being saved in a google doc spread sheet, see "experiment_doc_name"
  * Test (transfer learning)
  ```
  python test_NN.py
  ```
  ```
  python make_yields.py
  ```
  The harvest data is in code/static_data_file directory
