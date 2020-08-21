# Data Science for Social Good + RegLab
Team members: Seiji Eicher, David Kang, Sandy Lee 

## Overview
This is the GitHub repository for Stanford DSSG2020 RegLab Project. Our goal is to use machine learning and deep model techniques to locate possible expansion of Concentrated Animal Farming Operations (CAFO). This is done in two steps: (1) applying machine learning and deep model techniques to identify and segment CAFOs from satellite images and (2) detecting progression of CAFOs by comparing their predicted masks from different dates.

Our [report](https://github.com/StanfordDataScience/dssg-reglab/blob/master/weekly_task/final_report_draft.pdf) contains a detailed summary of the project.

## Environment
All of the necessary environment information is stored in the environment.yml file. The environment could then be activated by
```
conda activate dssg
```

## Dev Setup
We are using a Microsoft VM. See Slack Channel for more details.

### Accessing Jupyter Notebooks

After running

```
jupyter notebook
```

Point your browser at https://13.82.199.8:8000/

## Data
Our data is available on the VM.
* 112 satellite images of CAFO locations in Illinois with masks ('ground truth' labels)
* 229 coordinates of suspected CAFO locations from Illinois
* 2000 coordinates of suspected CAFO locations from Iowa
* Microsoft building footprint polygon geometries in Illinois and Iowa

The data used to train the model is currently in 'datadrive/data/raw/ground_truth/'. The Driven Data dataset can be found in 'datadrive/train/'. Images to build the null distribution can be found in 'datadrive/data/raw/planet_images_il-2019-07/' and 'datadrive/data/raw/planet_images_il-2020-07/'.

## Data Preprocessing
Before training our model, we need to preprocess the data in order to get an input suitable for the model. We use the Planet API to download satellite images of CAFO coordinates. We then create masks for the satellite images using the Microsoft building footprint shapefiles; this is our artificial 'ground truth' labels. /data_access_script directory contains the scripts used for data preprocessing.

## Transfer Learning
The transfer learning script can be run by running the shell script in segmentation/script/driven_script.sh. Parameters can be shifted in the shell file. The DrivenData dataset is currently in 'datadrive/train/masks' and 'datadrive/train/images'.

## Model/Training
For baseline image segmenation, we use the PCA and K-means algorithms. The scripts for running these algorithms on our satellite imagery are available in /notebooks. Our main model is the UNet fully convolutional network. The model can be found in segmenation/models. The training all occurs in a Jupyter Notebook due to the relative speed of training (~20-30 minutes) and can be found in segmentation/notebooks/cafo_segmentation_training.ipynb.

## Prediction
We take the trained model, and generate prediction masks for Illinois CAFO locations at different time points. Then, we compare predicted masks of a location with progression using various metrics, and comparing these numbers with metrics from locations with no progression (i.e., a null distribution). The file used to generate the null can be found in segmentation/notebooks/calculating_null.ipynb, and the file used to assess the null can be found in segmentation/notebooks/assessing_null_on_progressed_images.ipynb. Test set prediciton occurs in the file segmentation/notebooks/segmenter_test_set_evaluation.ipynb

## Adding new images to training
First, to gather images from Planet use the planet_images_downloader.ipynb. Then create the corresponding masks with the Microsoft Building Footprints. Then these two pairs need to be converted into pickle files in order to be placed into /datadrive/data/raw/ground_truth as part of the dataset. The process to do this can be seen in segmentation/notebooks/converting_tiff_to_pickle_for_training.ipynb. What is required is the directory of satellite images and their tiff files, and the directory of corresponding masks, which is a single pickle file with the directory of the aforementioned images as the key and the masks as the value. The new pickle files generate are dictionaries with the following format

```
'image' : PIL image of the specified tiff file
'masks'
    'CAFO Shed' : A tensor of the size of the original image corresponding with pixels equal to 1 in the mask.
    'BACKGROUND': A tensor of the size of the original image corresponding with pixels equal to 0 in the mask.
    'Lagoon' : Saved as a zero tensor of the shape of the tiff file. Largely ignored
```

## Future Use Cases
1. General Building Permit Violations (e.g. expansion of in-law apartments in San Francisco): A Building Footprint Change Point detection model will likely be useful in many other applications where building expansion over time is unpermitted or non-compliant. One example might be unpermitted expansion of in-law apartments in San Francisco.
