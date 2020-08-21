**Folder contents:**
  
- planet_images_downloader.ipynb: Jupyter notebook to download and view imagery from the Planet API. Loads download.py as a module for helper functions. Must specify .csv of lat/lon pairs.
- download.py: Contains helper functions to search, create an order, poll and order, and download images from the Planet orders API. Tried to bring all tweaks to our API calls into the notebook in order to edit this less and keep the notebook legible.
- create_image_masks.py: script that produces a dictionary of building footprint masks based on the Microsoft Building Footprints dataset given a directory of tiffs. Parameters are:
  - image_path_dir: Full path to image directory  
  - building_footprint_fn: Full path to Microsoft Building Footprint GeoJSON
- process_data.ipynb: processes Driven Data Open AI challenge building footprint data into smaller images and masks for transfer learning
- mask_post_process.ipynb: post-processing for the results of the UNet CAFO segmenations. counts the number of contours in the mask (CAFO pixel clusters) and displays their convex hull
