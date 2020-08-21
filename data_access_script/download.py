"""
Downloading Images from Planet

Given a set of latitude/longitude coordinate pairs, included are functions that will assist in downloading an image of the location from the Planet API.

Steps:
    1. Generate a box of coordinates around the location
    2. Using filters, search for scene bands that include our area of interest (AOI)
    3. Submit a subset of these as an order to be clipped to our AOI and composited to fill in any missing pixels
    4. Poll the Planet order API to see if the order is ready to download
    5. Download the composited image from the returned URL

"""
import datetime as dt
import os
import shutil
import time
import requests
from requests.auth import HTTPBasicAuth

import random
import json


MAX_ATTEMPTS = 20
URL = 'https://api.planet.com/compute/ops/orders/v2'


def gen_box_coords(lat, lon, height=0.00450, width=0.00592):
    """
    Args:
        lat (float): latitude in decimal degrees
        lon (float): longitude in decimal degrees
        height (float): height of image in decimal degrees [default = 0.00450, appx 500m at IL latitude]
        width (float): width of image in decimal degrees [default = 0.00592, appx 500m at IL latitude]

    Returns:
        box polygon coordinates
    """

    w = width / 2
    h = height / 2

    # format is [[l, b], [r, b], [r, t], [l, t], [l, b]]

    box_coords = [[lon - w, lat - h],
                  [lon + w, lat - h],
                  [lon + w, lat + h],
                  [lon - w, lat + h],
                  [lon - w, lat - h]]
    
    return box_coords

def search_api(coordinates, start_date, end_date, item_type, clear_percent=95, cloud_cover=.05):
    """
    Args:
        coordinates (list): output of gen_box_coords(), a list of lat/lon pairs
        start_date (string): RFC 3339 date
        end_date (string): RFC 3339 date
        item_type (string): either 'PSScene3Band' or 'PSScene4Band'
        clear_percent (int, 0-100): filter for images at least this clear (for use with PSScene4Band imagery)
        cloud_cover (double, 0-1): filter for images at most this cloudy (for use with PSScene3Band imagery)
        
    Returns:
        A list of item IDs that matched the search filters
    """
    # needs update to handle multiple images
    if len(coordinates) > 1:
        coordinates = [coordinates]
    
    geo_json_geometry = {
        "type": "Polygon",
        "coordinates": coordinates
    }

    # filter for items the overlap with our chosen geometry
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": geo_json_geometry
    }

    # filter images acquired in a certain date range
    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": start_date,
        "lte": end_date
      }
    }

    # filter images based on cloud tolerance
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": cloud_cover
      }
    }
    
    # filter based on total image clarity
    clear_percent_filter = {
        "type": "RangeFilter",
        "field_name": "clear_percent",
        "config": {
            "gte": clear_percent
        }
    }
    
    usable_data_filter = {}
    if item_type == "PSScene3Band":
        usable_data_filter = cloud_cover_filter
    elif item_type == "PSScene4Band":
        usable_data_filter = clear_percent_filter

    # create a filter that combines our geo and date filters
    combined_filter = {
      "type": "AndFilter",
      "config": [geometry_filter, date_range_filter, usable_data_filter]
    }

    # Search API request object
    search_endpoint_request = {
      "item_types": [item_type],
      "filter": combined_filter
    }
    
    attempts = 0

    while attempts < MAX_ATTEMPTS:
        result = \
          requests.post(
            'https://api.planet.com/data/v1/quick-search',
            auth=HTTPBasicAuth(os.environ['PL_API_KEY'], ''),
            json=search_endpoint_request)
        if result.status_code != 429:
            if result.status_code != 200:
                raise Exception(result)
            break            

        # If rate limited, wait and try again
        time.sleep((2 ** attempts) + random.random())
        attempts = attempts + 1
        
    if 'json' not in result.headers.get('Content-Type'):
        raise Exception(f"{result} in search_api()")
        
    ids = []
    for result in result.json()['features']:
        ids.append(result['id'])
    
    return ids

def create_order(coordinates, item_ids, item_type):
    """
    Args:
        coordinates (list): results of gen_box_coords(), a list of lat/lon pairs that define a polygon to clip to
        item_ids (list): a list of Planet item ids to clip and composite together
        item_type (string): either 'PSScene3Band' or 'PSScene4Band', will determine whether 3 or 4 band imagery is returned. 4 band imagery more often respects the usable data (clear_percent) filter.
        
    Returns:
        A string, the UUID of the order that was created
    """
    bundle = ""
    if item_type == "PSScene3Band":
        bundle = "visual"
    elif item_type == "PSScene4Band":
        bundle = "analytic"
                                      
    clip_composite = {
      "name": "clip_composite",
      "products": [
        {
          "item_ids": item_ids,
          "item_type": item_type,
          "product_bundle": bundle
        }
      ],
      "tools": [
        {
          "clip": {
            "aoi": {
              "type": "Polygon",
              "coordinates": [coordinates]
            }
          }
        },
        {
          "composite": {}
        } 
      ]
    }
    
    # Creating an Order: https://developers.planet.com/docs/orders/reference/#operation/orderScene
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        response_orders = requests.post(
            URL,
            auth=HTTPBasicAuth(os.environ['PL_API_KEY'], ''),
            json=clip_composite
        )
        
        if response_orders.status_code != 429:
            if not response_orders.ok:
                raise Exception(response.content)
            break

        # If rate limited, wait and try again
        time.sleep((2 ** attempts) + random.random())
        attempts = attempts + 1
    
    if 'json' not in response_orders.headers.get('Content-Type'):
        raise Exception(f"{response_orders} in create_order()")

    # Return the order UUID
    return response_orders.json()['id']
                                       
def check_order(order_uuid):
    """
    Args:
        order_uuid (string): the id of the Planet order ID we want to poll
        num_loops (int): number of times to check for success (should be more than enough, but may want to change if having issues with more complex toolchains/ordering many images)
        
    Returns:
        A string, the URL where the order can be downloaded
    """
    # setup auth
    session = requests.Session()
    session.auth = (os.environ['PL_API_KEY'], '')
    
    state = 'init'
    success_states = ['success', 'partial']
    while state not in success_states:
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            r = session.get((URL + '/{}').format(order_uuid))
            if r.status_code != 429:
                break
                
            time.sleep((2 ** attempts) + random.random())
            attempts = attempts + 1
            
        if 'json' not in r.headers.get('Content-Type'):
            raise Exception(f"{r} in check_order(), attempts={attempts}")
            
        response = r.json()
        state = response['state']
        if state == 'failed':
            raise Exception(response)
        
        time.sleep(10)
        
    # Return the URL
    for result in response['_links']['results']:
        if 'composite.tif' in result['name']: 
            return result['location']

def download_image(url, out_file):
    """
    Args:
        url (string): the url where the order can be downloaded
        out_file (string): the path where the file should be saved
    
    Returns: 
        nothing
    """
    # setup auth
    session = requests.Session()
    session.auth = (os.environ['PL_API_KEY'], '')
    
    token = url.partition('?token=')[2]
    params = (
        ('token', token),
    )
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        download_response = session.get('https://api.planet.com/compute/ops/download/', params=params, stream=True)
        if download_response.status_code != 429:
            break

        # If rate limited, wait and try again
        time.sleep((2 ** attempts) + random.random())
        attempts = attempts + 1

    if download_response.status_code == 200:
#         print(f'Downloading {out_file}')
        with open(out_file, 'wb') as f:
            download_response.raw.decode_content = True
            shutil.copyfileobj(download_response.raw, f)
                                       