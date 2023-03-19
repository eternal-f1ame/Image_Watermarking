"""Fetch StockFree Images from Pixabay"""

import requests
import os
import shutil
import json
import time

# API Key
API_KEY = '23472760-4803b4e62cc35fe7f8b11406c'

# Search Term
search_terms = ['cat','dog','owl','building','car']

# Number of Images to Fetch
num_images = 20

# URL
url = 'https://pixabay.com/api/'

for search_term in search_terms:
    # Directory to Save Images
    save_dir = 'test_images/'+search_term

    # Check if Save Directory Exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Parameters
    params = {
        'key': API_KEY,
        'q': search_term,
        'image_type': 'photo',
        'per_page': 20,
        'page': 1
    }

    # Fetch Images
    for _ in range(num_images // 40 + 1):
        print('Fetching Images from Page: {}'.format(params['page']))
        r = requests.get(url, params=params)
        data = json.loads(r.text)
        for i, image in enumerate(data['hits']):
            try:
                r = requests.get(image['largeImageURL'], stream=True)
                if r.status_code == 200:
                    image_path = os.path.join(save_dir, str(i).zfill(3) + '.jpg')
                    with open(image_path, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
            except Exception as e:
                print('Failed to Fetch: {}'.format(image['largeImageURL']))
                raise e

        params['page'] += 1
        time.sleep(1)
