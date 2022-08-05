import json
import os

import requests
import pandas as pd
from tqdm import tqdm


def get_image(image_url: str) -> bytes:
    'Get image from the given url, returns raw image data'
    response = requests.get(image_url)

    if response.status_code == 200:
        return response.content
    else:
        raise requests.HTTPError(f'Could not get data from url {image_url},\
                                 response status code {response.status_code}')


def save_image(image: object, save_path: str) -> None:
    'Save raw image data to a given file. Use with get_image'
    with open(save_path, 'wb') as save_file:
        save_file.write(image)


def get_all_cards_info() -> pd.DataFrame:
    'Get names, ids and image urls of all yugioh cards'
    # get all cards info from the API:
    response = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php')

    # return data as a pandas dataframe
    # only cards' ids, names and image url are really needed:
    return pd.DataFrame(data=[[card['id'], card['name'], card['card_images'][0]['image_url']]
                              for card in json.loads(response.content.decode())['data']],
                        columns=['id', 'name', 'image_url'],
                        dtype=str)


def save_to_json(collection, save_path: str) -> None:
    'Save a collection (list or dict) to a JSON file'
    with open(save_path, 'w') as save_file:
        save_file.write(json.dumps(collection, indent=4))


def make_local_copy(data_dir: str) -> None:
    'Makes a local copy of all cards info and images'
    # make the local data directory:
    os.makedirs(data_dir)

    # get all cards info:
    cards = get_all_cards_info()

    # save cards data to json:
    save_to_json(cards, os.path.join(data_dir, 'cards_data.json'))

    # save all card images, wrapped in a progress bar loop:
    for i in tqdm(range(len(cards)), desc="Downloading images...", ncols=80):
        image = get_image(cards[i]['image_url'])
        image_save_path = os.path.join(data_dir, str(cards[i]['id']) + '.jpg')
        save_image(image, image_save_path)
