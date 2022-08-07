import json
import os

import requests
import pandas as pd
from tqdm import tqdm

from config import CARD_DATA_FILE


def download_image(image_url: str, save_path: str) -> None:
    'Download and save locally an image from the given url'
    response = requests.get(image_url)

    if response.status_code == 200:
        with open(save_path, 'wb') as save_file:
            save_file.write(response.content)
    else:
        raise requests.HTTPError(f'Could not get data from url {image_url},\
                                 response status code {response.status_code}')


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


def make_local_copy(data_dir: str) -> None:
    'Makes a local copy of all cards info and images'
    # make the local data directory:
    os.makedirs(data_dir)

    # get all cards info:
    cards = get_all_cards_info()

    # save cards data to csv:
    cards.to_csv(os.path.join(data_dir, CARD_DATA_FILE), index=False)

    # get and save all card images, wrapped in a progress bar loop:
    for i in tqdm(range(len(cards)), desc="Downloading images...", ncols=80):
        image_url = cards['image_url'].loc[i]
        image_save_path = os.path.join(data_dir, cards['id'].loc[i] + '.jpg')

        download_image(image_url, image_save_path)


def check_local_copy(data_dir: str) -> bool:
    'Check the existence and validity of a local card data directory'
    # check if the directory exists:
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        return False

    # check if the card data file exists:
    csv_path = os.path.join(data_dir, CARD_DATA_FILE)
    if not os.path.exists(csv_path) or not os.path.isfile(csv_path):
        return False

    # read in the contents of card data file:
    cards_data = pd.read_csv(csv_path)

    # check for every card image:
    for _, card in cards_data.iterrows():
        image_name = str(card['id']) + '.jpg'
        image_path = os.path.join(data_dir, image_name)
        if not os.path.exists(image_path) or not os.path.isfile(image_path):
            return False

    return True
