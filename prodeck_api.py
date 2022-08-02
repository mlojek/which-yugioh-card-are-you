import json
import os

import requests


def get_image(image_url: str) -> object:
    'Get image from the given url, returns raw image data'
    response = requests.get(image_url)

    if response.status_code == 200:
        return response.content
    else:
        raise requests.HTTPError(f'Could not get data from url {image_url}, response status code {response.status_code}')


def save_image(image: object, save_path: str) -> None:
    'Save raw image data to a given file. Use with get_image'
    with open(save_path, 'wb') as save_file:
        save_file.write(image)


def get_all_cards_info() -> list:
    'Get names, ids and image urls of all yugioh cards'
    # get all cards info:
    response = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php')

    # resulting, simpler list of card data:
    result = list()

    # only id, name, and image_url of the card are needed:
    for card in json.loads(response.content.decode())['data']:
        new_card = dict()
        new_card['id'] = card['id']
        new_card['name'] = card['name']
        new_card['image_url'] = card['card_images'][0]['image_url']

        result.append(new_card)

    return result


def save_to_json(collection, save_path: str) -> None:
    'Save a collection (list or dict) to a JSON file'
    with open(save_path, 'w') as save_file:
        save_file.write(json.dumps(collection, indent=4))


def make_local_copy(data_dir: str) -> None:
    'Makes a local copy of all cards info and images'
    # get all cards data:
    # make local data directory:
    # save cards data to json:
    # save all card images:
    pass
