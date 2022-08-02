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


def save_image(save_path: str, file_name: str, image: object) -> None:
    # check if save_path exists, if not create the directory:
    if not (os.path.exists(save_path) and os.path.isdir(save_path)):
        os.makedirs(save_path)

    # save the image:
    with open(os.path.join(save_path, file_name), 'wb') as save_file:
        save_file.write(image)


def get_all_cards_info() -> list:
    url = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'

    # get all cards info
    response = requests.get(url)

    # convert the response to a list and return:
    return json.loads(response.content.decode())['data']


def process_cards_info(cards_info_list: list) -> list:
    result = list()

    # only id, name, and image_url of the card is needed:
    for card in cards_info_list:
        new_card = dict()
        new_card['id'] = card['id']
        new_card['name'] = card['name']
        new_card['image_url'] = card['card_images'][0]['image_url']

        result.append(new_card)

    return result


def save_to_json(collection, save_path: str) -> None:
    with open(save_path, 'w') as save_file:
        save_file.write(json.dumps(collection, indent=4))
