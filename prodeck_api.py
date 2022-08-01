import json
import os

import requests


def get_image(image_url) -> object:
    response = requests.get(image_url)
    return response.content


def save_image(save_path, file_name, image) -> None:
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


def process_cards_info(cards_info_list) -> list:
    result = list()

    for card in cards_info_list:
        new_card = dict()
        new_card['id'] = card['id']
        new_card['name'] = card['name']
        new_card['image_url'] = card['card_images'][0]['image_url']

        result.append(new_card)

    return result
