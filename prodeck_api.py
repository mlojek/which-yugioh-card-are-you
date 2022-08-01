import json
import os

import requests


def get_card_image(card_name, image_url):
    # if image directory does not exist, make one:
    save_path = 'card_images'
    if not (os.path.exists(save_path) and os.path.isdir(save_path)):
        os.mkdir(save_path)

    # get the image and save it:
    response = requests.get(image_url)
    image = response.content
    with open(os.path.join(save_path, card_name), 'wb') as save_file:
        save_file.write(image)


def get_all_cards_info() -> list:
    url = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'

    # get all cards info
    response = requests.get(url)

    # convert the response to a list and return:
    return json.loads(response.content.decode())['data']
