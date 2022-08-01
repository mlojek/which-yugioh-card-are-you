import json

import requests


def get_all_cards():
    url = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'

    # get all cards info
    response = requests.get(url)

    # convert the response to a list:
    all_cards = json.loads(response.content.decode())['data']

    print(len(all_cards))
