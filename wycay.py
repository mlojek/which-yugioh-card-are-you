import os
import json
import shutil

import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from prodeck_api import make_local_copy


CARD_DATA_DIR = 'cards'


def check_local_card_data(dir: str) -> bool:
    'Check the existence and validity of a local card data directory'
    # check if the directory exists:
    if not os.path.exists(dir) or not os.path.isdir(dir):
        return False

    # check if the cards_data.json file exists:
    json_path = os.path.join(dir, 'cards_data.json')
    if not os.path.exists(json_path) or not os.path.isfile(json_path):
        return False

    # read in the contents of cards_data.json:
    cards_data = json.load(open(json_path, 'r'))

    # check for every card image:
    for card in cards_data:
        image_name = '{}.jpg'.format(card['id'])
        image_path = os.path.join(dir, image_name)
        if not os.path.exists(image_path) or not os.path.isfile(image_path):
            return False

    return True


def extract_features_vgg16(image: np.ndarray) -> np.ndarray:
    '''
    Extract features from a given image using a pretrained vgg16 model.
    Input shape does not matter, but the output will always be (25088,).
    '''
    # initialize the model:
    model = VGG16(weights='imagenet', include_top=False)

    # resize the image to match the model's input size:
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # necessary preprocessing:
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict the features and return:
    return model.predict(x).flatten()


if __name__ == '__main__':
    # if local card data not there/not valid:
    if not check_local_card_data(CARD_DATA_DIR):
        # if invalid:
        if os.path.exists(CARD_DATA_DIR):
            shutil.rmtree(CARD_DATA_DIR)

        # make a local copy of card data:
        make_local_copy(CARD_DATA_DIR)

    # extract features from an image:
    image = cv2.imread('cards/2511.jpg')
    cropped = image[110:435, 50:370]
    print(extract_features_vgg16(cropped))
