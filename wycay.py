import os
import shutil

import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from prodeck_api import make_local_copy, check_local_copy
from config import CARD_DATA_DIR


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
    if not check_local_copy(CARD_DATA_DIR):
        # if invalid:
        if os.path.exists(CARD_DATA_DIR):
            shutil.rmtree(CARD_DATA_DIR)

        # make a local copy of card data:
        make_local_copy(CARD_DATA_DIR)

    # extract features from an image:
    image = cv2.imread('cards/2511.jpg')
    cropped = image[110:435, 50:370]
    print(extract_features_vgg16(cropped))
