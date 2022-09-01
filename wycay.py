import os
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications import vgg16, resnet50, mobilenet

from prodeck_api import make_local_copy, check_local_copy
from crop import dumb_crop
from config import CARD_DATA_DIR, CARD_DATA_FILE


def extract_features(model: callable, preprocess_function: callable, image: np.ndarray) -> np.ndarray:
    '''
    Extract features from a given image using a pretrained model.
    Input shape does not matter, but the output will depend on the model.

    | model     | output len |
    |-----------|------------|
    | vgg16     | 25088      |
    | mobilenet | 50176      |
    | resnet50  | 100352     |
    '''
    # initialize the model:
    net = model(weights='imagenet', include_top=False)

    # resize the image to match the model's input size:
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # necessary preprocessing:
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_function(x)

    # predict the features and return:
    return net.predict(x).flatten()


def predict_imagenet_classes(model: callable, preprocess_function: callable, image: np.ndarray) -> np.ndarray:
    '''
    Predict imagenet set classes for a given image using a given model.
    Regardless of the model used the ouput will always have shape (1000)
    '''
    # initialize the model:
    net = model(weights='imagenet')

    # resize the image to match the model's input size:
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # necessary preprocessing:
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_function(x)

    # predict the features and return:
    return net.predict(x).flatten()


def find_closest(model: callable, preprocess_function: callable, include_top_: bool, data_dir_path: str, image_path: str, crop_function: callable) -> str:
    # initialize the model:
    net = model(weights='imagenet', include_top=include_top_)

    # read, crop and resize the image:
    image = cv2.imread(image_path)
    image = crop_function(image)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # extract features:
    x = np.array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_function(x)
    image_features = net.predict(x, verbose=0).flatten()

    # init best match variables:
    closest_name = 'none'
    closest_value = np.inf

    # read in card data:
    cards = pd.read_csv(os.path.join(data_dir_path, CARD_DATA_FILE))

    # find the closest match:
    for i in tqdm(range(cards.shape[0]), desc="Finding the best match...", ncols=80):
        # read, crop and resize the card image:
        image_name = str(cards['id'].iloc[i]) + '.jpg'
        card_image = cv2.imread(os.path.join(data_dir_path, image_name))
        card_image = crop_function(card_image)
        card_image = cv2.resize(card_image, (224, 224), interpolation=cv2.INTER_LINEAR)

        # extract card's features:
        x = np.array(card_image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_function(x)
        card_features = net.predict(x, verbose=0).flatten()

        # compare image and card features:
        distance = sum((image_features - card_features)**2)

        # if the distance is a new record, make it the new best match:
        if distance < closest_value:
            closest_name = image_name
            closest_value = distance

    # return info about the closest match:
    return closest_name, closest_value


if __name__ == '__main__':
    # if local card data not there/not valid:
    if not check_local_copy(CARD_DATA_DIR):
        # if invalid:
        if os.path.exists(CARD_DATA_DIR):
            shutil.rmtree(CARD_DATA_DIR)

        # make a local copy of card data:
        make_local_copy(CARD_DATA_DIR)

    # vgg16.VGG16
    # vgg16.preprocess_input

    # resnet50.ResNet50
    # resnet50.preprocess_input

    # mobilenet.MobileNet
    # mobilenet.preprocess_input

    print(find_closest(mobilenet.MobileNet,
                       mobilenet.preprocess_input,
                       True,
                       CARD_DATA_DIR,
                       'images/charmander.jpg',
                       dumb_crop))
