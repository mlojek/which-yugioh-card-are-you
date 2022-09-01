import os
import shutil
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications import vgg16, resnet50, mobilenet

from prodeck_api import make_local_copy, check_local_copy
from crop import dumb_crop
from config import CARD_DATA_DIR, CARD_DATA_FILE


def find_closest(model: callable, preprocess_function: callable, include_top_: bool, data_dir_path: str, image_path: str, crop_function: callable) -> str:
    # initialize the model:
    net = model(weights='imagenet', include_top=include_top_)

    # read and resize the image:
    image = cv2.imread(image_path)
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
    # parse CLI arguments:
    parser = argparse.ArgumentParser(description='See which Yu-Gi-Oh! card you look like.')
    parser.add_argument('model', type=str, help='name of NN model to use [vgg16/resnet50/mobilenet]')
    parser.add_argument('--include-top', action='store_true', help='include top of the model')
    parser.add_argument('image_path', type=str, help='path to the input image')
    args = parser.parse_args()

    # check local card data:
    if not check_local_copy(CARD_DATA_DIR):
        # if invalid:
        if os.path.exists(CARD_DATA_DIR):
            shutil.rmtree(CARD_DATA_DIR)

        # make a local copy of card data:
        make_local_copy(CARD_DATA_DIR)

    # select model according to the CLI arg:
    if args.model == 'vgg16':
        model = vgg16.VGG16
        preprocess_fun = vgg16.preprocess_input
    elif args.model == 'resnet50':
        model = resnet50.ResNet50
        preprocess_fun = resnet50.preprocess_input
    elif args.model == 'mobilenet':
        model = mobilenet.MobileNet
        preprocess_fun = mobilenet.preprocess_input
    else:
        raise ValueError('given model name is not a valid option')

    # do the magic:
    print(find_closest(model,
                       preprocess_fun,
                       args.include_top,
                       CARD_DATA_DIR,
                       args.image_path,
                       dumb_crop))
