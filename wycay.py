import os
import shutil

import cv2
import numpy as np
import pandas as pd
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


def predict_imagenet_classes(model: callable, preprocess_function: callable, image:np.ndarray) -> np.ndarray:
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


def find_closest_neighbor_by_features(model: callable, preprocess_function: callable, data_dir: str, image: np.ndarray) -> np.ndarray:
    # initialize the model:
    net = model(weights='imagenet', include_top=False)

    # extract features from the image:
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_function(x)
    image_features = net.predict(x).flatten()

    # setup variables for finding the best match:
    closest_name = 'none'
    closest_value = np.inf

    # read in card data:
    cards = pd.read_csv(os.path.join(data_dir, CARD_DATA_FILE))

    # find the closest match:
    for index, row in cards.iterrows():
        # card image file name:
        image_name = str(row['id']) + '.jpg'

        # read in the card image:
        card_image = cv2.imread(os.path.join(data_dir, image_name))

        # one of the card images is too small and could possibly cause an error:
        if np.shape(card_image)[0] < 435 or np.shape(card_image)[1] < 370:
            continue

        # crop card image:
        cropped = card_image[110:435, 50:370]

        # extract card's features:
        img = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_function(x)
        card_features = net.predict(x).flatten()

        # compare image and card features:
        distance = sum((image_features - card_features)**2)

        # if the distance is a new record, make it the new best match:
        if distance < closest_value:
            closest_name = image_name
            closest_value = distance

        # print name and distance:
        print(f'{index}\t{image_name}\t{distance}')
        print(f'current closest: {closest_name}')

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

    # crop the card portrait:
    image = cv2.imread('cards/2511.jpg')
    cropped = image[110:435, 50:370]

    # # extract features from images using various models:
    # print(extract_features(vgg16.VGG16,
    #                        vgg16.preprocess_input,
    #                        cropped.copy()))
    # print(extract_features(resnet50.ResNet50,
    #                        resnet50.preprocess_input,
    #                        cropped.copy()))
    # print(extract_features(mobilenet.MobileNet,
    #                        mobilenet.preprocess_input,
    #                        cropped.copy()))

    # predict imagenet classes:
    # classes = predict_imagenet_classes(vgg16.VGG16,
    #                                    vgg16.preprocess_input,
    #                                    cropped.copy())

    # print(np.shape(classes))
    # print(np.sum(classes))
    # print(np.sort(classes, axis=-1, kind='quicksort')[990:])

    # # find the closest neighbor of charmander.jpg:
    # print(find_closest_neighbor_by_features(vgg16.VGG16,
    #                                         vgg16.preprocess_input,
    #                                         CARD_DATA_DIR,
    #                                         cv2.imread('charmander.jpg')))
