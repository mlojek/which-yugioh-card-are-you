import os
import shutil

import cv2
import numpy as np
from tensorflow.keras.applications import vgg16, resnet50, mobilenet

from prodeck_api import make_local_copy, check_local_copy
from config import CARD_DATA_DIR


def extract_features_vgg16(image: np.ndarray) -> np.ndarray:
    '''
    Extract features from a given image using a pretrained vgg16 model.
    Input shape does not matter, but the output will always be (25088,).
    '''
    # initialize the model:
    model = vgg16.VGG16(weights='imagenet', include_top=False)

    # resize the image to match the model's input size:
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # necessary preprocessing:
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = vgg16.preprocess_input(x)

    # predict the features and return:
    return model.predict(x).flatten()


def extract_features_resnet50(image: np.ndarray) -> np.ndarray:
    '''
    Extract features from a given image using a pretrained resnet50 model.
    Input shape does not matter, but the output will always be (100352,).
    '''
    # initialize the model:
    model = resnet50.ResNet50(weights='imagenet', include_top=False)

    # resize the image to match the model's input size:
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # necessary preprocessing:
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet50.preprocess_input(x)

    # predict the features and return:
    return model.predict(x).flatten()


def extract_features_mobilenet(image: np.ndarray) -> np.ndarray:
    '''
    Extract features from a given image using a pretrained mobilenet model.
    Input shape does not matter, but the output will always be (50176,).
    '''
    # initialize the model:
    model = mobilenet.MobileNet(weights='imagenet', include_top=False)

    # resize the image to match the model's input size:
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # necessary preprocessing:
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = mobilenet.preprocess_input(x)

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

    # crop the card portrait:
    image = cv2.imread('cards/2511.jpg')
    cropped = image[110:435, 50:370]

    # extract features from the image:
    # print(extract_features_vgg16(cropped))
    # print(extract_features_resnet50(cropped))
    print(extract_features_mobilenet(cropped))
