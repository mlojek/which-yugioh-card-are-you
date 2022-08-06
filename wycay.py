import os
import shutil

import cv2
import numpy as np
from tensorflow.keras.applications import vgg16, resnet50, mobilenet

from prodeck_api import make_local_copy, check_local_copy
from config import CARD_DATA_DIR


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

    # extract features from images using various models:
    print(extract_features(vgg16.VGG16,
                           vgg16.preprocess_input,
                           cropped.copy()))
    print(extract_features(resnet50.ResNet50,
                           resnet50.preprocess_input,
                           cropped.copy()))
    print(extract_features(mobilenet.MobileNet,
                           mobilenet.preprocess_input,
                           cropped.copy()))
