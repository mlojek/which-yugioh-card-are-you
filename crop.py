import random
import os

import cv2
import numpy as np


def show_image(image: np.ndarray, title: str) -> None:
    'Display the image'
    cv2.imshow(title, image)
    cv2.waitKey(0)


def random_jpg(dir_path: str) -> str:
    'Get the name of a random jpg file in the given directory'
    return random.choice([file_name for file_name in os.listdir(dir_path)
                         if os.path.isfile(os.path.join(dir_path, file_name))
                         and file_name.split('.')[-1] == 'jpg'])


def filter_canny(image: np.ndarray) -> np.ndarray:
    'Apply canny filter to the image'
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 30, 300)

    return cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)


def filter_sobel(image: np.ndarray) -> np.ndarray:
    'Apply X and Y sobel filters to the image'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    sobelX = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    # calculate combined sobel:
    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    return cv2.cvtColor(sobelCombined, cv2.COLOR_GRAY2RGB)


def draw_bounding_boxes(image: np.ndarray) -> np.ndarray:
    'Detect contours in the image and draw their bounding boxes'
    # detect contours, can only be done on a grayscale image:
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    (contours, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    # draw bounding boxes on the original image:
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image


def dumb_crop(image: np.ndarray) -> np.ndarray:
    'Crops the card image the dumb way (with constant values)'
    # get image shape:
    image_shape = np.shape(image)

    # if image big enough crop, else just upscale and return
    if image_shape[0] < 435 or image_shape[1] < 370:
        return cv2.resize(image, (325, 320), interpolation=cv2.INTER_LINEAR)
    else:
        return image[110:435, 50:370]
