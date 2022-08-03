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
