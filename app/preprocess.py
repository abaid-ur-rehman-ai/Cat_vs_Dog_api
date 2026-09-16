import cv2
import numpy as np

def preprocess_image(file_bytes):
    np_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(1, 256, 256, 3)
    img = img / 255.0
    return img