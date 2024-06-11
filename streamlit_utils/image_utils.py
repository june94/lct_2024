import cv2
import numpy as np


def read_image(img_file_buffer):
    bytes_data = img_file_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), -1)
    return img
    
