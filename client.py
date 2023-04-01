import requests
import cv2
import numpy as np
import win32gui, win32ui, win32con
from windowcapture import WindowCapture
from io import BytesIO
from PIL import Image
import time
# Define the endpoint URL
url = "http://127.0.0.1:5000/predict"

test_img = cv2.imread('img.jpg')
win_cap = WindowCapture('MuMu-1')



def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".jpg", arr)
    io_buf = BytesIO(buffer)
    print(type(io_buf))
    return io_buf.read()

# binary_image = numpy_to_binary(test_img)
 
# print(type(binary_image))  

for i in range(4):

    screen_shot = win_cap.get_screenshot()

    img_bytes = cv2.imencode('.jpg',screen_shot)[1].tobytes()
    data = BytesIO(img_bytes)
    response = requests.post(url, data=data)
    print(response.json())
    time.sleep(1)