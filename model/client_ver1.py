import cv2
import io
import socket
import struct
import time
import pickle
import numpy as np
import imutils
from PIL import ImageGrab

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('0.tcp.ngrok.io', 19194))
client_socket.connect(('172.20.10.4', 5555))

cam = cv2.VideoCapture(0)
img_counter = 0
image = ImageGrab.grab()
#encode to jpeg format
#encode param image quality 0 to 100. default:95
#if you want to shrink data size, choose low image quality.
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

while True:
    img_rgb = ImageGrab.grab((960, 0, 1920,1080 ))
    #img_rgb = ImageGrab.grab((960, 540, 1620,780 ))
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    ret, frame = cam.read()
    frame = img_bgr
    # 影像縮放
    #frame = imutils.resize(frame, width=320)
    # 鏡像
    #frame = cv2.flip(frame,360)
    result, image = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(image, 0)
    size = len(data)

    if img_counter%10==0:
        client_socket.sendall(struct.pack(">L", size) + data)
        #cv2.imshow('client',frame)
        
    img_counter += 1

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cam.release()