# -*- coding: utf8 -*-

import socket

import io
import os
import scipy.misc
import numpy as np
import six
import time

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils


host = '172.20.10.5'  # 對server端為主機位置
port = 5555
# host = socket.gethostname()
# port = 5000
address = (host, port)

socket01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# AF_INET:默認IPv4, SOCK_STREAM:TCP

socket01.bind(address)  # 讓這個socket要綁到位址(ip/port)
socket01.listen(1)  # listen(backlog)
# backlog:操作系統可以掛起的最大連接數量。該值至少為1，大部分應用程序設為5就可以了
print('Socket Startup')

conn, addr = socket01.accept()  # 接受遠程計算機的連接請求，建立起與客戶機之間的通信連接
# 返回（conn,address)
# conn是新的套接字對象，可以用來接收和發送數據。address是連接客戶端的地址
print('Connected by', addr)

##################################################
print('begin write image file "moonsave.png"')
imgFile = open('test_images/IMG_5781.JPG', 'wb')  # 開始寫入圖片檔
while True:
    imgData = conn.recv(512)  # 接收遠端主機傳來的數據
    if not imgData:
        break  # 讀完檔案結束迴圈
    imgFile.write(imgData)
imgFile.close()
print('image save')
##################################################

conn.close()  # 關閉
socket01.close()
print('server close')

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

category_index = {
    1: {'id': 1, 'name': 'Weed Lambsquarters'},
    2: {'id': 2, 'name': 'Weed Rosemary'},
    3: {'id': 3, 'name': 'Weed Wiregrass'},
    4: {'id': 4, 'name': 'Weed Jagged'},
    5: {'id': 5, 'name': 'Weed Biden'},
    6: {'id': 6, 'name': 'Weed Boardleaf'},
    7: {'id': 7, 'name': 'Weed Shy'},
    8: {'id': 8, 'name': 'Weed Circle'},
    9: {'id': 9, 'name': 'Corn July Multi Side'},
    10: {'id': 10, 'name': 'Corn August'}
}
	  
#label_map  = label_map_util.load_labelmap(FLAGS.class_labels)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)


start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('HE(2)/saved_model/')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')

import time

image_dir = 'test_images'
#image_dir=''

elapsed = []
for i in range(1):
#  image_path = os.path.join(image_dir, 'image(' + str(i + 1) + ').png')
  image_path= os.path.join(image_dir, 'IMG_5781.JPG')
  print(image_path)
  image_np = load_image_into_numpy_array(image_path)
  input_tensor = np.expand_dims(image_np, 0)
  start_time = time.time()
  detections = detect_fn(input_tensor)
  end_time = time.time()
  elapsed.append(end_time - start_time)

  plt.rcParams['figure.figsize'] = [21, 21]

  label_id_offset = 1
  image_np_with_detections = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)
  plt.subplot(1, 1, 1)
  plt.imshow(image_np_with_detections)
  plt.savefig('D:\\trizze\\Desktop\\fin\\' + str(i + 1) +'.JPG', format='jpg')
  

mean_elapsed = sum(elapsed) / float(len(elapsed))
print('Elapsed time: ' + str(mean_elapsed) + ' second per image')

"""

# -*- coding: utf8 -*-

#import socket

# host = socket.gethostname()
# port = 5000
host = '172.20.10.4'  # 對server端為主機位置
port = 5555
address = (host, port)

socket02 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# AF_INET:默認IPv4, SOCK_STREAM:TCP

socket02.connect(address)  # 用來請求連接遠程服務器

##################################
# 開始傳輸
print('start send image')
imgFile = open("fin/1.jpg", "rb")
while True:
    imgData = imgFile.readline(512)
    if not imgData:
        break  # 讀完檔案結束迴圈
    socket02.send(imgData)
imgFile.close()
print('transmit end')
##################################

socket02.close()  # 關閉
print('client close')"""