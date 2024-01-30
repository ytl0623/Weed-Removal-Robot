import os
import pathlib
import tensorflow as tf
import pathlib
import numpy as np
import cv2
from PIL import ImageGrab
import socket
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import imutils
import pickle
import struct
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


# GPU 設定為 記憶體動態調整 (dynamic memory allocation)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PATH_TO_MODEL_DIR = os.path.join(pathlib.Path.home(), 'D:\\trizze\\Desktop\\HE(2)')
PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

# @tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load the COCO Label Map
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

# 使用 webcam
cap = cv2.VideoCapture(0)

# 讀取視訊檔案
#cap = cv2.VideoCapture('./0_video/pedestrians.mp4')
i=0
while True:
    # 讀取一幀(frame) from camera or mp4
    # ret, image_np = cap.read()
    #img_rgb = ImageGrab.grab((130, 80, 720,520  ))
    img_rgb = ImageGrab.grab((0, 0, 860,540  ))
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    image_np = img_bgr
    # 加一維，變為 (筆數, 寬, 高, 顏色)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # 轉為 TensorFlow tensor 資料型態
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    # detections：物件資訊 內含 (候選框, 類別, 機率)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))

    if i==0:
        print(f'物件個數：{num_detections}')
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(int)

    # 第一個 label 編號
    label_id_offset = 1
    
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'] + label_id_offset,
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    
    # 顯示偵測結果
    img = cv2.resize(image_np_with_detections, (800, 600))
    cv2.imshow('object detection', img)
    
    # 按 q 可以結束
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
