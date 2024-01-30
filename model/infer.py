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
detect_fn = tf.saved_model.load(r'D:\tf2_backup\v26.0_150k\saved_model')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')

import time
image_dir = r'C:\Users\ytlWin\Desktop\1101Linux\\'
elapsed = []

for i in range(15):
  image_path = os.path.join(image_dir, str(i + 1) + '.jpg')
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
  plt.savefig(image_dir + str(i + 1) + '++.jpg', format='jpg')
mean_elapsed = sum(elapsed) / float(len(elapsed))
print('Elapsed time: ' + str(mean_elapsed) + ' second per image')

print( "\nSuccessfully!!!\n" )