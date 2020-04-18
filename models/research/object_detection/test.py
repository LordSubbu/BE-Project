######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a webcam feed.
# It draws boxes, scores, and labels around the objects of interest in each frame
# from the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import datetime
from firebase import firebase

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import custom_visualization_utils as vis_util
import Please as ocr
import firebase_storage as fb_store

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

#Firebase Initialization
firebase =  firebase.FirebaseApplication("https://python-test-1235f.firebaseio.com/",None)

#Saving Images when Object is detected
IMAGE_SAVE_PATH = 'C:/Users/Navin Subbu/Documents/BEEP/Project/models/research/object_detection/detected'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(4,1280)
ret = video.set(3,720)

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    # cv2.imshow("Test_Video", frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    
    
    # if(randomList>=0):
    # if(len(randomList)>=1):
        # x = vis_util.returnrandomList()
        # # x = vis_util.returnrandomList()[-1]
        # if x > 0.8:
        #     # print("LetsCheck",vis_util.returnrandomList()[-1])
        #     print("LetsCheck",x)
            
        # else :
            # print('Not Detected')
        
        
    # Draw the results of the detection (aka 'visulaize the results')
    capture,frame,confidence = vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    # print(confidence)
    # All the results have been drawn on the frame, so it's time to display it.
    if confidence > 90 :
        print(confidence)
        time = datetime.datetime.now()
        print ("Current date and time : ")
        x = time.strftime("%d-%m-%y_%I:%M %p")
        y = time.strftime("%d_%b_%I-%M%p")
        print(x)
        
        child_path = time.strftime("/%B/%d")
        image_path = os.path.join(IMAGE_SAVE_PATH,childpath)
        os.mkdir(path)
        local_image_path = time.strftime("/%B/%d")
        local_image_name = "CAP_{}.png".format(y)
        img_name = "{0}/{1}/CAP_{2}.png".format(IMAGE_SAVE_PATH,local_image_path,y)
        cv2.imwrite(img_name, capture)
        print("{} written!".format(img_name))
        truth = False
        text = ocr.static_image_ocr(capture,truth)
        print(text)
        result = firebase.post('/Detected', text)
        print(result)
        
        # fb_store.firebase_store(local_image_path,local_image_name)
        
    else:
        print('No')
        
    cv2.imshow('Object detector', frame)
    

    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

