import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import random
import tensorflow as tf
import scipy.io
import math
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import save_img
from tensorflow.keras.utils import img_to_array

detector = MTCNN()

if not os.path.exists("/app/DatasetKinetosis/outputData/csv"):
    os.mkdir("/app/DatasetKinetosis/outputData/csv")
    
if not os.path.exists("/app//DatasetKinetosis/outputData/images"):
    os.mkdir("/app/DatasetKinetosis/outputData/images")
    
header = ['imageId', 'gender', 'age']
filename = '/app/DatasetKinetosis/outputData/csv/cropped_dataset.csv'

# stores total number of images
total = 0

with open(filename, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    
    for image in os.listdir('/app/DatasetKinetosis/inputDataUTKFace/UTKface/'):
        print("image Name: ", image)
        try:
            split = image.split('_')
            age = int(split[0])
            gender = int(split[1])
            img = cv2.imread('/app/DatasetKinetosis/inputDataUTKFace/UTKface/' + image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # using MTCNN to detect the face bounding box in the image
            detected_face = detector.detect_faces(img)

            # if face is detected
            if detected_face != []:
                # increment the image count in dataset
                total += 1
                data = []
                if (total%100 == 0):
                    print("Converted {} files!".format(total))

                for face in detected_face:
                    x , y  = max(face['box'][0],0),max(face['box'][1],0)
                    width, height = min(face['box'][2],img.shape[1]-x),min(face['box'][3],img.shape[0]-y)
                    # crop the face
                    face_img = tf.image.crop_to_bounding_box(img, y, x,height,width)

                    # resize image
                    face_img = tf.image.resize(face_img, (120,120), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
                    face_img = tf.dtypes.cast(face_img, tf.int32)
                    # save image in numpy array
                    img_array = img_to_array(face_img)
                    imagePath = '/app/DatasetKinetosis/outputData/images/file_' + str(total) + '.jpg'
                    imageId = 'file_' + str(total) + '.jpg'
                    save_img(imagePath , img_array)
                    # append data in dataset
                    data= [imageId, gender, age]
                    writer.writerow(data)

        except:
            print("An exception occurred in image: ", image)
            pass