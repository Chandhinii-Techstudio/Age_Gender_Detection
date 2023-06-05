# Object Detection using YOLOv3

## Objective: 
The objective of this code is to perform object detection in images and videos using the YOLOv3 pretrained model and OpenCV techniques.

## Installation: 
Ensure that OpenCV is installed on your system along with the necessary dependencies.

## Model Weights: 
Download the YOLOv3 model weights to enable accurate object detection.

## Class Names: 
Provide a text file containing the names of the classes for object detection.

## Image Processing: 
The process_image function resizes and normalizes the input image for YOLOv3.

## Drawing Bounding Boxes: 
The draw function overlays bounding boxes, class names, and detection probabilities on the image.

## Detecting Objects in Images: 
Use the detect_image function to detect objects in a single image.

## Detecting Objects in Videos: 
Use the detect_video function to detect objects in a video file.

## Output Images: 
Detected images with bounding boxes and detection information are saved in the 'res' folder.

## Output Videos: 
Detected videos with bounding boxes and detection information are saved in the 'res' folder.

## Adjusting Confidence Threshold: 
You can modify the confidence threshold (0.6 in this code) to control detection sensitivity.
