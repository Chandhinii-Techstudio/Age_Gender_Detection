# Age and Gender Prediction using RESNET34 model with UTKFace Dataset
This project aims to predict the age and gender of individuals using the UTKFace dataset.
It utilizes the MTCNN face detection algorithm to extract facial features and trains a pre-trained ResNet-34 model for age prediction and gender classification.

## Dataset
The UTKFace dataset is used for training and evaluation. It contains a large number of labeled face images with annotations for age and gender. 
The Folder "MTCNN Dataset Preprocess" contains code "dataset_gen.py" for Dataset preprocess which performs MTCNN feature extraction from UTKFace dataset.

## Train and Test
The Folder "Age_Gender_Detection_Train_Test" contains code for age "age_resnet34.py" and gender "gen_resnet34.py"
The MTCNN extracted features are used as an input to a pretrained RESNET34 model with modification in layers according to age and gender prediction.

## Performance metrics
The Folder "Results" contains the output log files 
The Gender Accuracy and Age's Mean absolute error are found in the file "gender_batch32.log.txt" & "age_modified_batch32.log.txt" respectively

## Inference part
The Folder "Inference_resnet" contain file "Inference_resnet.py" which loads the best saved model check point and predict new unseen inference image.
Using file "app.py" the image from webcam are fetched and inference part run to predict the age and gender in real-time

## Conclusion
This project demonstrates how to use the UTKFace dataset, MTCNN face detection, and a pre-trained ResNet-34 model to predict the age and gender of individuals. 
By training the model on the UTKFace dataset and using the provided inference script, you can apply this model to detect age and gender in real-time using a webcam.

