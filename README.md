## Anomaly Detection for Compressed Air Tube Leak Detection
This project aims to perform anomaly detection for compressed air tube leak detection using the Franhofer IDMT-ISA-COMPRESSED-AIR-TUBELEAK cloud dataset. 
The dataset is processed to extract MFCC (Mel-frequency cepstral coefficients) features using the librosa library. 
A sequential deep neural network (DNN) model is then built and trained using these extracted features. The model achieves a test accuracy of 84%.

# Dataset
The Franhofer IDMT-ISA-COMPRESSED-AIR-TUBELEAK cloud dataset contains audio recordings of compressed air tube systems, both with and without leaks. 

# Feature Extraction
The MFCC features are extracted from the audio recordings using the librosa library in Python.
These features capture the frequency characteristics of the audio signals and are commonly used in speech and audio processing tasks. 
The librosa library provides a convenient way to compute MFCC features from audio signals.
The Extracted MFCC features are stored in CSV file for Anomoly detection "Final.csv" and Multiclass detection "Multiclassnew.csv" 

# Model Architecture
The anomaly detection model is built using a sequential deep neural network (DNN) can be found in the file "Sequential DNN.ipynb". 
The architecture consists of multiple layers of densely connected neurons. 
The model takes the MFCC features as input and learns to classify whether a given audio sample indicates a leak or no leak. 
The model architecture can be further optimized based on the specific requirements and characteristics of the dataset.
The Multiclass heirarchical clalssification model of tube leaks under various environment conditions can be found in the file "Multiclass_Airleakage_Detection.ipynb"

# Training and Evaluation
The dataset is split into training and testing sets to train and evaluate the model. 
The training set is used to optimize the model parameters through backpropagation, while the testing set is used to assess the generalization performance of the model. 
The model is trained using the extracted MFCC features, and the performance is evaluated using test accuracy.

# Results
After training the model with the extracted MFCC features, we achieved a test accuracy of 84%. 
This indicates that the model can accurately classify whether a given audio sample represents a compressed air tube leak or not.
