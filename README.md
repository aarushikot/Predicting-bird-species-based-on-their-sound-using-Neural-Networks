## Project Title
Predicting Bird Species Based on Their Sound Using Neural Networks

## Abstract
This project leverages custom Convolutional Neural Networks (CNNs) to classify bird species based on their vocalizations. Spectrogram data from bird call recordings was used for both binary and multi-class classification tasks, achieving up to 100% accuracy. Data was sourced from the Xeno-Canto database, focusing on bird species commonly found in the Seattle area.

## Features
Binary Classification: Differentiates between two species: American Crow (amecro) and Barn Swallow (barswa).
Multi-Class Classification: Identifies 12 bird species using spectrogram data.
External Test Data: Models tested on unlabeled external audio clips to assess real-world performance.
## Dataset
Source: Xeno-Canto bird sound archive.
Data Format: Preprocessed HDF5 files containing spectrograms.
Structure: Spectrograms of bird calls (343x256 pixels) for each species.
## Model Architecture
# Binary Classification:
Two CNN architectures with varying filters, pooling layers, and dropout rates.
Activation functions: ReLU for hidden layers, Sigmoid for output layer.
# Multi-Class Classification:
Multi-layer CNN with softmax activation for output.
Optimized for categorical cross-entropy loss.
## Performance Metrics
Binary Models:
Achieved 100% training and testing accuracy over 20 epochs.
Multi-Class Models:
Accuracy improved from 39% to 99.21% over 10 epochs.
External Test Data:
Predictions showed species classification probabilities, highlighting consistency.
## Technology Stack
Programming Language: Python
Libraries:
TensorFlow/Keras: Model training and evaluation.
Librosa: Spectrogram generation and preprocessing.
Matplotlib/Seaborn: Visualization.
## How to Use
Prerequisites:
Python 3.x installed.
Required libraries: tensorflow, librosa, h5py, matplotlib, numpy.
## Steps:
Preprocess bird call audio files into spectrograms.
Train CNN models for binary and multi-class classification.
Evaluate models using validation datasets and external test data.
## Results
Binary Classification:
Model 1: Accuracy reached 100% with reduced training loss of ~0.000001.
Model 2: Accuracy peaked at 100%, showcasing robustness to dropout changes.
Multi-Class Classification:
Model performance consistently improved, demonstrating high accuracy across classes.
External Validation:
Accurately predicted bird species probabilities with minimal errors.
## Key Challenges
Lack of labeled data for external test sets, limiting validation reliability.
Audio quality variations (e.g., background noise) affected predictions.
## Future Enhancements
Integrating additional audio preprocessing techniques to reduce noise impact.
Exploring advanced architectures like CRNNs or transfer learning models (e.g., VGGNet).
Expanding the dataset to include more bird species and geographical regions.
## Acknowledgments
Data Source: Xeno-Canto
## References:
Gareth James et al., An Introduction to Statistical Learning.
EdgeImpulse Documentation on Spectrogram Processing.
