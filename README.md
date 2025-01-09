# ANN-Classification-with-Pytorch
# CIFAR-10 Image Classification with PyTorch

## Overview
This repository contains the code for building, training, and evaluating an Artificial Neural Network (ANN) to classify images from the CIFAR-10 dataset using PyTorch. The model is designed with at least 5 hidden layers and uses ReLU activation functions. The project includes data preprocessing, model design, training, evaluation, and visualization of results.

## Table of Contents
- [Overview](#overview)
- [Dataset Preparation](#dataset-preparation)
- [Model Design](#model-design)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results Visualization](#results-visualization)
- [Requirements](#requirements)
- [Usage](#usage)
- [Report](#report)
- [License](#license)

## Dataset Preparation
1. Load and preprocess the CIFAR-10 dataset.
2. Normalize the data (mean = 0, std = 1) for faster convergence.
3. Split the dataset into training and testing sets.

## Model Design
1. Construct an Artificial Neural Network with at least 5 hidden layers.
2. Use the ReLU activation function for hidden layers.
3. Include a final output layer with 10 neurons and no activation (for classification).
4. Use CrossEntropyLoss as the loss function.

## Model Training
1. Train the model for 50 epochs using the Adam optimizer.
2. Use a mini-batch size of 64.

## Model Evaluation
1. Evaluate the model on the testing set.
2. Plot the following:
   - Training and validation loss curves.
   - Confusion matrix for test set predictions.
   - Examples of misclassified images with their predicted and true labels.

## Results Visualization
- Training and validation loss curves.
- Confusion matrix.
- Misclassified image samples.

## Requirements
- Python 3.7 or higher
- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cifar10-classification.git
   cd cifar10-classification
