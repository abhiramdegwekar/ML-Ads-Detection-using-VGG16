# Ads Classification using Transfer Learning with VGG16

This repository contains an image classification project that utilizes the VGG16 model with transfer learning. The project aims to classify images to determine whether they are advertisements or not by leveraging the pre-trained VGG16 model.
!["ad"](/images/ad.jpg)

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction

Transfer learning is a machine learning technique where a model developed for a task is reused as the starting point for a model on a second task. This project uses the VGG16 model, pre-trained on the ImageNet dataset, and fine-tunes it for the task of classifying images as advertisements or non-advertisements.

## Dataset


The model is trained on a total of 10,000 images. The dataset was obtained from [Open Data Lab](https://opendatalab.com/) and consists of images categorized into two classes: Ads and NonAds.

- **Training Data**: The model is trained on 8,000 images of Ads and 8,000 images of NonAds.
- **Testing Data**: 2,000 images of Ads and 2,000 images of NonAds are used as testing data.

- **Dataset Description**: The dataset used for this project contains images categorized into two classes: ads and non-ads. Include details about the number of images in each class and the source of the dataset.
- **Data Preprocessing**: Images are resized to the input size required by VGG16 (224x224 pixels), normalized, and augmented using techniques such as rotation, flipping, and zooming to enhance the model's robustness.

You can find the data here : [Drive](https://drive.google.com/drive/folders/1rDCScSBFYpNL0npV9gxpmBTQZQ-RV0MZ?usp=sharing)

## Model Architecture

The VGG16 model is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition". 

### Transfer Learning

- **Base Model**: VGG16 pre-trained on ImageNet.
- **Fine-tuning**: The top layers of the VGG16 model are replaced with custom fully connected layers to adapt the model for the new classification task.
- **Architecture Modifications**: The modified architecture includes dropout layers to prevent overfitting and a final dense layer with a sigmoid activation function for binary classification.

# Installation

Follow these steps to set up the project on your local machine.

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

# Usage

This project includes two Jupyter notebooks: one for training the model and the other for loading the weights and predicting on random images.

## Training the Model

To train the model, follow these steps:

1. Open the notebook for training:

    ```bash
    ads-vgg16-transferlearning.ipynb
    ```

2. Execute the cells in the notebook to train the model on your dataset. This notebook includes data preprocessing, model architecture setup, and training steps.

## Predicting on New Images

To predict on new images using the trained model, follow these steps:

1. Open the Jupyter notebook for prediction:

    ```bash
    predict_vgg.ipynb
    ```

2. Execute the cells in the notebook to load the trained model weights and make predictions on random images. This notebook includes loading the model, preparing the input images, and displaying the prediction results.


# Results
The accuracy of the model is **78%**, here is the confusion matrix.

!["cm"](/results/cm.png)
