# Artificial Neural Network to Recognise Images using Keras & TensorFlow

This repository contains an implementation of an Artificial Neural Network (ANN) for image recognition tasks using Keras and TensorFlow. The project demonstrates the power of deep learning techniques in solving image classification problems.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Overview](#project-overview)
- [Results](#results)

---

## Introduction

Image recognition is a fundamental task in computer vision with applications ranging from object detection to facial recognition. This project showcases an ANN model built with Keras and TensorFlow to classify images effectively. The project aims to provide a simple yet powerful baseline for image recognition tasks.

---

## Features

- **Customizable Neural Network Architecture**: Modify the architecture to suit your dataset.
- **Comprehensive Preprocessing**: Includes image normalization and augmentation techniques.
- **Visualization Tools**: Provides tools to visualize training metrics and model performance.
- **Modular Code Structure**: Easy-to-understand and extend for your specific needs.

---

## Technologies Used

This project leverages the following technologies and libraries:

- **Python**: The primary programming language for the project.
- **Jupyter Notebook**: For interactive development and visualization.
- **Keras**: High-level neural networks API.
- **TensorFlow**: Backend library for deep learning.
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization.

---

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/LovedeepRajpurohit/Artificial-Neural-Network-to-Recognise-Images-using-Keras-Tensorflow.git
   cd Artificial-Neural-Network-to-Recognise-Images-using-Keras-Tensorflow
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare Your Dataset**:
   - Place your dataset in the `data/` directory or specify the path in the code.
   - Ensure your dataset is organized into separate folders for each class.

2. **Run the Jupyter Notebook**:
   - Open the `ANN_Image_Recognition.ipynb` notebook in Jupyter.
   - Execute the cells to preprocess data, train the model, and evaluate its performance.

3. **Customize the Model**:
   - Modify the architecture in the code to experiment with different layer configurations.

4. **Visualize Results**:
   - Utilize provided visualization tools to analyze training and testing results.

---

## Project Overview

### Dataset
The project is designed to be flexible with datasets. Users can train the model on any image dataset by organizing it into appropriate folders.

### Model Architecture
The ANN consists of multiple dense layers with ReLU activations and dropout for regularization. The final layer uses a softmax activation for multi-class classification.

### Training
The model is trained using the Adam optimizer and categorical crossentropy loss. Key metrics like accuracy and loss are tracked during training.

### Evaluation
The trained model is evaluated on a separate test set, and performance metrics are displayed.

---

## Results

The performance of the model is evaluated on various metrics such as:

- **Accuracy**: The percentage of correctly classified images.
- **Loss**: The difference between predicted and actual outputs.
- **Confusion Matrix**: A detailed breakdown of classification results.

Example results:
- Training Accuracy: 95%
- Validation Accuracy: 92%

Graphs of training and validation accuracy/loss over epochs are included in the notebook.

---

Enjoy using and experimenting with this project to enhance your understanding of image recognition with ANNs!
