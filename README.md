# Fashion-MNIST Classification using Convolutional Neural Networks (CNN)

<img src="images/image.png" />

## 1. Objective

The objective of this section is to develop a Convolutional Neural Network (CNN) to classify the clothing-articles using the widely used Fashion-MNIST dataset.

## 2. Motivation

The MNIST handwritten digit classification problem is a standard dataset used in computer vision and deep learning.

Since the MINIST dataset is effectively solved,  the Fashion-MNIST has recently been collected and labelled to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine and deep learning algorithms. The Fashion-MNIST dataset shares the same image size and structure of training and testing splits.as the original MNIST dataset

In this section, we shall demonstrate how to develop convolutional neural network for clothing items classification from scratch, using the Fashion-MNIST dataset, including:

* How to prepare the input training and test data 
* How to deploy the model
* How to use the trained model to make predictions
* How to evaluate its performance

## 3. Data

The Fashion-MNIST is a dataset of clothing articles images:
* It contains training set of 60,000 examples
* It contains a test set of 10,000 examples.
  * Each example is a 28x28 grayscale image
  * Each example is associated with a label from 10 classes.
* Recently, the Fashion-MNIST is starting to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine and deep learning algorithms.
  * It shares the same image size and structure of training and testing splits.
  * Additional detailed about the Fashion can be in [1]

Sample images from the MNIST data set are illustrated next:
  * There are significant variations between the different types of clothing articles
  * There are significant variations between different examples of the same clothing article class.
